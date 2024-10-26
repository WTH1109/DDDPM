import torch
import loralib as lora
from einops import rearrange
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingWarmRestarts
from ldm.util import default
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
from taming.modules.losses.vqperceptual import vanilla_d_loss


class DecouplingDDPM(LatentDiffusion):
    def __init__(self, first_stage_config, cond_stage_config, apply_lora=False, flow_weight=1.0,
                 d_weight=1.0, *args, **kwargs):

        super().__init__(first_stage_config, cond_stage_config, *args, **kwargs)

        self.apply_lora = apply_lora

        self.disc_loss = vanilla_d_loss

        self.discriminator = NLayerDiscriminator(input_nc=1,
                                                 n_layers=3,
                                                 use_actnorm=False
                                                 ).apply(weights_init)

        self.flow_weight = flow_weight

        self.d_weight = d_weight

        if self.apply_lora:
            lora.mark_only_lora_as_trainable(self.cond_stage_model)



    def training_step(self, batch, batch_idx, optimizer_idx):
        # print(batch['contrast'].shape)
        loss, loss_dict = self.shared_step_gan(batch, optimizer_idx)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True, batch_size=self.batch_size)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False, batch_size=self.batch_size)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, batch_size=self.batch_size)

        return loss

    def get_input_ori(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step_gan(self, batch, optimizer_idx, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key)
        c_ori_image = self.get_input_ori(batch, self.first_stage_key)
        nll_loss, loss_dic = self(x, c)
        xd = self.first_stage_model.decode(x)
        prefix = 'train' if self.training else 'val'
        if optimizer_idx == 0:
            # opt dis: fake -> -1 true -> 1
            # opt generate: let fake to 1
            logit_fake = self.discriminator(xd.contiguous())
            g_loss = -torch.mean(logit_fake)
            loss = nll_loss + self.d_weight * g_loss
            # loss = nll_loss

            loss_dic.update({f'{prefix}/g_loss': g_loss})
            loss_dic.update({f'{prefix}/final_loss': loss})
        else:
            logit_real = self.discriminator(c_ori_image.contiguous().detach())
            logit_fake = self.discriminator(xd.contiguous().detach())
            d_loss = self.disc_loss(logit_real, logit_fake)
            loss = d_loss
            loss_dic.update({f'{prefix}/d_loss': d_loss})

        return loss, loss_dic
    def g_losses(self, x_start, cond, t, noise=None):
        deformable_feature_x = x_start[:, -self.first_stage_model.deformable_dim:, :, :]
        deformable_feature_cond = cond[:, -self.first_stage_model.deformable_dim:, :, :]
        z_deformable = torch.cat((deformable_feature_x, deformable_feature_cond), dim=1)

        x_image = self.first_stage_model.decode(x_start)

        flow_pred, target_pred, z_represent = self.first_stage_model.flow_pred(z_deformable, x_image)

        flow_loss = torch.abs(torch.mean(flow_pred))

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        self.logvar = self.logvar.to(self.device)
        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t + flow_loss * self.flow_weight
        # loss = loss_simple / torch.exp(logvar_t) + logvar_t
        loss_dict.update({f'{prefix}/flow_loss': flow_loss * self.flow_weight})
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)

            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.g_losses(x, c, t, *args, **kwargs)

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        disc_opt = torch.optim.Adam(self.discriminator.parameters(), lr=lr)

        # self.use_scheduler = True

        # if self.use_scheduler:
        #     # assert 'target' in self.scheduler_config
        #     # scheduler = instantiate_from_config(self.scheduler_config)
        #
        #     # print("Setting up LambdaLR scheduler...")
        #     scheduler = [
        #         {
        #             # 'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
        #             'scheduler': CosineAnnealingWarmRestarts(opt, T_0=1, T_mult=2, eta_min=0, last_epoch=-1),
        #             'interval': 'step',
        #             'frequency': 1
        #         },
        #         # {
        #         #     'scheduler': LambdaLR(disc_opt, lr_lambda=scheduler.schedule),
        #         #     'interval': 'step',
        #         #     'frequency': 1
        #         # }
        #     ]
        # scheduler = [
        #     {
        #         # 'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
        #         'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=50),
        #         'interval': 'step',
        #         'monitor': "train/loss_simple",
        #         'frequency': 1
        #     }
        # ]
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, 0.8, last_epoch=-1)

        # return [opt, disc_opt], scheduler
        return [opt, disc_opt], scheduler
