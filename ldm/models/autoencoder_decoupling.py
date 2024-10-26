import os

import numpy as np
import random

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from contextlib import contextmanager

from ldm.modules.deformable.deformable_net import DeformableNet, SpatialTransformer, estimate_mixture_flow
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

from ldm.modules.diffusionmodules.model import Encoder, Decoder
from ldm.my_lora import dynamic_conv2d
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution, DiagonalGaussianDistributionInvariance

from ldm.util import instantiate_from_config

import loralib as lora


class VQModel(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False,  # tell vector quantizer to return indices as bhw
                 use_ema=False
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def encode_to_prequant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b):
        quant_b = self.quantize.embed_code(code_b)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_, _, ind) = self.encode(input)
        dec = self.decode(quant)
        if return_pred_indices:
            return dec, diff, ind
        return dec, diff

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size + 16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # https://github.com/pytorch/pytorch/issues/37142
        # try not to fool the heuristics
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencode
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train",
                                            predicted_indices=ind)

            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return aeloss

        if optimizer_idx == 1:
            # discriminator
            discloss, log_dict_disc = self.loss(qloss, x, xrec, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="val" + suffix,
                                        predicted_indices=ind
                                        )

        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val" + suffix,
                                            predicted_indices=ind
                                            )
        rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
        self.log(f"val{suffix}/rec_loss", rec_loss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"val{suffix}/aeloss", aeloss,
                 prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        if version.parse(pl.__version__) >= version.parse('1.4.0'):
            del log_dict_ae[f"val{suffix}/rec_loss"]
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor * self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quantize.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class VQModelInterface(VQModel):
    def __init__(self, embed_dim, *args, **kwargs):
        super().__init__(embed_dim=embed_dim, *args, **kwargs)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h

    def decode(self, h, force_not_quantize=False):
        # also go through quantization layer
        if not force_not_quantize:
            quant, emb_loss, info = self.quantize(h)
        else:
            quant = h
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec


class AutoencoderKL(pl.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 apply_lora=False,
                 lora_r=0,
                 lora_alpha=1,
                 lora_dropout=0,
                 ):
        super().__init__()
        self.image_key = image_key
        self.apply_lora = apply_lora

        lora_config = {
            'r': lora_r,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout,
        }

        encoder_config = ddconfig
        decoder_config = ddconfig
        encoder_config.update({'apply_lora': apply_lora})
        encoder_config.update({'lora_config': lora_config})

        self.encoder = Encoder(**encoder_config)
        self.decoder = Decoder(**decoder_config)
        self.loss = instantiate_from_config(lossconfig)
        assert ddconfig["double_z"]

        self.quant_conv = dynamic_conv2d(apply_lora, lora_config, 2 * ddconfig["z_channels"], 2 * embed_dim, 1)

        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        self.embed_dim = embed_dim
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def dic_convert_lora(self, sd):
        new_dic = {}
        for key, value in sd.items():
            parts = key.split('.')
            if len(parts) >= 2:
                parts.insert(-1, 'conv')
            new_key = '.'.join(parts)
            new_dic[new_key] = value
            new_dic[key] = value
        return new_dic


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        if self.apply_lora:
            sd = self.dic_convert_lora(sd)
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)

        if missing_keys:
            print("Missing keys:")
            print(missing_keys)

        # if unexpected_keys:
        #     print("Unexpected keys:")
        #     print(unexpected_keys)

        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
                                                last_layer=self.get_last_layer(), split="train")

            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    def validation_step(self, batch, batch_idx):
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val")

        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val")

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x


class IdentityFirstStage(torch.nn.Module):
    def __init__(self, *args, vq_interface=False, **kwargs):
        self.vq_interface = vq_interface  # TODO: Should be true by default but check to not break older stuff
        super().__init__()

    def encode(self, x, *args, **kwargs):
        return x

    def decode(self, x, *args, **kwargs):
        return x

    def quantize(self, x, *args, **kwargs):
        if self.vq_interface:
            return x, None, [None, None, None]
        return x

    def forward(self, x, *args, **kwargs):
        return x


class AutoencoderKLInvariance(AutoencoderKL):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 source_key="source",
                 target_key="target",
                 invariance_dim=2,
                 *args, **kwargs
                 ):
        super().__init__(ddconfig,
                         lossconfig,
                         embed_dim,
                         *args, **kwargs
                         )
        self.source_key = source_key
        self.target_key = target_key
        self.invariance_dim = invariance_dim

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistributionInvariance(moments)
        return posterior

    def validation_step(self, batch, batch_idx):
        inputs_source = self.get_input(batch, self.source_key)
        inputs_target = self.get_input(batch, self.target_key)
        reconstructions_source, posterior_source = self(inputs_source)
        reconstructions_target, posterior_target = self(inputs_target)

        loss_invariance = posterior_source.kl_invariance(other=posterior_target, invariance_dim=self.invariance_dim)
        loss_invariance = torch.sum(loss_invariance) / loss_invariance.shape[0]

        aeloss_source, log_dict_ae_source = self.loss(inputs_source, reconstructions_source, posterior_source,
                                                      0, self.global_step,
                                                      last_layer=self.get_last_layer(), split="val")
        aeloss_target, log_dict_ae_target = self.loss(inputs_target, reconstructions_target, posterior_target,
                                                      0, self.global_step,
                                                      last_layer=self.get_last_layer(), split="val")
        aeloss = aeloss_source + aeloss_target + loss_invariance

        discloss_source, log_dict_disc_source = self.loss(inputs_source, reconstructions_source, posterior_source,
                                                          1, self.global_step,
                                                          last_layer=self.get_last_layer(), split="val")
        discloss_target, log_dict_disc_target = self.loss(inputs_source, reconstructions_source, posterior_source,
                                                          1, self.global_step,
                                                          last_layer=self.get_last_layer(), split="val")
        discloss = discloss_source + discloss_target

        self.log("val/rec_loss", log_dict_ae_source["val/rec_loss"])
        self.log("val/loss_invariance", loss_invariance)
        self.log("val/ae_loss", aeloss)
        self.log_dict(log_dict_ae_source)
        self.log_dict(log_dict_disc_source)
        return self.log_dict

    def training_step(self, batch, batch_idx, optimizer_idx):
        inputs_source = self.get_input(batch, self.source_key)
        inputs_target = self.get_input(batch, self.target_key)
        reconstructions_source, posterior_source = self(inputs_source)
        reconstructions_target, posterior_target = self(inputs_target)

        loss_invariance = posterior_source.kl_invariance(other=posterior_target, invariance_dim=self.invariance_dim)
        loss_invariance = torch.sum(loss_invariance) / loss_invariance.shape[0]

        if optimizer_idx == 0:
            # train encoder+decoder+logvar
            aeloss_source, log_dict_ae_source = self.loss(inputs_source, reconstructions_source, posterior_source,
                                                          optimizer_idx, self.global_step,
                                                          last_layer=self.get_last_layer(), split="train")
            aeloss_target, log_dict_ae_target = self.loss(inputs_target, reconstructions_target, posterior_target,
                                                          optimizer_idx, self.global_step,
                                                          last_layer=self.get_last_layer(), split="train")
            aeloss = aeloss_source + aeloss_target + loss_invariance
            self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log("loss_invariance", loss_invariance, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_ae_source, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss_source, log_dict_disc_source = self.loss(inputs_source, reconstructions_source, posterior_source,
                                                              optimizer_idx, self.global_step,
                                                              last_layer=self.get_last_layer(), split="train")
            discloss_target, log_dict_disc_target = self.loss(inputs_source, reconstructions_source, posterior_source,
                                                              optimizer_idx, self.global_step,
                                                              last_layer=self.get_last_layer(), split="train")
            discloss = discloss_source + discloss_target
            self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
            self.log_dict(log_dict_disc_source, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            return discloss

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.source_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples_source"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions_source"] = xrec
        log["inputs_source"] = x

        y = self.get_input(batch, self.target_key)
        y = y.to(self.device)
        if not only_inputs:
            yrec, posteriory = self(y)
            if y.shape[1] > 3:
                # colorize with random projection
                assert yrec.shape[1] > 3
                y = self.to_rgb(y)
                yrec = self.to_rgb(yrec)
            log["samples_target"] = self.decode(torch.randn_like(posteriory.sample()))
            log["reconstructions_target"] = yrec
        log["inputs_target"] = y

        return log


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()


class AutoencoderDecoupling(AutoencoderKL):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 embed_dim,
                 deformable_config,
                 first_stage_step,
                 second_stage_step,
                 ckpt_path=None,
                 ignore_keys=[],
                 size=256,
                 contrast_num=5,
                 source_key="source",
                 target_key="target",
                 wrap_key='wrap',
                 invariance_dim=2,
                 deformable_dim=1,
                 wrap_strength=4,
                 flow_weight=1,
                 flow_pred_weight=0.1,
                 flow_rec_weight=1,
                 smooth_weight=1,
                 flow_zero_prob=0.3,
                 ae_weight=1,
                 if_modify_input=False,
                 cond_contrast_p=0.5,
                 cond_mode='default',
                 *args, **kwargs
                 ):
        super().__init__(ddconfig,
                         lossconfig,
                         embed_dim,
                         *args, **kwargs
                         )
        self.source_key = source_key
        self.target_key = target_key
        self.wrap_key = wrap_key
        self.invariance_dim = invariance_dim
        self.deformable_dim = deformable_dim
        self.Deformable = DeformableNet(**deformable_config)
        self.Transformer = SpatialTransformer((size, size))
        self.first_stage_step = first_stage_step
        self.second_stage_step = second_stage_step
        self.automatic_optimization = False
        self.contrast_num = contrast_num
        self.wrap_strength = wrap_strength
        self.flow_weight = flow_weight
        self.flow_pred_weight = flow_pred_weight
        self.flow_rec_weight = flow_rec_weight
        self.smooth_weight = smooth_weight
        self.ae_weight = ae_weight
        self.flow_zero_prob = flow_zero_prob
        self.smooth_loss = Grad(penalty='l2').loss
        self.size = size
        self.if_modify_input = if_modify_input
        self.cond_contrast_target = 'target'
        self.cond_contrast_p = cond_contrast_p
        self.cond_mode = cond_mode

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.test = 1

    def auto_expand_dim(self, x):
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        opt_flow = torch.optim.Adam(list(self.encoder.parameters()) +
                                    list(self.decoder.parameters()) +
                                    list(self.quant_conv.parameters()) +
                                    list(self.post_quant_conv.parameters()) +
                                    list(self.Deformable.parameters()),
                                    lr=lr, betas=(0.5, 0.9))
        return [opt_ae, opt_disc, opt_flow], []

    def flow_pred(self, z, y_source):
        flow, y_target, z_represent = self.Deformable(z, y_source)
        return flow, y_target, z_represent

    def modify_input(self, batch):
        inputs_source = self.get_input(batch, self.source_key)
        if self.cond_contrast_target == 'target':
            inputs_target = self.get_input(batch, self.target_key)
        elif self.cond_contrast_target == 'source':
            inputs_target = self.get_input(batch, self.source_key)
        else:
            raise ValueError(f'cond_contrast_target is {self.cond_contrast_target}, not in target or source')

        # if_amp = False
        #
        # if inputs_source.dtype == torch.float16:
        #     if_amp = True
        #     inputs_source = inputs_source.to(torch.float32)
        #     inputs_target = inputs_target.to(torch.float32)

        flow_idx = random.randint(0, self.contrast_num - 1)

        # At the first let the source and target image pass the network
        self.eval()
        with torch.no_grad():
            slice_data = batch[self.wrap_key][flow_idx]
            inputs_source_other_slice = slice_data[self.source_key]
            inputs_target_other_slice = slice_data[self.target_key]

            inputs_source_other_slice = self.auto_expand_dim(inputs_source_other_slice)
            inputs_target_other_slice = self.auto_expand_dim(inputs_target_other_slice)

            # if inputs_source_other_slice.dtype == torch.float16:
            #     inputs_source_other_slice = inputs_source_other_slice.to(torch.float32)
            #     inputs_target_other_slice = inputs_target_other_slice.to(torch.float32)

            get_init_flow_value_dic = self(inputs_source_other_slice, inputs_target_other_slice, stage='deformable')
            flow = get_init_flow_value_dic['flow_pred'].detach()

            # Then Add flow on the target image, let deformable network learn the transform
            random_number = random.uniform(-self.wrap_strength, self.wrap_strength)
            flow = flow * random_number / 5
            random_p = random.random()
            if random_p < self.flow_zero_prob:
                correction_flow = flow * 0.0
            else:

                inputs_pre_target = self.Transformer(inputs_source, flow)
                value_dic_pre = self(inputs_source, inputs_pre_target, stage='deformable')
                correction_flow = value_dic_pre['flow_pred'].detach()

            inputs_target = self.Transformer(inputs_target, correction_flow).detach()
            inputs_source_modify = self.Transformer(inputs_source, correction_flow).detach()
        if inputs_target.dim() == 4:
            inputs_target = torch.squeeze(inputs_target, dim=1)
            inputs_source_modify = torch.squeeze(inputs_source_modify, dim=1)
        batch[self.target_key] = inputs_target
        batch['source_modify'] = inputs_source_modify
        batch['flow'] = correction_flow
        # if if_amp:
        #     batch[self.target_key] = batch[self.target_key].to(torch.float16)
        return batch

    def generate_cond(self, cond_dic):

        if self.cond_mode == 'default':
            inputs_source = self.get_input(cond_dic, self.source_key)

            random_p = random.random()
            if random_p < self.cond_contrast_p:
                inputs_target = self.get_input(cond_dic, self.target_key)
            else:
                inputs_target = self.get_input(cond_dic, 'source_modify')

            value_dic = self(inputs_source, inputs_target, stage='deformable')
            z_source = value_dic['z_source']
            z_represent = value_dic['z_represent']
            z_cond = torch.cat((z_source, z_represent), dim=1)
            return z_cond
        elif self.cond_mode == 'represent_zero':
            inputs_source = self.get_input(cond_dic, self.source_key)
            inputs_target = self.get_input(cond_dic, self.target_key)
            value_dic = self(inputs_source, inputs_target, stage='deformable')
            z_source = value_dic['z_source']
            z_represent = value_dic['z_represent']
            z_represent = z_represent * 0
            z_cond = torch.cat((z_source, z_represent), dim=1)
            return z_cond
        elif self.cond_mode == 'represent_None':
            inputs_source = self.get_input(cond_dic, self.source_key)
            inputs_target = self.get_input(cond_dic, self.target_key)
            value_dic = self(inputs_source, inputs_target, stage='deformable')
            z_source = value_dic['z_source']
            z_cond = z_source
            return z_cond
        elif self.cond_mode == 'sample':
            inputs_source = self.get_input(cond_dic, self.source_key)
            inputs_target = self.get_input(cond_dic, self.source_key)

            value_dic = self(inputs_source, inputs_target, stage='deformable')
            z_source = value_dic['z_source']
            z_represent = value_dic['z_represent']
            z_cond = torch.cat((z_source, z_represent), dim=1)
            return z_cond
        elif self.cond_mode == 'default_slice_guide':
            inputs_source = self.get_input(cond_dic, self.source_key)

            random_p = random.random()
            if random_p < self.cond_contrast_p:
                inputs_target = self.get_input(cond_dic, self.target_key)
            else:
                inputs_target = self.get_input(cond_dic, 'source_modify')

            value_dic = self(inputs_source, inputs_target, stage='deformable')
            z_source = value_dic['z_source']
            z_represent = value_dic['z_represent']
            z_cond = torch.cat((z_source, z_represent), dim=1)

            tmp_slice_idx = cond_dic['slice']
            total_slice_idx = cond_dic['slice_num']

            class_slice = int(tmp_slice_idx / total_slice_idx * 100) + 1

            out_cond_dic = {
                'c_concat': [z_cond],
                'c_y': [class_slice]
            }

            return z_cond

    # stage: {ae, contrast}
    def forward(self, source, target, sample_posterior=True, stage='deformable'):
        if stage == 'ae':
            posterior_source = self.encode(source)
            if sample_posterior:
                z_source = posterior_source.sample()
            else:
                z_source = posterior_source.mode()
            dec_source = self.decode(z_source)
            value_dic = {
                'dec_source': dec_source,
                'posterior_source': posterior_source,
                'z_source': z_source,
            }
        elif stage == 'deformable':
            posterior_source = self.encode(source)
            posterior_target = self.encode(target)
            if sample_posterior:
                z_source = posterior_source.sample()
                z_target = posterior_target.sample()
            else:
                z_source = posterior_source.mode()
                z_target = posterior_target.mode()
            dec_source = self.decode(z_source)
            dec_target = self.decode(z_target)
            z_source_deformable = z_source[:, -self.deformable_dim:, :, :]
            z_target_deformable = z_target[:, -self.deformable_dim:, :, :]
            z_deformable = torch.cat((z_source_deformable, z_target_deformable), dim=1)
            flow_pred, target_pred, z_represent = self.flow_pred(z_deformable, source)
            value_dic = {
                'dec_source': dec_source,
                'dec_target': dec_target,
                'posterior_source': posterior_source,
                'posterior_target': posterior_target,
                'z_source': z_source,
                'z_target': z_target,
                'flow_pred': flow_pred,
                'target_pred': target_pred,
                'z_represent': z_represent,
            }
        else:
            value_dic = None
        return value_dic

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistributionInvariance(moments)
        return posterior

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.source_key)
        x = x.to(self.device)
        if not only_inputs:
            value_dic = self(x, None, stage='ae')
            xrec = value_dic['dec_source']
            posterior = value_dic['posterior_source']
            # z_sample = value_dic['z_source']
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples_source"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions_source"] = xrec
        log["inputs_source"] = x

        y = self.get_input(batch, self.target_key)
        y = y.to(self.device)
        if not only_inputs:
            value_dic = self(y, None, stage='ae')
            yrec = value_dic['dec_source']
            posterior = value_dic['posterior_source']
            if y.shape[1] > 3:
                # colorize with random projection
                assert yrec.shape[1] > 3
                y = self.to_rgb(y)
                yrec = self.to_rgb(yrec)
            log["samples_target"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions_target"] = yrec
        log["inputs_target"] = y

        return log

    def train_ae_generate(self, batch):
        opt_ae, opt_disc, opt_flow = self.optimizers()
        inputs_source = self.get_input(batch, self.source_key)
        inputs_target = self.get_input(batch, self.target_key)
        value_dic = self(inputs_source, inputs_target, stage='deformable')
        reconstructions_source = value_dic['dec_source']
        reconstructions_target = value_dic['dec_target']
        posterior_source = value_dic['posterior_source']
        posterior_target = value_dic['posterior_target']
        loss_invariance = posterior_source.kl_invariance(other=posterior_target, invariance_dim=self.invariance_dim)
        # train encoder+decoder+logvar
        aeloss_source, log_dict_ae_source = self.loss(inputs_source, reconstructions_source, posterior_source,
                                                      0, self.global_step,
                                                      last_layer=self.get_last_layer(), split="train")
        aeloss_target, log_dict_ae_target = self.loss(inputs_target, reconstructions_target, posterior_target,
                                                      0, self.global_step,
                                                      last_layer=self.get_last_layer(), split="train")
        loss_invariance = torch.mean(loss_invariance)
        aeloss = (aeloss_source + aeloss_target) * self.ae_weight + loss_invariance
        self.log("ae/aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("ae/loss_invariance", loss_invariance, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae_source, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.manual_backward(aeloss)
        opt_ae.step()

    def train_ae_dis(self, batch):
        opt_ae, opt_disc, opt_flow = self.optimizers()
        inputs_source = self.get_input(batch, self.source_key)
        inputs_target = self.get_input(batch, self.target_key)
        value_dic = self(inputs_source, inputs_target, stage='deformable')
        reconstructions_source = value_dic['dec_source']
        reconstructions_target = value_dic['dec_target']
        posterior_source = value_dic['posterior_source']
        posterior_target = value_dic['posterior_target']
        # loss_invariance = posterior_source.kl_invariance(other=posterior_target, invariance_dim=self.invariance_dim)

        # train the discriminator
        discloss_source, log_dict_disc_source = self.loss(inputs_source, reconstructions_source,
                                                          posterior_source, 1, self.global_step,
                                                          last_layer=self.get_last_layer(), split="train")
        discloss_target, log_dict_disc_target = self.loss(inputs_target, reconstructions_target,
                                                          posterior_target,
                                                          1, self.global_step,
                                                          last_layer=self.get_last_layer(), split="train")
        discloss = discloss_source + discloss_target
        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc_source, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.manual_backward(discloss)
        opt_disc.step()

    def train_self_deformable(self, batch, flow_idx):
        opt_ae, opt_disc, opt_flow = self.optimizers()
        inputs_source = self.get_input(batch, self.source_key)

        slice_data = batch[self.wrap_key][flow_idx]
        inputs_source_other_slice = slice_data[self.source_key]
        inputs_target_other_slice = slice_data[self.target_key]

        inputs_source_other_slice = self.auto_expand_dim(inputs_source_other_slice)
        inputs_target_other_slice = self.auto_expand_dim(inputs_target_other_slice)

        self.eval()

        with torch.no_grad():
            get_init_flow_value_dic = self(inputs_source_other_slice, inputs_target_other_slice, stage='deformable')
            flow = get_init_flow_value_dic['flow_pred'].detach()

            # flow_data = batch[self.wrap_key]

            random_number = random.uniform(-self.wrap_strength, self.wrap_strength)
            # flow = flow_data[:, flow_idx, :, :, :] * random_number
            # flow = flow.float()
            # flow = flow.to(self.device)
            flow = flow * random_number

            random_p = random.random()
            if random_p < self.flow_zero_prob:
                flow = flow * 0.0

            inputs_pre_target = self.Transformer(inputs_source, flow)
            value_dic_pre = self(inputs_source, inputs_pre_target, stage='deformable')
            correction_flow = value_dic_pre['flow_pred'].detach()

            inputs_target = self.Transformer(inputs_source, correction_flow).detach()

        self.train()

        value_dic = self(inputs_source, inputs_target, stage='deformable')

        reconstructions_source = value_dic['dec_source']
        posterior_source = value_dic['posterior_source']
        posterior_target = value_dic['posterior_target']
        flow_pred = value_dic['flow_pred']
        target_pred = value_dic['target_pred']

        aeloss_source, log_dict_ae_source = self.loss(inputs_source, reconstructions_source, posterior_source,
                                                      0, self.global_step,
                                                      last_layer=self.get_last_layer(), split="train")

        loss_invariance = torch.mean(
            posterior_source.kl_invariance(other=posterior_target, invariance_dim=self.invariance_dim))

        # flow loss
        flow_pred_loss = torch.nn.functional.mse_loss(correction_flow, flow_pred)
        rec_flow_loss = torch.nn.functional.mse_loss(inputs_target, target_pred)
        smooth_flow_loss = self.smooth_loss(flow_pred)
        flow_loss = (flow_pred_loss * self.flow_pred_weight + rec_flow_loss * self.flow_rec_weight
                     + smooth_flow_loss * self.smooth_weight)

        z_loss = loss_invariance + flow_loss * self.flow_weight

        total_loss = z_loss + aeloss_source * self.ae_weight

        self.log("self/total_loss_s", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("self/z_loss_s", z_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("self/loss_invariance_s", loss_invariance, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("self/flow_loss_s", flow_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("self/flow_pred_loss_s", flow_pred_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("self/rec_flow_loss_s", rec_flow_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("self/smooth_flow_loss_s", smooth_flow_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("self/aeloss_source_s", aeloss_source, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae_source, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.manual_backward(total_loss)
        opt_flow.step()

    # 预训练模型产生flow能力
    def pre_train_deformable(self, batch):
        opt_ae, opt_disc, opt_flow = self.optimizers()
        inputs_source = self.get_input(batch, self.source_key)
        inputs_target = self.get_input(batch, self.target_key)
        # flow_data = batch[self.wrap_key]

        value_dic = self(inputs_source, inputs_target, stage='deformable')
        reconstructions_source = value_dic['dec_source']
        reconstructions_target = value_dic['dec_target']
        posterior_source = value_dic['posterior_source']
        posterior_target = value_dic['posterior_target']
        flow_pred = value_dic['flow_pred']
        target_pred = value_dic['target_pred']

        aeloss_source, log_dict_ae_source = self.loss(inputs_source, reconstructions_source, posterior_source,
                                                      0, self.global_step,
                                                      last_layer=self.get_last_layer(), split="train")

        aeloss_target, log_dict_ae_target = self.loss(inputs_target, reconstructions_target, posterior_target,
                                                      0, self.global_step,
                                                      last_layer=self.get_last_layer(), split="train")

        loss_invariance = torch.mean(
            posterior_source.kl_invariance(other=posterior_target, invariance_dim=self.invariance_dim))

        # flow loss
        # flow_pred_loss = torch.nn.functional.mse_loss(flow, flow_pred)
        rec_flow_loss = torch.nn.functional.mse_loss(inputs_target, target_pred)
        smooth_flow_loss = self.smooth_loss(flow_pred)
        flow_loss = (rec_flow_loss * self.flow_rec_weight
                     + smooth_flow_loss * self.smooth_weight)

        z_loss = loss_invariance + flow_loss * self.flow_weight

        total_loss = z_loss + (aeloss_source + aeloss_target) / 2 * self.ae_weight

        self.log("first/total_loss_s", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("first/z_loss_s", z_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("first/loss_invariance_s", loss_invariance, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("first/flow_loss_s", flow_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("first/rec_flow_loss_s", rec_flow_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("first/smooth_flow_loss_s", smooth_flow_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("first/aeloss_source_s", aeloss_source, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae_source, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.manual_backward(total_loss)
        opt_flow.step()

    # 用低质量flow训练模型产生高质量flow
    def pre_train_self_deformable(self, batch, flow_idx):
        opt_ae, opt_disc, opt_flow = self.optimizers()
        inputs_source = self.get_input(batch, self.source_key)

        random_p_input = random.random()

        if random_p_input < 0.5:
            inputs_target = self.get_input(batch, self.target_key)
        else:
            inputs_target = self.get_input(batch, self.source_key)

        slice_data = batch[self.wrap_key][flow_idx]
        inputs_source_other_slice = slice_data[self.source_key]
        inputs_target_other_slice = slice_data[self.target_key]

        inputs_source_other_slice = self.auto_expand_dim(inputs_source_other_slice)
        inputs_target_other_slice = self.auto_expand_dim(inputs_target_other_slice)

        self.eval()
        with torch.no_grad():
            get_init_flow_value_dic = self(inputs_source_other_slice, inputs_target_other_slice, stage='deformable')
            flow = get_init_flow_value_dic['flow_pred'].detach()

            # flow_data = batch[self.wrap_key]

            random_number = random.uniform(-self.wrap_strength, self.wrap_strength)
            # flow = flow_data[:, flow_idx, :, :, :] * random_number
            # flow = flow.float()
            # flow = flow.to(self.device)

            # 对每个 b 维度进行chw归一化到0-1
            abs_max_vals = torch.abs(flow).view(flow.size(0), -1).max(dim=1, keepdim=True)[0].view(flow.size(0), 1, 1,
                                                                                                   1)
            # 归一化到0-1范围
            flow = flow / (torch.abs(abs_max_vals) + 1e-6)
            flow = flow * random_number
            random_p = random.random()
            if random_p < self.flow_zero_prob:
                flow = flow * 0.0
            inputs_target = self.Transformer(inputs_target, flow).detach()

        self.train()

        value_dic = self(inputs_source, inputs_target, stage='deformable')
        reconstructions_source = value_dic['dec_source']
        reconstructions_target = value_dic['dec_target']
        posterior_source = value_dic['posterior_source']
        posterior_target = value_dic['posterior_target']
        flow_pred = value_dic['flow_pred']
        target_pred = value_dic['target_pred']

        aeloss_source, log_dict_ae_source = self.loss(inputs_source, reconstructions_source, posterior_source,
                                                      0, self.global_step,
                                                      last_layer=self.get_last_layer(), split="train")
        aeloss_target, log_dict_ae_target = self.loss(inputs_target, reconstructions_target, posterior_target,
                                                      0, self.global_step,
                                                      last_layer=self.get_last_layer(), split="train")

        loss_invariance = torch.mean(
            posterior_source.kl_invariance(other=posterior_target, invariance_dim=self.invariance_dim))

        # flow loss
        # flow_pred_loss = torch.nn.functional.mse_loss(flow, flow_pred)
        rec_flow_loss = torch.nn.functional.mse_loss(inputs_target, target_pred)
        smooth_flow_loss = self.smooth_loss(flow_pred)
        flow_loss = (rec_flow_loss * self.flow_rec_weight
                     + smooth_flow_loss * self.smooth_weight)

        z_loss = loss_invariance + flow_loss * self.flow_weight

        total_loss = z_loss + (aeloss_source + aeloss_target) / 2 * self.ae_weight

        self.log("second/total_loss_s", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("second/z_loss_s", z_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("second/loss_invariance_s", loss_invariance, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("second/flow_loss_s", flow_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # self.log("self/flow_pred_loss_s", flow_pred_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("second/rec_flow_loss_s", rec_flow_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("second/smooth_flow_loss_s", smooth_flow_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("second/aeloss_source_s", aeloss_source, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae_source, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.manual_backward(total_loss)
        opt_flow.step()

    def train_contrast_deformable(self, batch, flow_idx):
        opt_ae, opt_disc, opt_flow = self.optimizers()
        inputs_source = self.get_input(batch, self.source_key)
        random_p_input = random.random()

        if random_p_input < 0.5:
            inputs_target = self.get_input(batch, self.target_key)
        else:
            inputs_target = self.get_input(batch, self.source_key)

        # At the first let the source and target image pass the network
        self.eval()
        with torch.no_grad():
            # value_dic_ori = self(inputs_source, inputs_target, stage='deformable')
            # flow_pred_ori = value_dic_ori['flow_pred'].detach()

            slice_data = batch[self.wrap_key][flow_idx]
            inputs_source_other_slice = slice_data[self.source_key]
            inputs_target_other_slice = slice_data[self.target_key]

            inputs_source_other_slice = self.auto_expand_dim(inputs_source_other_slice)
            inputs_target_other_slice = self.auto_expand_dim(inputs_target_other_slice)

            get_init_flow_value_dic = self(inputs_source_other_slice, inputs_target_other_slice, stage='deformable')
            flow = get_init_flow_value_dic['flow_pred'].detach()

            # Then Add flow on the target image, let deformable network learn the transform
            # flow_data = batch[self.wrap_key]
            random_number = random.uniform(-self.wrap_strength, self.wrap_strength)
            # flow = flow_data[:, flow_idx, :, :, :] * random_number
            # flow = flow.float()
            # flow = flow.to(self.device)
            # 对每个 b 维度进行chw归一化到0-1
            abs_max_vals = torch.abs(flow).view(flow.size(0), -1).max(dim=1, keepdim=True)[0].view(flow.size(0), 1, 1,
                                                                                                   1)
            # 归一化到0-1范围
            flow = flow / (torch.abs(abs_max_vals) + 1e-6)

            flow = flow * random_number
            random_p = random.random()
            if random_p < self.flow_zero_prob:
                flow = flow * 0.0

            inputs_pre_target = self.Transformer(inputs_source, flow)
            value_dic_pre = self(inputs_source, inputs_pre_target, stage='deformable')
            correction_flow = value_dic_pre['flow_pred'].detach()

            inputs_target = self.Transformer(inputs_target, correction_flow).detach()

        self.train()

        value_dic = self(inputs_source, inputs_target, stage='deformable')
        reconstructions_source = value_dic['dec_source']
        reconstructions_target = value_dic['dec_target']
        posterior_source = value_dic['posterior_source']
        posterior_target = value_dic['posterior_target']
        flow_pred = value_dic['flow_pred']
        target_pred = value_dic['target_pred']

        aeloss_source, log_dict_ae_source = self.loss(inputs_source, reconstructions_source, posterior_source,
                                                      0, self.global_step,
                                                      last_layer=self.get_last_layer(), split="train")
        aeloss_target, log_dict_ae_target = self.loss(inputs_target, reconstructions_target, posterior_target,
                                                      0, self.global_step,
                                                      last_layer=self.get_last_layer(), split="train")

        loss_invariance = torch.mean(posterior_source.kl_invariance(other=posterior_target,
                                                                    invariance_dim=self.invariance_dim))

        # flow_mix = estimate_mixture_flow(flow_pred_ori, correction_flow, size=(self.size, self.size))
        # flow loss
        # flow_pred_loss = torch.nn.functional.mse_loss(flow_mix, flow_pred)
        rec_flow_loss = torch.nn.functional.mse_loss(inputs_target, target_pred)
        smooth_flow_loss = self.smooth_loss(flow_pred)
        flow_loss = (rec_flow_loss * self.flow_rec_weight * 10
                     + smooth_flow_loss * self.smooth_weight)

        z_loss = loss_invariance + flow_loss * self.flow_weight

        total_loss = z_loss + (aeloss_source + aeloss_target) / 2 * self.ae_weight
        self.log("contrast/total_loss_c", total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("contrast/z_loss_c", z_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("contrast/flow_loss_c", flow_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        # self.log("contrast/flow_pred_loss_c", flow_pred_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("contrast/rec_flow_loss_c", rec_flow_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("contrast/smooth_flow_loss_c", smooth_flow_loss, prog_bar=True, logger=True, on_step=True,
                 on_epoch=True)
        self.log("contrast/loss_invariance_c", loss_invariance, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("contrast/aeloss_source_c", aeloss_source, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("contrast/aeloss_target_c", aeloss_target, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae_source, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        self.manual_backward(total_loss)
        opt_flow.step()

    def save_model(self, epoch, save_name=''):
        # 可以根据实际需求定义保存模型的逻辑
        save_path = f'log/Special_Save/{save_name}'
        os.makedirs(save_path, exist_ok=True)
        checkpoint_path = f'log/Special_Save/{save_name}/model_epoch_{epoch}.ckpt'
        self.save_checkpoint(checkpoint_path)
        print(f"Saved model checkpoint at {checkpoint_path}")

    def save_checkpoint(self, filepath):
        checkpoint = {
            'state_dict': self.state_dict(),
            # Add any other information you want to save
        }
        torch.save(checkpoint, filepath, _use_new_zipfile_serialization=False)
        print(f"Model checkpoint saved at {filepath}")

    def on_save_checkpoint(self, checkpoint: dict) -> dict:
        # 在这里添加自定义的检查点数据
        checkpoint['custom_state'] = 'This is custom state!'
        return checkpoint

    def training_step(self, batch, batch_idx):
        # first-stage: train the ae
        # print('global_step: ', str(self.global_step))
        if self.global_step < self.first_stage_step:
            # print('stage-1')
            # self.train_ae_dis(batch)
            # self.train_ae_generate(batch)
            # for i in range(self.contrast_num):
            # self.train_ae_dis(batch)
            self.pre_train_deformable(batch)
        elif self.first_stage_step <= self.global_step < self.first_stage_step + self.second_stage_step:
            if self.global_step == self.first_stage_step:
                # self.save_model(self.current_epoch, save_name='decoupling_stage-1_save')
                self.trainer.save_checkpoint(f"{self.trainer.checkpoint_callback.dirpath}/stage_1.ckpt")
            # print('stage-2')
            for i in range(self.contrast_num):
                # self.train_ae_dis(batch)
                self.pre_train_self_deformable(batch, i)
        else:
            if self.global_step == self.first_stage_step + self.second_stage_step:
                # self.save_model(self.current_epoch, save_name='decoupling_stage-2_save')
                self.trainer.save_checkpoint(f"{self.trainer.checkpoint_callback.dirpath}/stage_2.ckpt")
            # print('stage-3')
            for i in range(self.contrast_num):
                # self.train_ae_generate(batch)
                # print('train_ae_dis')
                # self.train_ae_dis(batch)
                # print('train_self_deformable')
                # self.train_self_deformable(batch, i)
                # print('train_contrast_deformable')
                self.train_contrast_deformable(batch, i)

    def validation_step(self, batch, batch_idx):
        inputs_source = self.get_input(batch, self.source_key)
        inputs_target = self.get_input(batch, self.target_key)
        value_dic = self(inputs_source, inputs_target, stage='deformable')
        reconstructions_source = value_dic['dec_source']
        reconstructions_target = value_dic['dec_target']
        posterior_source = value_dic['posterior_source']
        posterior_target = value_dic['posterior_target']

        loss_invariance = posterior_source.kl_invariance(other=posterior_target, invariance_dim=self.invariance_dim)
        loss_invariance = torch.sum(loss_invariance) / loss_invariance.shape[0]

        aeloss_source, log_dict_ae_source = self.loss(inputs_source, reconstructions_source, posterior_source,
                                                      0, self.global_step,
                                                      last_layer=self.get_last_layer(), split="val")
        aeloss_target, log_dict_ae_target = self.loss(inputs_target, reconstructions_target, posterior_target,
                                                      0, self.global_step,
                                                      last_layer=self.get_last_layer(), split="val")
        aeloss = aeloss_source + aeloss_target + loss_invariance

        discloss_source, log_dict_disc_source = self.loss(inputs_source, reconstructions_source, posterior_source,
                                                          1, self.global_step,
                                                          last_layer=self.get_last_layer(), split="val")
        discloss_target, log_dict_disc_target = self.loss(inputs_target, reconstructions_target, posterior_target,
                                                          1, self.global_step,
                                                          last_layer=self.get_last_layer(), split="val")
        discloss = discloss_source + discloss_target

        self.log("val/rec_loss", log_dict_ae_source["val/rec_loss"])
        self.log("val/loss_invariance", loss_invariance)
        self.log("val/ae_loss", aeloss)
        self.log_dict(log_dict_ae_source)
        self.log_dict(log_dict_disc_source)
        return self.log_dict
