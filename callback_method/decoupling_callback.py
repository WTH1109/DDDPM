import os

import torch
import loralib as lora
from pytorch_lightning.callbacks import ModelCheckpoint


class DecouplingCheckpoint(ModelCheckpoint):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(trainer, pl_module, checkpoint)

        cond_lora_save_path = f'{trainer.checkpoint_callback.dirpath}/lora_cond/'
        diffusion_ori_save_path = f'{trainer.checkpoint_callback.dirpath}/diffusion/'
        model_path = trainer.checkpoint_callback.dirpath

        os.makedirs(cond_lora_save_path, exist_ok=True)
        os.makedirs(diffusion_ori_save_path, exist_ok=True)

        self.clean_model(model_path, cond_lora_save_path, diffusion_ori_save_path)


        torch.save(lora.lora_state_dict(pl_module.cond_stage_model, bias='lora_only'),
                   os.path.join(cond_lora_save_path, f'lora_cond_{str(trainer.current_epoch).zfill(6)}.pth'))
        torch.save(pl_module.model.state_dict(),
                   os.path.join(diffusion_ori_save_path, f'diffusion_{str(trainer.current_epoch).zfill(6)}.pth'))

        torch.save(lora.lora_state_dict(pl_module.cond_stage_model, bias='lora_only'),
                   os.path.join(cond_lora_save_path, f'last.pth'))
        torch.save(pl_module.model.state_dict(),
                   os.path.join(diffusion_ori_save_path, f'last.pth'))

    @staticmethod
    def clean_model(_model_path, _lora_path, _diffusion_ori_save_path):
        # Delete the overage save model in lora and diffusion
        model_list = os.listdir(_model_path)
        lora_list = os.listdir(_lora_path)
        ep_list = []
        for model_name in model_list:
            try:
                ep_list.append(model_name.split('.')[-2].split('=')[-1])
            except:
                continue

        del_save_ep_list_lora = []
        for model_name in lora_list:
            try:
                save_model_ep = model_name.split('.')[-2].split('_')[-1].zfill(6)
                if save_model_ep not in ep_list:
                    del_save_ep_list_lora.append(model_name.split('.')[-2].split('_')[-1])
            except:
                continue

        for del_save in del_save_ep_list_lora:
            if os.path.exists(os.path.join(_lora_path, f'lora_cond_{del_save}.pth')):
                os.remove(os.path.join(_lora_path, f'lora_cond_{del_save}.pth'))
            if os.path.exists(os.path.join(_diffusion_ori_save_path, f'diffusion_{del_save}.pth')):
                os.remove(os.path.join(_diffusion_ori_save_path, f'diffusion_{del_save}.pth'))


if __name__ == '__main__':
    # model_path = '/mnt/hdd1/wengtaohan/Code/latent-diffusion-main/logs/2024-06-17T16-50-28_dddpm_simense_v3/checkpoints'
    # lora_path = '/mnt/hdd1/wengtaohan/Code/latent-diffusion-main/logs/2024-06-17T16-50-28_dddpm_simense_v3/checkpoints/lora_cond/'
    # diffusion_ori_save_path = '/mnt/hdd1/wengtaohan/Code/latent-diffusion-main/logs/2024-06-17T16-50-28_dddpm_simense_v3/checkpoints/diffusion/'

    # model_list = os.listdir(model_path)
    # lora_list = os.listdir(lora_path)
    # ep_list = []
    # for model_name in model_list:
    #     try:
    #         ep_list.append(model_name.split('.')[-2].split('=')[-1])
    #     except:
    #         continue
    #
    # del_save_ep_list_lora = []
    # for model_name in lora_list:
    #     try:
    #         save_model_ep = model_name.split('.')[-2].split('_')[-1].zfill(6)
    #         if save_model_ep not in ep_list:
    #             del_save_ep_list_lora.append(model_name.split('.')[-2].split('_')[-1])
    #     except:
    #         continue
    #
    # for del_save in del_save_ep_list_lora:
    #     if os.path.exists(os.path.join(lora_path, f'lora_cond_{del_save}.pth')):
    #         os.remove(os.path.join(lora_path, f'lora_cond_{del_save}.pth'))
    #     if os.path.exists(os.path.join(diffusion_ori_save_path, f'diffusion_{del_save}.pth')):
    #         os.remove(os.path.join(diffusion_ori_save_path, f'diffusion_{del_save}.pth'))

    print('done')
