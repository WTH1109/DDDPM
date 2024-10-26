import loralib as lora
import torch.nn as nn

def new_lora_forward(self, x):
    if self.r > 0 and not self.merged:
        return self.conv._conv_forward(
            x,
            self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
        )
    return self.conv(x)


lora.Conv2d.forward = new_lora_forward


# class LoraConv2d(lora.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super(LoraConv2d, self).__init__(*args, **kwargs)
#
#     def forward(self, x):
#         if self.r > 0 and not self.merged:
#             return self.conv._conv_forward(
#                 x,
#                 self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
#             )
#         return self.conv(x)


def dynamic_conv2d(use_lora=False, lora_config=None, *args, **kwargs):
    if lora_config is None:
        lora_config = {
            'r': 0,
            'lora_alpha': 1.0,
            'lora_dropout': 0.0,
        }
    if use_lora:
        return lora.Conv2d(*args, **lora_config, **kwargs)
    else:
        return nn.Conv2d(*args, **kwargs)


def dynamic_linear(use_lora=False, lora_config=None, *args, **kwargs):
    if lora_config is None:
        lora_config = {
            'lora_r': 0,
            'lora_alpha': 1.0,
            'lora_dropout': 0.0,
        }
    if use_lora:
        return lora.Linear(*args, **lora_config, **kwargs)
    else:
        return nn.Linear(*args, **kwargs)