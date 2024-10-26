import torch
from torch import nn
import torch.nn.functional as nnf
from torch.distributions.normal import Normal


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to('cuda')

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        if src.dtype == torch.float16:
            new_locs = new_locs.to(torch.float16)

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, num_groups=in_channels // 2)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.norm2 = Normalize(out_channels, num_groups=in_channels // 2)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


class DeformableNet(nn.Module):
    def __init__(self, in_ch, out_ch, mid_channel, up_scale_num, dropout, size, int_steps=7):
        super().__init__()
        self.up_scale_num = up_scale_num
        self.conv_in = torch.nn.Conv2d(in_ch,
                                       mid_channel,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        self.res_block = ResnetBlock(in_channels=mid_channel,
                                     out_channels=mid_channel,
                                     dropout=dropout)

        self.up = nn.ModuleList()

        up = nn.Module()

        for i_level in range(up_scale_num):
            up_block = nn.ModuleList()
            up_block.append(ResnetBlock(in_channels=mid_channel,
                                        out_channels=mid_channel * 2,
                                        dropout=dropout))
            up.up_sample = Upsample(mid_channel * 2, with_conv=True)
            up_block.append(ResnetBlock(in_channels=mid_channel * 2,
                                        out_channels=mid_channel,
                                        dropout=dropout))
            up.block = up_block

            self.up.insert(0, up)

        self.norm_out = Normalize(mid_channel, num_groups=mid_channel // 2)

        self.flow = torch.nn.Conv2d(mid_channel,
                                    out_ch,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        self.transformer = SpatialTransformer((size, size))

        self.integrate = VecInt((size, size), int_steps) if int_steps > 0 else None

    def forward(self, x, y_source):
        x = self.conv_in(x)
        x = self.res_block(x)
        h = x
        for i_level in range(self.up_scale_num):
            x = self.up[i_level].block[0](x)
            x = self.up[i_level].up_sample(x)
            x = self.up[i_level].block[1](x)
        x = self.norm_out(x)
        flow = self.flow(x)
        flow = self.integrate(flow)
        y_target = self.transformer(y_source, flow)

        return flow, y_target, h


def estimate_mixture_flow(flow1, flow2, size):
    batch_size, _, _, _ = flow1.shape
    transformer = SpatialTransformer(size)
    vectors = [torch.arange(0, s) for s in size]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)
    grid = torch.unsqueeze(grid, 0)
    grid = grid.repeat(batch_size, 1, 1, 1)
    grid = grid.type(torch.FloatTensor)
    grid = grid.to('cuda')
    grid_new = transformer(grid, flow1)
    grid_new = transformer(grid_new, flow2)
    mix_flow = grid_new - grid
    return mix_flow



if __name__ == '__main__':
    input_tensor = torch.randn((3, 2, 64, 64))
    model = DeformableNet(2, 2, 3, 2, 0.5)
    out, emb = model(input_tensor)

    print('done')
