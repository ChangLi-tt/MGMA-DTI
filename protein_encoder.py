import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
import torch.fft
from torch.nn import LayerNorm

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)

class GlobalLocalFilter(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='chann`els_first')
 
    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)
        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')
        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear',
                                   align_corners=True).permute(1, 2, 3, 0)
        weight = torch.view_as_complex(weight.contiguous())
        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')
        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x

class Multi_order_gated(nn.Module):
    def __init__(self, in_planes, out_planes, order=3, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order # order
        self.dims = [in_planes // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(in_planes, 2* out_planes, 1)
        
        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)

        self.proj_out = nn.Conv2d(in_planes, out_planes, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )

        self.scale = s

    def forward(self, x, mask=None, dummy=False):

        fused_x = self.proj_in(x)# channel double
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)# split channel to c/2**order and c(1-2**oder)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        # print('dw_list',dw_list.shape)
        x = pwa * dw_list[0]

        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]
        x = self.proj_out(x)

        return x







