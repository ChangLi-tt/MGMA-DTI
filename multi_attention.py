import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from einops import reduce

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        # if self.data_format not in ["channels_last", "channels_first"]:
        #     raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, kqv_bias=False, device='cuda'):
        super().__init__()
        # ConvFormer
        self.device = device
        self.norm = LayerNorm(dim, eps=1e-6)
        self.ava = nn.Sequential(
            nn.Conv1d(dim, dim, 1),
            nn.GELU(),
            nn.Conv1d(dim, dim, 11, padding=5, groups=dim)
        )
        self.v = nn.Conv1d(dim, dim, 1)
        self.proj = nn.Conv1d(dim, dim, 1)
        self.dim = dim
        self.head = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, 'dim must be divisible by num_heads!' 

        self.wq = nn.Linear(dim, dim, bias=kqv_bias)
        self.wk = nn.Linear(dim, dim, bias=kqv_bias)
        self.wv = nn.Linear(dim, dim, bias=kqv_bias)
        self.softmax = nn.Softmax(dim=-2) 
        self.out = nn.Linear(dim, dim)
        
    
    def forward(self, x, query=None):
        if query is None:
            query = x

        query = self.wq(query)
        key = self.wk(x)
        value = self.wv(x)

        b, n, c = x.shape  
        key = key.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3) 
        key_t = key.clone().permute(0, 1, 3, 2) 
        value = value.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3) 

        b, n, c = query.shape
        query = query.reshape(b, n, self.head, self.head_dim).permute(0, 2, 1, 3)

        kk = key_t @ key
        kk_inv = torch.inverse(kk)
        attn_map = (kk_inv @ key_t) @ value
        attn_map = self.softmax(attn_map)

        x1 = (query @ attn_map)  
        x1 = x1.permute(0, 2, 1, 3).reshape(b, n, c)
        x = self.norm(x1)  #
        x = x.permute(0, 2, 1).reshape(b, c, n)
        a = self.ava(x)
        v = self.v(x)
        x = a * v
        x = self.proj(x)

        out = x.permute(0, 2, 1).reshape(b, n, c)
        out = self.out(out)
        return out


# SelfAttention
class SelfAttention(nn.Module):

    def __init__(self, dim, num_heads, dropout=0.):
        super(SelfAttention, self).__init__()
        self.wq = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.wk = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        self.wv = nn.Sequential(
             nn.Linear(dim, dim),
             nn.Dropout(p=dropout)
            )
        # MultiheadAttention
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads)

    def forward(self, x):
        query = self.wq(x)
        key = self.wk(x)
        value = self.wv(x)
        att, _ = self.attn(query, key, value)
        out = att + x
        return out

class AttentionBlock(nn.Module):

    def __init__(self, dim, num_heads=8, kqv_bias=True,device='cuda'):
        super(AttentionBlock, self).__init__()
        self.norm_layer = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, num_heads=num_heads, kqv_bias=kqv_bias,device=device) 
        self.softmax = nn.Softmax(dim=-2)
        self.beta = nn.Parameter(torch.rand(1))

    def forward(self, x, q):
        x = self.norm_layer(x)
        q_t = q.permute(0, 2, 1)
        att = self.attn(x, q)
        att_map = self.softmax(att)
        out = self.beta * q_t @ att_map
        return out

class Fea_extractor(nn.Module):
    def __init__(self, embed_dim, layer=1, num_head=8, device='cuda'):
        super(Fea_extractor, self).__init__()


        self.layer = layer
        self.drug_intention = nn.ModuleList([
            AttentionBlock(dim=embed_dim, device=device, num_heads=num_head) for _ in range(layer)])
        self.protein_intention = nn.ModuleList([
            AttentionBlock(dim=embed_dim, device=device, num_heads=num_head) for _ in range(layer)])
        #self attention
        self.attn_drug = SelfAttention(dim=embed_dim, num_heads=num_head)
        self.attn_protein = SelfAttention(dim=embed_dim, num_heads=num_head)

    def forward(self, drug, protein):
        drug = self.attn_drug(drug)
        protein = self.attn_protein(protein)
        for i in range(self.layer):
            v_p = self.drug_intention[i](drug, protein)
            v_d = self.protein_intention[i](protein, drug)
            drug, protein = v_d, v_p

        v_d = reduce(drug, 'B H W -> B H', 'max')
        v_p = reduce(protein, 'B H W -> B H', 'max')

        f = torch.cat((v_d, v_p), dim=1)

        return f, v_d, v_p, None



