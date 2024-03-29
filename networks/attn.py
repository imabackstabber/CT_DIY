import torch
from torch import nn
from einops import rearrange
from torch import einsum
from networks.backbone import LayerNorm2d

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def stable_softmax(t, dim = -1):
    t = t - t.amax(dim = dim, keepdim = True)
    return t.softmax(dim = dim)

# bidirectional cross attention - have two sequences attend to each other with 1 attention step

class BidirectionalCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads = 8,
        dim_head = 64,
        context_dim = None,
        dropout = 0.,
        talking_heads = False,
        prenorm = False,
    ):
        super().__init__()
        context_dim = default(context_dim, dim)

        self.norm = nn.LayerNorm(dim) if prenorm else nn.Identity()
        self.context_norm = nn.LayerNorm(context_dim) if prenorm else nn.Identity()

        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = dim_head * heads

        self.dropout = nn.Dropout(dropout)
        self.context_dropout = nn.Dropout(dropout)

        self.to_qk = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_qk = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_v = nn.Linear(dim, inner_dim, bias = False)
        self.context_to_v = nn.Linear(context_dim, inner_dim, bias = False)

        self.to_out = nn.Linear(inner_dim, dim)
        self.context_to_out = nn.Linear(inner_dim, context_dim)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()
        self.context_talking_heads = nn.Conv2d(heads, heads, 1, bias = False) if talking_heads else nn.Identity()

    def forward(
        self,
        x,
        context,
        mask = None,
        context_mask = None,
        return_attn = False,
        rel_pos_bias = None
    ):
        b, i, j, h, device = x.shape[0], x.shape[-2], context.shape[-2], self.heads, x.device

        x = self.norm(x)
        context = self.context_norm(context)

        # get shared query/keys and values for sequence and context

        qk, v = self.to_qk(x), self.to_v(x)
        context_qk, context_v = self.context_to_qk(context), self.context_to_v(context)

        # split out head

        qk, context_qk, v, context_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (qk, context_qk, v, context_v))

        # get similarities

        sim = einsum('b h i d, b h j d -> b h i j', qk, context_qk) * self.scale

        # relative positional bias, if supplied

        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias

        # mask

        if exists(mask) or exists(context_mask):
            mask = default(mask, torch.ones((b, i), device = device, dtype = torch.bool))
            context_mask = default(context_mask, torch.ones((b, j), device = device, dtype = torch.bool))

            attn_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)

        # get attention along both sequence length and context length dimensions
        # shared similarity matrix

        attn = stable_softmax(sim, dim = -1)
        context_attn = stable_softmax(sim, dim = -2)

        # dropouts

        attn = self.dropout(attn)
        context_attn = self.context_dropout(context_attn)

        # talking heads

        attn = self.talking_heads(attn)
        context_attn = self.context_talking_heads(context_attn)

        # src sequence aggregates values from context, context aggregates values from src sequence

        out = einsum('b h i j, b h j d -> b h i d', attn, context_v)
        context_out = einsum('b h j i, b h j d -> b h i d', context_attn, v)

        # merge heads and combine out

        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))

        out = self.to_out(out)
        context_out = self.context_to_out(context_out)

        if return_attn:
            return out, context_out, attn, context_attn

        return out, context_out


# adopted from https://github.com/swz30/Restormer/blob/HEAD/basicsr/models/archs/restormer_arch.py
class MDTA(nn.Module):
    def __init__(self, dim = 64, num_heads = 4, bias = False):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class MDTABlock(nn.Module):
    def __init__(self, dim = 64, num_heads = 4, bias = False):
        super(MDTABlock, self).__init__()
        
        self.mdta = MDTA(dim = dim, num_heads = num_heads, bias = bias)
        self.norm = LayerNorm2d(dim)

    def forward(self, x):
        # pre-norm and skip connection
        x = x + self.mdta(self.norm(x))
        return x

# To see the layout of CrossAttn, refer to https://medium.com/@geetkal67/attention-networks-a-simple-way-to-understand-cross-attention-3b396266d82e
class CrossAttention(nn.Module):
    def __init__(self ,dim = 64, num_heads = 4, bias = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.x_qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.x_qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.x_project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        self.ctx_qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.ctx_qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.ctx_project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, ctx):
        b,c,h,w = x.shape

        x_qkv = self.x_qkv_dwconv(self.x_qkv(x))
        x_q, x_k, x_v = x_qkv.chunk(3, dim=1)

        x_q = rearrange(x_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        x_k = rearrange(x_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        x_v = rearrange(x_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        ctx_qkv = self.ctx_qkv_dwconv(self.ctx_qkv(ctx))
        ctx_q, ctx_k, ctx_v = ctx_qkv.chunk(3, dim=1)

        ctx_q = rearrange(ctx_q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ctx_k = rearrange(ctx_k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        ctx_v = rearrange(ctx_v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        x_q = torch.nn.functional.normalize(x_q, dim=-1)
        ctx_k = torch.nn.functional.normalize(ctx_k, dim=-1)
        ctx_attn = (x_q @ ctx_k.transpose(-2,-1)) * self.temperature
        ctx_attn = ctx_attn.softmax(dim=-1)

        ctx_out = (ctx_attn @ ctx_v)
        ctx_out = rearrange(ctx_out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        ctx_out = self.ctx_project_out(ctx_out)

        ctx_q = torch.nn.functional.normalize(ctx_q, dim=-1)
        x_k = torch.nn.functional.normalize(x_k, dim=-1)
        x_attn = (ctx_q @ x_k.transpose(-2,-1)) * self.temperature
        x_attn = x_attn.softmax(dim=-1)

        x_out = (x_attn @ x_v)
        x_out = rearrange(x_out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        x_out = self.x_project_out(x_out)

        return x_out, ctx_out

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim = 64, num_heads = 4, bias = False):
        super().__init__()
        
        self.attn = CrossAttention(dim=dim, num_heads = num_heads, bias = bias)
        self.norm_x = LayerNorm2d(dim)
        self.norm_ctx = LayerNorm2d(dim)

    def forward(self, x, ctx):
        # pre-norm and skip connection
        out_x , out_ctx = self.attn(self.norm_x(x), self.norm_ctx(ctx))
        return out_x + x, out_ctx + ctx

if __name__ == '__main__':
    attn = MDTA()
    fake = torch.randn(40,64,64,64)
    out = attn(fake)
    assert out.shape == (40,64,64,64)

    attn = CrossAttention()
    fake_x = torch.randn(40,64,64,64)
    fake_ctx = torch.randn(40,64,64,64)
    x, ctx = attn(fake_x, fake_ctx)
    assert x.shape == fake_x.shape
    assert ctx.shape == fake_ctx.shape