import torch.nn as nn
from einops import rearrange
import math
import torch
import torch.nn.functional as F
from torch.autograd import Function
from torch import nn


class LowerBound(Function):
    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones(inputs.size())*bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)
    
    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None

class GSDN(nn.Module):
    """Generalized Subtractive and Divisive Normalization layer.
    y[i] = (x[i] - )/ sqrt(beta[i] + sum_j(gamma[j, i] * x[j]))
    """
  
    def __init__(self,
                 ch,
                 inverse=False,
                 beta_min=1e-6,
                 gamma_init=.1,
                 reparam_offset=2**-18):
        super().__init__()
        self.inverse = inverse
        self.build(ch, beta_min, gamma_init, reparam_offset)
  
    def build(self, ch, beta_min, gamma_init, reparam_offset ):
        self.pedestal = reparam_offset**2
        self.beta_bound = torch.FloatTensor([ ( beta_min + reparam_offset**2)**.5 ] )
        self.gamma_bound = torch.FloatTensor( [ reparam_offset] )
        
        ###### param for divisive ######
        # Create beta param
        beta = torch.sqrt(torch.ones(ch)+self.pedestal)
        self.beta = nn.Parameter(beta)
        # Create gamma param
        eye = torch.eye(ch)
        g = gamma_init*eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)
        self.gamma = nn.Parameter(gamma)
        
        ###### param for subtractive ######
        # Create beta2 param
        beta2 = torch.zeros(ch)
        self.beta2 = nn.Parameter(beta2)
        # Create gamma2 param
        eye = torch.eye(ch)
        g = gamma_init*eye
        g = g + self.pedestal
        gamma2 = torch.sqrt(g)
        self.gamma2 = nn.Parameter(gamma2)

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size() 
            inputs = inputs.view(bs, ch, d*w, h)

        _, ch, _, _ = inputs.size()
        
        if self.inverse:
            # Scale
            beta = LowerBound.apply(self.beta, self.beta_bound)
            beta = beta**2 - self.pedestal 
            gamma = LowerBound.apply(self.gamma, self.gamma_bound)
            gamma = gamma**2 - self.pedestal
            gamma = gamma.view(ch, ch, 1, 1)
            norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
            norm_ = torch.sqrt(norm_)
      
            outputs = inputs * norm_  # modified
            
            # Mean
            beta2 = LowerBound.apply(self.beta2, self.beta_bound)
            beta2 = beta2**2 - self.pedestal 
            gamma2 = LowerBound.apply(self.gamma2, self.gamma_bound)
            gamma2 = gamma2**2 - self.pedestal
            gamma2 = gamma2.view(ch, ch, 1, 1)
            mean_ = nn.functional.conv2d(inputs, gamma2, beta2)
      
            outputs = outputs + mean_
        else:
            # Mean
            beta2 = LowerBound.apply(self.beta2, self.beta_bound)
            beta2 = beta2**2 - self.pedestal 
            gamma2 = LowerBound.apply(self.gamma2, self.gamma_bound)
            gamma2 = gamma2**2 - self.pedestal
            gamma2 = gamma2.view(ch, ch, 1, 1)
            mean_ = nn.functional.conv2d(inputs, gamma2, beta2)
      
            outputs = inputs - mean_  # modified

            # Scale
            beta = LowerBound.apply(self.beta, self.beta_bound)
            beta = beta**2 - self.pedestal 
            gamma = LowerBound.apply(self.gamma, self.gamma_bound)
            gamma = gamma**2 - self.pedestal
            gamma = gamma.view(ch, ch, 1, 1)
            norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
            norm_ = torch.sqrt(norm_)
      
            outputs = outputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)

        return outputs

class CfgNode:
    """ a lightweight configuration class inspired by yacs """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ return a dict representation of the config """
        return { k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items() }

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].

        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == '--'
            key = key[2:] # strip the '--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CorssCasualAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.q_attn = nn.Linear(config.n_embd, config.n_embd)
        self.kv_attn = nn.Linear(config.n_embd, 2 * config.n_embd)
        
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, q, kv):
        device = q.device

        vi, bi, ci, hi, wi = q.size()
        vj, bj, cj, hj, wj = kv.size()


        q = rearrange(q, 'v b c h w -> b (v h w) c')
        kv = rearrange(kv, 'v b c h w -> b (v h w) c')

        B, T, C = q.size() 


        q  = self.q_attn(q)
        k, v  = self.kv_attn(kv).split(self.n_embd, dim=2)
        

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)# (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        mask = torch.zeros((T,T))
        for i in range(vi):
            mask[i*hi*wi:, :(i+1)*hi*wi] = 1

        mask = mask.to(device)

        att = att.masked_fill(mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        y = rearrange(y, 'b (v h w) c -> v b c h w', h=hi, w=wi)
        return y

class CorssCasualAttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CorssCasualAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        i,c,j = x

        v, b, ch, h, w = i.size()
        
        i = rearrange(i, 'v b c h w -> (v b) (h w) c')
        j = rearrange(j, 'v b c h w -> (v b) (h w) c')

        i = self.ln_1(i)
        j = self.ln_1(j)
        
        i = rearrange(i, '(v b) (h w) c -> v b c h w', v = v, h = h)
        j = rearrange(j, '(v b) (h w) c -> v b c h w', v = v, h = h)


        

        if c == None:
            c = self.attn(i, j)
        else:
            c = rearrange(c, 'v b c h w -> (v b) (h w) c')
            c = self.ln_1(c)
            c = rearrange(c, '(v b) (h w) c -> v b c h w', v = v, h = h)
            c = c + self.attn(i, j)

        c = rearrange(c, 'v b c h w -> (v b) (h w) c')
        c = c + self.mlpf(self.ln_2(c))
        c = rearrange(c, '(v b) (h w) c -> v b c h w', v = v, h = h)

        return c

class CrossViewAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn_kv = nn.Linear(config.n_embd, 2*config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, q, kv):
        device = q.device

        vi, bi, ci, hi, wi = q.size()
        vj, bj, cj, hj, wj = kv.size()

        q = rearrange(q, 'v b c h w -> (v b) (h w) c')
        kv = rearrange(kv, 'v b c h w -> (v b) (h w) c')

        assert vi == vj, 'Viewport number error'
        assert hi == hj, 'Viewport number error'
        assert wi == wj, 'Viewport number error'

        B, T, C = q.size()
        


        k ,v  = self.c_attn_kv(kv).split(self.n_embd, dim=2) # (v b) (h w) c

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # (B nh T T)

        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)


        # output projection
        y = self.resid_dropout(self.c_proj(y))

        y = rearrange(y, '(v b) (h w) c -> v b c h w', v = vi, h = hi)

        return y

class CrossViewAttentionBlock(nn.Module):
    """ an Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CrossViewAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        i,c,j = x

        v, b, ch, h, w = i.size()
        
        i = rearrange(i, 'v b c h w -> (v b) (h w) c')
        j = rearrange(j, 'v b c h w -> (v b) (h w) c')
        i = self.ln_1(i)
        j = self.ln_1(j)
        i = rearrange(i, '(v b) (h w) c -> v b c h w', v = v, h = h)
        j = rearrange(j, '(v b) (h w) c -> v b c h w', v = v, h = h)


        

        if c == None:
            c = self.attn(i, j)
        else:
            c = rearrange(c, 'v b c h w -> (v b) (h w) c')
            c = self.ln_1(c)
            c = rearrange(c, '(v b) (h w) c -> v b c h w', v = v, h = h)
            c = c + self.attn(i, j)

        c = rearrange(c, 'v b c h w -> (v b) (h w) c')
        c = c + self.mlpf(self.ln_2(c))
        c = rearrange(c, '(v b) (h w) c -> v b c h w', v = v, h = h)

        return c

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        device = x.device
        vp, b, c, h, w = x.size()
        x = rearrange(x, 'v b c h w -> (v b) (h w) c')
        B, T, C = x.size() 


        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)# (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        y = rearrange(y, '(v b) (h w) c -> v b c h w', v=vp, h=h)
        return y

class SelfAttentionBlock(nn.Module):
    """ an Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        v, b, c, h, w = x.size()
        
        x = rearrange(x, 'v b c h w -> (v b) (h w) c')
        x = self.ln_1(x)
        x = rearrange(x, '(v b) (h w) c -> v b c h w', v = v, h = h)

        x = x + self.attn(x)

        x = rearrange(x, 'v b c h w -> (v b) (h w) c')
        x = x + self.mlpf(self.ln_2(x))
        x = rearrange(x, '(v b) (h w) c -> v b c h w', v = v, h = h)

        return x

class CasualAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        device = x.device
        vp, b, c, h, w = x.size()
        x = rearrange(x, 'v b c h w -> b (v h w) c')
        B, T, C = x.size() 


        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)# (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) 

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        mask = torch.zeros((T,T))
        for i in range(vp):
            mask[i*h*w:, :(i+1)*h*w] = 1

        mask = mask.to(device)

        att = att.masked_fill(mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        y = rearrange(y, 'b (v h w) c -> v b c h w', h=h, w=w)
        return y

class CasualAttentionBlock(nn.Module):
    """ an Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        v, b, c, h, w = x.size()
        
        x = rearrange(x, 'v b c h w -> (v b) (h w) c')
        x = self.ln_1(x)
        x = rearrange(x, '(v b) (h w) c -> v b c h w', v = v, h = h)

        x = x + self.attn(x)

        x = rearrange(x, 'v b c h w -> (v b) (h w) c')
        x = x + self.mlpf(self.ln_2(x))
        x = rearrange(x, '(v b) (h w) c -> v b c h w', v = v, h = h)

        return x


class BasicBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.sa = SelfAttentionBlock(config)
        self.cva = CorssCasualAttentionBlock(config)

    def forward(self, icj):
        i, c, j = icj
        i = self.sa(i)
        c = self.cva((i, c, j))

        return (i, c, j)

    
class VPCT(nn.Module):
    @staticmethod
    def get_default_config():
        C = CfgNode()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.model_type = 'vpct'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C
    
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None


        # define model parameter
        config.merge_from_dict(
            dict(n_layer=6, n_head=8, n_embd=256)
        )


        self.transformer = nn.ModuleDict(dict(
            wte_i = nn.Linear(config.vocab_size, config.n_embd),
            wte_j = nn.Linear(config.vocab_size, config.n_embd),
            
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([BasicBlock(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.out_size, bias=False)
        self.param_token = nn.Parameter(torch.randn(1, 1, config.out_size))
        
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.transformer.parameters())
        self.config = config


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, yi, yj):
        vi,bi,ci,hi,wi = yi.size()
        yi = rearrange(yi, 'v b c h w -> (v b) (h w) c')
        yi = self.transformer.wte_i(yi) 
        yi = self.transformer.drop(yi)
        yi = rearrange(yi, '(v b) (h w) c -> v b c h w', v=vi, h=hi)

        vnj,bj,cj,hj,wj = yj.size()
        yj = rearrange(yj, 'v b c h w -> (v b) (h w) c')
        yj = self.transformer.wte_j(yj) 
        yj = self.transformer.drop(yj)
        yj = rearrange(yj, '(v b) (h w) c -> v b c h w', v=vnj, h=hj)

        
        x = (yi, None, yj)
        for block in self.transformer.h:
            x = block(x)
        x = x[2]
        x = rearrange(x, 'v b c h w -> (v b) (h w) c')
        x = self.transformer.ln_f(x)
        
        logits = self.lm_head(x)
        logits = rearrange(logits, '(v b) (h w) c -> v b c h w', v=vi, h=hi)

        return logits


class VPCTModule(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super().__init__()
        self.model_config = VPCT.get_default_config()
        self.model_config.vocab_size = in_c
        self.model_config.out_size = out_c
        self.attention = VPCT(self.model_config)
    
    def forward(self, yi, yj):
        y = self.attention(yi, yj)
        return y

    



