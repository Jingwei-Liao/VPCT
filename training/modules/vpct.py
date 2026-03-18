import torch
import torch.nn as nn


class CausalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def _reshape_heads(self, x):
        batch_size, seq_len, dim = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, query, key, value, attn_mask):
        q = self._reshape_heads(self.q_proj(query))
        k = self._reshape_heads(self.k_proj(key))
        v = self._reshape_heads(self.v_proj(value))

        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        mask = attn_mask.to(device=logits.device, dtype=torch.bool)
        while mask.dim() < logits.dim():
            mask = mask.unsqueeze(0)

        masked_logits = logits.masked_fill(~mask, -1e9)
        weights = torch.softmax(masked_logits, dim=-1)
        weights = weights * mask.to(weights.dtype)
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights = weights / denom

        out = torch.matmul(weights, v)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(query.size(0), query.size(1), -1)
        return self.out_proj(out)


class IntraViewModule(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim, bias=False)
        self.attn = CausalAttention(dim=dim, num_heads=num_heads)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim, bias=False),
            nn.Linear(dim, 4 * dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * dim, dim, bias=False),
        )

    def _build_causal_mask(self, token_count, device):
        rows = torch.arange(token_count, device=device).unsqueeze(1)
        cols = torch.arange(token_count, device=device).unsqueeze(0)
        return cols < rows

    def _prepend_start_token(self, x_tokens):
        start_token = torch.zeros(
            x_tokens.size(0), 1, x_tokens.size(2), device=x_tokens.device, dtype=x_tokens.dtype
        )
        return torch.cat([start_token, x_tokens], dim=1)

    def forward(self, x):
        batch_size, num_views, channels, height, width = x.shape
        token_count = height * width

        x_tokens = x.view(batch_size * num_views, channels, token_count).transpose(1, 2)
        # x_tokens = self._prepend_start_token(x_tokens)
        x_norm = self.norm(x_tokens)

        mask = self._build_causal_mask(token_count, x_tokens.device)
        attn_out = self.attn(x_norm, x_norm, x_norm, mask)
        x_tokens = x_tokens + attn_out
        x_tokens = x_tokens + self.ffn(x_tokens)
        # x_tokens = x_tokens[:, :-1, :]

        x_tokens = x_tokens.transpose(1, 2).contiguous()
        return x_tokens.view(batch_size, num_views, channels, height, width)


class InterViewModule(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(dim, bias=False)
        self.attn = CausalAttention(dim=dim, num_heads=num_heads)
        self.ffn = nn.Sequential(
            nn.LayerNorm(dim, bias=False),
            nn.Linear(dim, 4 * dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * dim, dim, bias=False),
        )

    def _build_inter_view_mask(self, num_views, token_count, device):
        total = num_views * token_count
        query_ids = torch.arange(total + 1, device=device)
        key_ids = torch.arange(total + 1, device=device)

        query_is_start = query_ids == 0
        key_is_start = key_ids == 0

        query_ids_shift = torch.clamp(query_ids - 1, min=0)
        key_ids_shift = torch.clamp(key_ids - 1, min=0)

        query_view = query_ids_shift // token_count
        query_pos = query_ids_shift % token_count
        key_view = key_ids_shift // token_count
        key_pos = key_ids_shift % token_count

        query_is_start = query_is_start.unsqueeze(1)
        key_is_start = key_is_start.unsqueeze(0)

        query_view = query_view.unsqueeze(1)
        query_pos = query_pos.unsqueeze(1)
        key_view = key_view.unsqueeze(0)
        key_pos = key_pos.unsqueeze(0)

        prev_view_allowed = key_view < query_view
        same_view_prev_token_allowed = (key_view == query_view) & (key_pos < query_pos)
        regular_allowed = prev_view_allowed | same_view_prev_token_allowed

        allow_start_token = (~query_is_start) & key_is_start
        return regular_allowed | allow_start_token

    def _prepend_start_token(self, x_tokens):
        start_token = torch.zeros(
            x_tokens.size(0), 1, x_tokens.size(2), device=x_tokens.device, dtype=x_tokens.dtype
        )
        return torch.cat([start_token, x_tokens], dim=1)

    def forward(self, x):
        batch_size, num_views, channels, height, width = x.shape
        token_count = height * width

        x_tokens = x.view(batch_size, num_views, channels, token_count)
        x_tokens = x_tokens.permute(0, 1, 3, 2).contiguous()
        x_tokens = x_tokens.view(batch_size, num_views * token_count, channels)
        x_tokens = self._prepend_start_token(x_tokens)

        x_norm = self.norm(x_tokens)
        mask = self._build_inter_view_mask(num_views, token_count, x_tokens.device)
        attn_out = self.attn(x_norm, x_norm, x_norm, mask)

        x_tokens = x_tokens + attn_out
        x_tokens = x_tokens + self.ffn(x_tokens)
        x_tokens = x_tokens[:, :-1, :]

        x_tokens = x_tokens.view(batch_size, num_views, token_count, channels)
        x_tokens = x_tokens.permute(0, 1, 3, 2).contiguous()
        return x_tokens.view(batch_size, num_views, channels, height, width)


class VPCTModule(nn.Module):
    def __init__(self, channels, num_heads=8, num_layers=1):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.intra_layers = nn.ModuleList(
            [IntraViewModule(dim=channels, num_heads=num_heads) for _ in range(num_layers)]
        )
        self.inter_layers = nn.ModuleList(
            [InterViewModule(dim=channels, num_heads=num_heads) for _ in range(num_layers)]
        )

    def forward(self, y_hat):
        for intra_layer, inter_layer in zip(self.intra_layers, self.inter_layers):
            y_hat = intra_layer(y_hat)
            y_hat = inter_layer(y_hat)
        return y_hat
