import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Transformer(nn.Module):
    """Multi-layer single-head Transformer decoder."""

    def __init__(self, vocab_size, hidden_dim, context_len, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.context_len = context_len
        self.num_layers = num_layers

        # Token and Positional embeddings
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(context_len, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer_modules = nn.ModuleDict({
                'w_q': nn.Linear(hidden_dim, hidden_dim, bias=False),
                'w_k': nn.Linear(hidden_dim, hidden_dim, bias=False),
                'w_v': nn.Linear(hidden_dim, hidden_dim, bias=False),
                'w_o': nn.Linear(hidden_dim, hidden_dim, bias=False),
                'w_up': nn.Linear(hidden_dim, 4 * hidden_dim),
                'w_down': nn.Linear(4 * hidden_dim, hidden_dim)
            })
            self.layers.append(layer_modules)

        # Layer norm parameters
        self.gamma_attn = nn.ParameterList([nn.Parameter(torch.ones(hidden_dim)) for _ in range(num_layers)])
        self.beta_attn = nn.ParameterList([nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_layers)])
        self.gamma_mlp = nn.ParameterList([nn.Parameter(torch.ones(hidden_dim)) for _ in range(num_layers)])
        self.beta_mlp = nn.ParameterList([nn.Parameter(torch.zeros(hidden_dim)) for _ in range(num_layers)])

    def layer_norm(self, x, gamma, beta):
        # USE_LN = False
        # if not USE_LN: return x
        mu = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        sigma = torch.sqrt(var + 1e-5)
        return gamma * (x - mu) / sigma + beta

    def forward(self, x):
        B, T = x.size()
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)

        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]
            residual = h

            Q = layer['w_q'](h)
            K = layer['w_k'](h)
            V = layer['w_v'](h)

            d_k = self.hidden_dim
            wei = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)

            tril = torch.tril(torch.ones(T, T, device=x.device))
            wei = wei.masked_fill(tril == 0, float('-inf'))

            attn_weights = F.softmax(wei, dim=-1)
            out = attn_weights @ V
            Z = layer['w_o'](out)

            h = self.layer_norm(residual + Z, self.gamma_attn[layer_idx], self.beta_attn[layer_idx])

            residual_mlp = h
            mlp_out = layer['w_up'](h)
            mlp_out = F.relu(mlp_out)
            mlp_out = layer['w_down'](mlp_out)
            h = self.layer_norm(residual_mlp + mlp_out, self.gamma_mlp[layer_idx], self.beta_mlp[layer_idx])

        return h


class MultiHeadTransformer(Transformer):
    """Multi-layer multi-head Transformer decoder."""

    def __init__(self, vocab_size, hidden_dim, context_len, num_heads=4, num_layers=2):
        super().__init__(vocab_size, hidden_dim, context_len, num_layers)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "Hidden dim must be divisible by num_heads"

    def forward(self, x):
        B, T = x.size()
        positions = torch.arange(0, T, device=x.device).unsqueeze(0)
        h = self.embedding(x) + self.pos_embedding(positions)

        for layer_idx in range(self.num_layers):
            layer = self.layers[layer_idx]
            residual = h

            q = layer['w_q'](h)
            k = layer['w_k'](h)
            v = layer['w_v'](h)

            q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

            d_k = self.head_dim
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

            tril = torch.tril(torch.ones(T, T, device=x.device))
            scores = scores.masked_fill(tril == 0, float('-inf'))

            attn_weights = F.softmax(scores, dim=-1)
            context = attn_weights @ v

            context = context.transpose(1, 2).contiguous().view(B, T, self.hidden_dim)
            z = layer['w_o'](context)

            h = self.layer_norm(residual + z, self.gamma_attn[layer_idx], self.beta_attn[layer_idx])

            residual_mlp = h
            mlp_out = layer['w_up'](h)
            mlp_out = F.relu(mlp_out)
            mlp_out = layer['w_down'](mlp_out)

            h = self.layer_norm(residual_mlp + mlp_out, self.gamma_mlp[layer_idx], self.beta_mlp[layer_idx])

        return h