import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Embedding, Linear

@dataclass
class ModelArgs_Vicuna:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048

    lora_rank: int = 8
    dynamic_active_target: float = 0.5
    dynamic_start_layer: int = 2
    dynamic_router_hdim: int = 512
    dynamic_reserve_initials: int = 2

    # Vicuna
    max_position_embeddings: int=4096


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        return (self.weight * hidden_states).to(input_dtype)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.

    
        

    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        

    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

# [Modified]
# def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
#     """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
#     bs, slen, n_kv_heads, head_dim = x.shape
#     if n_rep == 1:
#         return x
#     return (
#         x[:, :, :, None, :]
#         .expand(bs, slen, n_kv_heads, n_rep, head_dim)
#         .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
#     )


class ActiveLoss(nn.Module):
    def __init__(self, target, reserve_initials):
        """
        Initialize the ActiveLoss module to calculate about routes.

        Args & Attributes:
            target (float): Target execution ratio expecting the D-LLM to behave.
            reserve_initials (int): Number of tokens at the beginning of sentences reserved as executed.

        """
        super().__init__()
        self.target = target
        self.reserve_initials = reserve_initials

    @torch.no_grad()
    def metric(self, activation: torch.Tensor, mask: torch.Tensor):
        """
        Provide metrics on activations. Used for logging.

        Args:
            activation (torch.Tensor): Input routes tensor.
            mask (torch.Tensor): Mask tensor to indicate unconsidered routes.

        Returns:
            metrics (Dict): Metrics on activations.

        """
        activation = activation[:, self.reserve_initials:, :]
        mask = mask[:, self.reserve_initials:]

        metrics = {}

        mean_scalar_ratio = (activation.mean(dim=-1) * mask).sum() / mask.sum()
        mean_batch_ratio = (activation.mean(dim=-1) * mask).sum(dim=-1) / mask.sum(dim=-1)
        max_scalar_ratio = (activation.mean(dim=-1) * mask).max()
        min_scalar_ratio = (activation.mean(dim=-1) * mask).min()

        metrics.update(mean_scalar_ratio=mean_scalar_ratio)
        metrics.update(mean_batch_ratio=mean_batch_ratio)
        metrics.update(max_scalar_ratio=max_scalar_ratio)
        metrics.update(min_scalar_ratio=min_scalar_ratio)

        return metrics


    def forward(self, activation: torch.Tensor, mask: torch.Tensor):
        """
        Forward pass of the ActiveLoss module.

        Args:
            activation (torch.Tensor): Input routes tensor.
            mask (torch.Tensor): Mask tensor to indicate unconsidered routes.

        Returns:
            loss (torch.Tensor): Loss between actual ratio of activations and the target ratio.

        """
        activation = activation[:, self.reserve_initials:, :]
        mask = mask[:, self.reserve_initials:]

        activation = activation.mean(dim=-1)
        target = torch.ones_like(activation) * self.target
        loss = F.l1_loss(activation, target, reduction='none')
        loss = loss * mask
        loss = loss.sum() / mask.sum()
        
        return loss


class LoRAModule(nn.Module):
    def __init__(self, in_dim : int, rank : int, out_dim : int):
        """
        Initialize the LoRA module.

        Args:
            in_dim (int): Dim of input feature.
            rank (int): Dim of hidden layer.
            out_dim (int): Dim of output tensor.

        Attributes:
            linear_A (Linear): Linear transformation A for rank reduction.
            linear_B (Linear): Linear transformation B for rank up.
            dropout (nn.Dropout): Droupout layer.

        """
        super().__init__()
        self.linear_A = nn.Linear(in_dim, rank, bias=False)
        self.linear_B = nn.Linear(rank, out_dim, bias=False)
        self.dropout = nn.Dropout(0.05)

        nn.init.normal_(self.linear_A.weight, std=1 / rank)
        nn.init.zeros_(self.linear_B.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear_B(self.linear_A(x.float()))
        return self.dropout(x).bfloat16()


class RouterModule(nn.Module):
    def __init__(
        self, 
        in_dim: int,
        hidden_dim: int,
        reserve_initials: int,
        norm_eps: float
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Initialize the Router module.

        Args:
            in_dim (int): Dim of input feature.
            hidden_dim (int): Dim of hidden layer.
            reserve_initials (int): Number of tokens at the beginning of sentences reserved as executed.
            norm_eps (float): Eps for RMSNorm.

        Attributes:
            linear_1 (Linear): Linear transformation 1 for router.
            norm (RMSNorm): Layer normalization for the model output.
            linear_2 (Linear): Linear transformation 2 for router.
            reserve_initials (int): Number of tokens at the beginning of sentences reserved as executed.

        """
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, hidden_dim)
        self.norm = LlamaRMSNorm(hidden_dim, norm_eps)
        self.linear_2 = nn.Linear(hidden_dim, 2)

        self.linear_2.bias.data[0] = 0.1
        self.linear_2.bias.data[1] = 2.0

        self.reserve_initials = reserve_initials

    def asymmetric_quantization_for_similarity(self, tensor_3d, quant_bits):
        min_vals = tensor_3d.min(dim=-1, keepdim=True)[0]
        max_vals = tensor_3d.max(dim=-1, keepdim=True)[0]
        epsilon = 1e-7
        scale = (max_vals - min_vals) / (2 ** quant_bits - 1) + epsilon
        q_tensor = torch.round((tensor_3d - min_vals) / scale)
        q_tensor = torch.clamp(q_tensor, 0, 2 ** quant_bits - 1)
        return q_tensor

    def forward(self, x):
        """
        Forward pass of the router module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            x (torch.Tensor): Output router tensor. Last dim is 2, [0, 1] or [1, 0].
            eviction_mask (torch.Tensor): Output eviction mask for kv_cache, calculated from routes and will be applied on attention.

        """
        bsz, seqlen, _ = x.shape
        # x = self.asymmetric_quantization_for_similarity(x, quant_bits=4)
        x = self.linear_2(F.silu(self.norm(self.linear_1(x.float()))))
        if self.training:
            x = F.gumbel_softmax(x, dim=-1, tau=1, hard=True)
        else:
            x = F.softmax(x, dim=-1)
            x = torch.where(x < 0.5, 0, 1)

        x = x.bfloat16()
        
        x[:, :self.reserve_initials, :] = torch.Tensor([0, 1]).type_as(x)
        m = torch.zeros(2, seqlen).type_as(x)
        m[1, :] = 1
        eviction_mask = torch.matmul(x, m).transpose(2, 1)
        eviction_mask = eviction_mask.unsqueeze(1)

        return x, eviction_mask

# [Modified]
class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
            self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: ModelArgs_Vicuna):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (Linear): Linear transformation for queries.
            wk (Linear): Linear transformation for keys.
            wv (Linear): Linear transformation for values.
            wo (Linear): Linear transformation for output.

        """
        super().__init__()
        self.hidden_size = args.dim
        self.num_heads = args.n_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = args.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

        # self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # self.n_local_heads = args.n_heads
        # self.n_local_kv_heads = self.n_kv_heads
        # self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # self.head_dim = args.dim // args.n_heads
        # self.max_position_embeddings = args.max_position_embeddings
        #
        # self.wq = Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        # self.wk = Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # self.wv = Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        # self.wo = Linear(args.n_heads * self.head_dim, args.dim, bias=False)

        self.lora_q = LoRAModule(in_dim=args.dim, rank=args.lora_rank, out_dim=self.head_dim * self.num_heads)
        self.lora_k = LoRAModule(in_dim=args.dim, rank=args.lora_rank, out_dim=self.head_dim * self.num_heads)
        self.lora_v = LoRAModule(in_dim=args.dim, rank=args.lora_rank, out_dim=self.head_dim * self.num_heads)
        
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        causal_mask: Optional[torch.Tensor],
        eviction_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor], # [Modified]
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor.
            causal mask (torch.Tensor, optional): Attention causal mask tensor.
            eviction mask (torch.Tensor, optional): Attention eviction mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape

        xq = self.q_proj(x) + self.lora_q(x)
        xk = self.k_proj(x) + self.lora_k(x)
        xv = self.v_proj(x) + self.lora_v(x)

        # [Modified]
        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        xk = xk.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        xv = xv.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = xk.shape[-2]
        cos, sin = self.rotary_emb(xv, seq_len=kv_seq_len)
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin, position_ids)

        keys = xk
        values = xv

        # repeat k/v heads if n_kv_heads < n_heads, no repeat_kv for vicuna
        # keys = repeat_kv(keys, self.n_rep)
        # values = repeat_kv(values, self.n_rep)

        # xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        # xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        # xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        #
        # xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        #
        # keys = xk
        # values = xv
        #
        # # repeat k/v heads if n_kv_heads < n_heads
        # keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        # values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        #
        # xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        # keys = keys.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)
        # values = values.transpose(1, 2) # (bs, n_local_heads, cache_len + seqlen, head_dim)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if causal_mask is not None:
            scores = scores + causal_mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1)

        if eviction_mask is not None:
            scores = scores * eviction_mask
            scores = scores / (scores.sum(-1, keepdim=True) + 1e-6)
        scores = scores.type_as(xq)

        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (Linear): Linear transformation for the first layer.
            w2 (Linear): Linear transformation for the second layer.
            w3 (Linear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.gate_proj = Linear(dim, hidden_dim, bias=False)
        self.up_proj = Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

        # self.w1 = Linear(dim, hidden_dim, bias=False)
        # self.w2 = Linear(hidden_dim, dim, bias=False)
        # self.w3 = Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        # return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs_Vicuna):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.
            dynamic_start_layer (int): Indicate the layer start to apply router module.
            router (Optional[RouterModule]): Router module.


        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.self_attn = Attention(args)
        self.mlp = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.input_layernorm = LlamaRMSNorm(args.dim, eps=args.norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(args.dim, eps=args.norm_eps)

        self.dynamic_start_layer = args.dynamic_start_layer
        if self.layer_id >= self.dynamic_start_layer:
            self.router = RouterModule(args.dim, args.dynamic_router_hdim, args.dynamic_reserve_initials, args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor], # [Modified]
    ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Causal masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.
            w (torch.Tensor): Output routes after applying router module.

        """
        bsz, seqlen, _ = x.shape

        if self.layer_id < self.dynamic_start_layer:
            w = torch.ones((bsz, seqlen, 1)).type_as(x)

            h = x + self.self_attn(self.input_layernorm(x), start_pos, freqs_cis, mask, None,
                                   position_ids=position_ids)  # [Modified]
            out = h + self.mlp(self.post_attention_layernorm(h))

            return out, w
        else:
            w, eviction_mask = self.router(x)
            w = w[:, :, 1:2]

            h = x + self.self_attn(self.input_layernorm(x), start_pos, freqs_cis, mask, eviction_mask,
                                   position_ids=position_ids)  # [Modified]
            out = h + self.mlp(self.post_attention_layernorm(h))

            return out * w + x * (1 - w), w


class Transformer_vicuna(nn.Module):
    def __init__(self, params: ModelArgs_Vicuna):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            criterion (torch.nn.CrossEntropyLoss): CrossEntropy function.
            criterion_active (ActiveLoss): Active ratio function.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (Linear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        
        self.embed_tokens = Embedding(params.vocab_size, params.dim)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.criterion_active = ActiveLoss(
            target=params.dynamic_active_target,
            reserve_initials=params.dynamic_reserve_initials
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = LlamaRMSNorm(params.dim, eps=params.norm_eps)

        self.lm_head = Linear(params.dim, params.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )

    def forward(
        self,
        examples: torch.Tensor,
        labels: torch.Tensor,
        example_masks: torch.Tensor,
        label_masks: torch.Tensor,
        position_ids: Optional[torch.LongTensor] = None, # [Modified]
    ):
        """
        Perform a forward pass through the Transformer model.

        Args:
            examples (torch.Tensor): Input token indices representing whole sentences.
            labels (torch.Tensor): Input token indices representing labels.
            example_masks (torch.Tensor): Mask corresponding to sentences.
            label_masks (torch.Tensor): Mask corresponding to labels.

        Returns:
            c_loss (torch.Tensor): Output CE after applying the D-LLM.
            a_loss (torch.Tensor): Output active loss after applying the D-LLM.
            active_metric (Dict): Output metrics about activations after applying the D-LLM.

        """
        _bsz, seqlen = examples.shape

        h = self.embed_tokens(examples)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[:seqlen]

        mask = None
        mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=h.device)
        mask = torch.triu(mask, diagonal=0 + 1).type_as(h)
        
        start_pos = 0
        acts = []

        # [Modified]
        if position_ids is None:
            device = examples.device if examples is not None else h.device
            position_ids = torch.arange(
                0, seqlen + 0, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seqlen)
        else:
            position_ids = position_ids.view(-1, seqlen).long()

        for layer in self.layers:
            h, w = layer(h, start_pos, freqs_cis, mask, position_ids=position_ids) # [Modified]
            acts.append(w)
        activation = torch.cat(acts, dim=-1)

        h = self.norm(h)
        output = self.lm_head(h)
        output = output[:, :-1, :].reshape(-1, self.vocab_size)
        labels = labels[:, 1:].flatten()

        c_loss = self.criterion(output, labels)
        a_loss = self.criterion_active(activation[:, :-1, :], example_masks[:, 1:])
        active_metric = self.criterion_active.metric(activation[:, :-1, :], example_masks[:, 1:])
        
        return c_loss, a_loss, active_metric
