# from typing import Tuple, List, Dict, Optional
# from dataclasses import dataclass
# import math
# import torch
# import copy
# import torch.nn.functional as F
# from torch import nn
# from pydantic import BaseModel
# import random
# from models.common import trunc_normal_init_
# from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
# from models.sparse_embedding import CastedSparseEmbedding
# # from mamba_ssm import Mamba
# from mambapy.mamba import Mamba, MambaConfig


# IGNORE_LABEL_ID = -100

# @dataclass
# class TinyRecursiveReasoningModel_ACTV1InnerCarry:
#     z_H: torch.Tensor
#     z_L: torch.Tensor


# @dataclass
# class TinyRecursiveReasoningModel_ACTV1Carry:
#     inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
#     steps: torch.Tensor
#     halted: torch.Tensor
    
#     current_data: Dict[str, torch.Tensor]


# class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
#     batch_size: int
#     seq_len: int
#     puzzle_emb_ndim: int = 0
#     num_puzzle_identifiers: int
#     vocab_size: int

#     H_cycles: int
#     L_cycles: int

#     H_layers: int # ignored
#     L_layers: int

#     mc_samples: int = 2

#     # Transformer config
#     hidden_size: int
#     expansion: float
#     num_heads: int
#     pos_encodings: str

#     rms_norm_eps: float = 1e-5
#     rope_theta: float = 10000.0
    
#     # Halting Q-learning config
#     halt_max_steps: int
#     halt_exploration_prob: float

#     forward_dtype: str = "bfloat16"

#     # Alexia: added
#     mlp_t: bool = False # use mlp on L instead of transformer
#     puzzle_emb_len: int = 16 # if non-zero, its specified to this value
#     no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense



# # class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
# #     def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
# #         super().__init__()

# #         self.config = config
# #         if self.config.mlp_t:
# #             self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
# #             self.mlp_t = SwiGLU(
# #                 hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
# #                 expansion=config.expansion,
# #             )
# #         else:
# #             self.self_attn = Attention(
# #                 hidden_size=config.hidden_size,
# #                 head_dim=config.hidden_size // config.num_heads,
# #                 num_heads=config.num_heads,
# #                 num_key_value_heads=config.num_heads,
# #                 causal=False
# #             )
# #         self.mlp = SwiGLU(
# #             hidden_size=config.hidden_size,
# #             expansion=config.expansion,
# #         )
# #         self.norm_eps = config.rms_norm_eps

# #     def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
# #         # B, L, D = hidden_states.shape
# #         # Post Norm
# #         if self.config.mlp_t:
# #             hidden_states = hidden_states.transpose(1,2)
# #             out = self.mlp_t(hidden_states)
# #             hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
# #             hidden_states = hidden_states.transpose(1,2)
# #         else:
# #             # Self Attention
# #             hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
# #         # Fully Connected
# #         out = self.mlp(hidden_states)
# #         hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
# #         return hidden_states







# # class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
# #     def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config):
# #         super().__init__()
# #         self.config = config
# #         self.norm_eps = config.rms_norm_eps

# #         # === Replace transformer/MLP with Vision Mamba ===
# #         self.mamba = Mamba(
# #             d_model=config.hidden_size,
# #             d_state=16,          # Mamba state size (you can tune)
# #             d_conv=4,            # depthwise conv expand
# #             expand=2             # MLP expansion inside Mamba
# #         )

# #         # === Keep the feed-forward SwiGLU MLP ===
# #         self.mlp = SwiGLU(
# #             hidden_size=config.hidden_size,
# #             expansion=config.expansion,
# #         )

# #     def forward(self, cos_sin, hidden_states):
# #         # Mamba expects (B, L, D)
# #         residual = hidden_states

# #         # ---- Mamba core ----
# #         out = self.mamba(hidden_states)
# #         hidden_states = rms_norm(residual + out, variance_epsilon=self.norm_eps)

# #         # ---- Post-Mamba MLP ----
# #         residual = hidden_states
# #         out = self.mlp(hidden_states)
# #         hidden_states = rms_norm(residual + out, variance_epsilon=self.norm_eps)

# #         return hidden_states
















# # class VisionMambaBlock(nn.Module):
# #     """
# #     Bidirectional Mamba block: runs a forward Mamba and a backward Mamba,
# #     sums outputs and projects back to model dim.
# #     Input/Output shape: (B, L, D)
# #     """
# #     def __init__(self, d_model: int, d_state: int = 16, n_layers: int = 1):
# #         super().__init__()
# #         # Configure Mamba (adapt these args to your installed mambapy if necessary)
# #         cfg = MambaConfig(d_model=d_model, n_layers=n_layers)
# #         # Forward and backward scanners
# #         self.mamba_fwd = Mamba(cfg)
# #         self.mamba_bwd = Mamba(cfg)
# #         self.out_proj = nn.Linear(d_model, d_model)

# #     def forward(self, x: torch.Tensor) -> torch.Tensor:
# #         # x: (B, L, D)
# #         x_fwd = self.mamba_fwd(x)                       # forward scan
# #         x_rev = torch.flip(x, dims=[1])                 # reverse sequence
# #         x_bwd = self.mamba_bwd(x_rev)                   # backward scan on reversed
# #         x_bwd = torch.flip(x_bwd, dims=[1])             # flip back
# #         return self.out_proj(x_fwd + x_bwd)             # project sum back to D


# # class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
# #     """
# #     Replacement block for the ACTV1 model:
# #     - If config.mlp_t == True: keep the MLP-T branch behavior (SwiGLU on transposed input).
# #     - Else: replace self-attention with VisionMambaBlock (bidirectional Mamba).
# #     Always keep the SwiGLU feedforward `self.mlp`.
# #     """
# #     def __init__(self, config):
# #         super().__init__()
# #         self.config = config
# #         self.norm_eps = config.rms_norm_eps

# #         # If mlp_t is enabled, keep the MLP-T path (like your original code)
# #         if self.config.mlp_t:
# #             # puzzle_emb_len ceil-div trick (same as in your original code)
# #             self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) \
# #                                   if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
# #             self.mlp_t = SwiGLU(
# #                 hidden_size=self.config.seq_len + self.puzzle_emb_len,
# #                 expansion=self.config.expansion,
# #             )
# #         else:
# #             # Replace self-attention with Vision Mamba
# #             # You can tune d_state / n_layers here if desired (kept small by default)
# #             self.mamba = VisionMambaBlock(d_model=self.config.hidden_size,
# #                                           d_state=16,
# #                                           n_layers=1)

# #         # Keep the standard feed-forward SwiGLU
# #         self.mlp = SwiGLU(
# #             hidden_size=self.config.hidden_size,
# #             expansion=self.config.expansion,
# #         )

# #     def forward(self, cos_sin, hidden_states: torch.Tensor) -> torch.Tensor:
# #         # hidden_states: (B, L, D)
# #         # Post-Norm + mixer (either mlp_t or mamba)
# #         if self.config.mlp_t:
# #             # MLP-T branch (same behavior as your original block)
# #             hidden_states = hidden_states.transpose(1, 2)            # (B, D, L)
# #             out = self.mlp_t(hidden_states)
# #             hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
# #             hidden_states = hidden_states.transpose(1, 2)            # (B, L, D)
# #         else:
# #             # Vision Mamba branch replaces self-attention
# #             out = self.mamba(hidden_states)                          # (B, L, D)
# #             hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)

# #         # Fully connected / SwiGLU feed-forward
# #         out = self.mlp(hidden_states)
# #         hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)

# #         return hidden_states







# # below is new vision mamba block


# # class VisionMambaBlock(nn.Module):
# #     """
# #     Vision-Mamba style block operating on (B, L, D)
# #     - Uses bidirectional Mamba (forward + backward)
# #     - Final projection back to D
# #     """
# #     def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
# #         super().__init__()

# #         # Standard Mamba config
# #         self.mamba_fwd = Mamba(
# #             d_model=d_model,
# #             d_state=d_state,
# #             expand=expand,
# #         )

# #         # Backward Mamba for bidirectional effect
# #         self.mamba_bwd = Mamba(
# #             d_model=d_model,
# #             d_state=d_state,
# #             expand=expand,
# #         )

# #         self.proj = nn.Linear(d_model, d_model)

# #     def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
# #         # (B, L, D)
# #         out_fwd = self.mamba_fwd(hidden_states)

# #         # Reverse sequence for backward pass
# #         rev = torch.flip(hidden_states, dims=[1])
# #         out_bwd = self.mamba_bwd(rev)
# #         out_bwd = torch.flip(out_bwd, dims=[1])

# #         # Combine both
# #         out = out_fwd + out_bwd
# #         out = self.proj(out)
# #         return out







# # class VimBlock(nn.Module):
# #     """
# #     Bidirectional Mamba Block (Vision Mamba style).
# #     It processes the sequence forward and backward and fuses the results.
# #     """
# #     def _init_(self, config: TinyRecursiveReasoningModel_ACTV1Config):
# #         super()._init_()
# #         if Mamba is None:
# #             raise ImportError("mamba_ssm library is required for VimBlock")
            
# #         self.hidden_size = config.hidden_size
        
# #         # In Vim, we typically run two SSMs (forward and backward)
# #         # We can implement this by sharing weights or having separate directions.
# #         # The official implementation often uses bimamba_type="v2" which runs 
# #         # forward and backward branches. Here is a manual implementation of the bidirectional logic:
        
# #         self.forward_mamba = Mamba(
# #             d_model=config.hidden_size,
# #             d_state=config.d_state,
# #             d_conv=config.d_conv,
# #             expand=config.expansion, # Mamba expansion
# #         )
        
# #         self.backward_mamba = Mamba(
# #             d_model=config.hidden_size,
# #             d_state=config.d_state,
# #             d_conv=config.d_conv,
# #             expand=config.expansion,
# #         )
        
# #         # Project back to hidden_size after concatenating forward/backward 
# #         # (This depends on specific flavor of Vim, usually they are summed or gated. 
# #         # Here we follow a gated fusion approach for stability).
# #         self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

# #     def forward(self, x: torch.Tensor):
# #         # x: [B, L, D]
        
# #         # Forward pass
# #         out_fwd = self.forward_mamba(x)
        
# #         # Backward pass: flip sequence, run mamba, flip back
# #         x_rev = torch.flip(x, dims=[1])
# #         out_rev = self.backward_mamba(x_rev)
# #         out_rev = torch.flip(out_rev, dims=[1])
        
# #         # Fuse: Simple addition is standard for many BiMamba implementations
# #         # Alternatively, you can concat and project.
# #         output = out_fwd + out_rev
        
# #         return self.out_proj(output)


# # class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
# #     def _init_(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
# #         super()._init_()

# #         self.config = config
        
# #         # 1. MLP Transpose logic (unchanged)
# #         if self.config.mlp_t:
# #             self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
# #             self.mlp_t = SwiGLU(
# #                 hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
# #                 expansion=config.expansion,
# #             )
# #         else:
# #             # 2. REPLACED ATTENTION WITH VIM
# #             # Original: self.self_attn = Attention(...)
# #             self.mixer = VimBlock(config)

# #         self.mlp = SwiGLU(
# #             hidden_size=config.hidden_size,
# #             expansion=config.expansion,
# #         )
# #         self.norm_eps = config.rms_norm_eps

# #     def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
# #         # B, L, D = hidden_states.shape
        
# #         if self.config.mlp_t:
# #             # MLP mixing over Time dimension
# #             hidden_states = hidden_states.transpose(1,2)
# #             out = self.mlp_t(hidden_states)
# #             hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
# #             hidden_states = hidden_states.transpose(1,2)
# #         else:
# #             # Vision Mamba / SSM Mixing
# #             # Note: Mamba generally does NOT use Rotary Embeddings (CosSin).
# #             # It relies on the internal state and 1D convolution for positional awareness.
            
# #             mixer_out = self.mixer(hidden_states)
# #             hidden_states = rms_norm(hidden_states + mixer_out, variance_epsilon=self.norm_eps)
        
# #         # Fully Connected (Channel mixing)
# #         out = self.mlp(hidden_states)
# #         hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
# #         return hidden_states






# # below is mamba without library

# # class SimpleMamba(nn.Module):
# #     """
# #     Pure PyTorch fallback Mamba-like SSM block
# #     - No CUDA kernels
# #     - Works everywhere (HPC, CPU, GPU)
# #     - Drop-in replacement for mamba_ssm.Mamba
# #     """
# #     def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
# #         super().__init__()
# #         hidden = expand * d_model

# #         # gating branch
# #         self.in_proj = nn.Linear(d_model, hidden * 2)
# #         self.out_proj = nn.Linear(hidden, d_model)

# #         # SSM parameters
# #         self.A = nn.Parameter(torch.randn(hidden, d_state) * 0.01)
# #         self.B = nn.Parameter(torch.randn(hidden, d_state) * 0.01)
# #         self.C = nn.Parameter(torch.randn(hidden, d_state) * 0.01)

# #         # convolution for local mixing
# #         self.conv = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1, groups=hidden)

# #     def forward(self, x):
# #         # x: (B, L, D)
# #         B, L, D = x.shape

# #         # Gate and update
# #         u, gate = self.in_proj(x).chunk(2, dim=-1)  # (B, L, hidden)

# #         # Local convolution mixing (same as Mamba)
# #         u_conv = self.conv(u.transpose(1, 2)).transpose(1, 2)

# #         # Simple SSM recurrence: x_t = x_(t-1) * A + u * B
# #         # This is a slow but functional CPU-compatible version
# #         h = torch.zeros(B, self.A.size(0), device=x.device)
# #         outputs = []
# #         for t in range(L):
# #             h = h + u_conv[:, t] @ self.A + u_conv[:, t] @ self.B
# #             y = h @ self.C.t()
# #             outputs.append(y)

# #         y = torch.stack(outputs, dim=1)  # (B, L, hidden)

# #         # gating
# #         y = y * torch.sigmoid(gate)

# #         return self.out_proj(y)



# # class VisionMambaBlock(nn.Module):
# #     def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
# #         super().__init__()
# #         self.mamba_fwd = SimpleMamba(d_model, d_state, expand)
# #         self.mamba_bwd = SimpleMamba(d_model, d_state, expand)
# #         self.proj = nn.Linear(d_model, d_model)

# #     def forward(self, x):
# #         # Forward direction
# #         out_fwd = self.mamba_fwd(x)

# #         # Backward direction
# #         x_rev = torch.flip(x, dims=[1])
# #         out_bwd = self.mamba_bwd(x_rev)
# #         out_bwd = torch.flip(out_bwd, dims=[1])

# #         return self.proj(out_fwd + out_bwd)







# #### below is using just mamba library











# class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
#     """
#     Replacement block for the ACTV1 model:
#     - If config.mlp_t == True: keep the MLP-T branch behavior (SwiGLU on transposed input).
#     - Else: replace self-attention with VisionMambaBlock (bidirectional Mamba).
#     Always keep the SwiGLU feedforward `self.mlp`.
#     """
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
#         self.norm_eps = config.rms_norm_eps

#         # If mlp_t is enabled, keep the MLP-T path (like your original code)
#         if self.config.mlp_t:
#             # puzzle_emb_len ceil-div trick (same as in your original code)
#             self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) \
#                                   if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
#             self.mlp_t = SwiGLU(
#                 hidden_size=self.config.seq_len + self.puzzle_emb_len,
#                 expansion=self.config.expansion,
#             )
#         else:
#             # Replace self-attention with Vision Mamba
#             # You can tune d_state / n_layers here if desired (kept small by default)
#             self.mamba = VisionMambaBlock(d_model=self.config.hidden_size,
#                                           d_state=16,
#                                           n_layers=1)

#         # Keep the standard feed-forward SwiGLU
#         self.mlp = SwiGLU(
#             hidden_size=self.config.hidden_size,
#             expansion=self.config.expansion,
#         )

#     def forward(self, cos_sin, hidden_states: torch.Tensor) -> torch.Tensor:
#         # hidden_states: (B, L, D)
#         # Post-Norm + mixer (either mlp_t or mamba)
#         if self.config.mlp_t:
#             # MLP-T branch (same behavior as your original block)
#             hidden_states = hidden_states.transpose(1, 2)            # (B, D, L)
#             out = self.mlp_t(hidden_states)
#             hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
#             hidden_states = hidden_states.transpose(1, 2)            # (B, L, D)
#         else:
#             # Vision Mamba branch replaces self-attention
#             out = self.mamba(hidden_states)                          # (B, L, D)
#             hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)

#         # Fully connected / SwiGLU feed-forward
#         out = self.mlp(hidden_states)
#         hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)

#         return hidden_states













# class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
#     def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
#         super().__init__()
#         self.layers = torch.nn.ModuleList(layers)

#     def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
#         hidden_states = hidden_states + input_injection
#         for layer in self.layers:
#             hidden_states = layer(hidden_states=hidden_states, **kwargs)
#         return hidden_states


# class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
#     def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
#         super().__init__()
#         self.config = config
#         self.forward_dtype = getattr(torch, self.config.forward_dtype)

#         # I/O

#         self.embed_scale = math.sqrt(self.config.hidden_size)
#         embed_init_std = 1.0 / self.embed_scale

#         self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)

#         self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
#         self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

#         self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
#         # self.embed_tokens = CastedEmbedding(vocab_size=self.config.vocab_size, hidden_size=self.config.hidden_size, max_len=self.config.seq_len + self.puzzle_emb_len)

#         if self.config.puzzle_emb_ndim > 0:
#             # Zero init puzzle embeddings
#             self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
#                                                     batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

#         # LM Blocks
#         if self.config.pos_encodings == "rope":
#             self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
#                                               max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
#                                               base=self.config.rope_theta)
#         elif self.config.pos_encodings == "learned":
#             self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
#         else:
#             pass

#         # Reasoning Layers
#         self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

#         # Initial states
#         self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
#         self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

#         # Q head special init
#         # Init Q to (almost) zero for faster learning during bootstrapping
#         with torch.no_grad():
#             self.q_head.weight.zero_()
#             self.q_head.bias.fill_(-5)  # type: ignore

#     def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
#         # Token embedding
#         embedding = self.embed_tokens(input.to(torch.int32))

#         # Puzzle embeddings
#         if self.config.puzzle_emb_ndim > 0:
#             puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
#             pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
#             if pad_count > 0:
#                 puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

#             embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

#         # Position embeddings
#         if self.config.pos_encodings == "learned":
#             # scale by 1/sqrt(2) to maintain forward variance
#             embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

#         # Scale
#         return self.embed_scale * embedding

#     def empty_carry(self, batch_size: int):
#         return TinyRecursiveReasoningModel_ACTV1InnerCarry(
#             z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
#             z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
#         )
        
#     def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
#         return TinyRecursiveReasoningModel_ACTV1InnerCarry(
#             z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
#             z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
#         )

#     def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
#         seq_info = dict(
#             cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
#         )

#         # Input encoding
#         input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

#         # Forward iterations
#         it = 0
#         z_H, z_L = carry.z_H, carry.z_L
#         # H_cycles-1 without grad
#         with torch.no_grad():
#             for _H_step in range(self.config.H_cycles-1):
#                 for _L_step in range(self.config.L_cycles):
#                     z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
#                 z_H = self.L_level(z_H, z_L, **seq_info)
#         # 1 with grad
#         for _L_step in range(self.config.L_cycles):
#             z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
#         z_H = self.L_level(z_H, z_L, **seq_info)

#         # LM Outputs
#         new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
#         output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
#         q_logits = self.q_head(z_H[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
#         return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


# class TinyRecursiveReasoningModel_ACTV1(nn.Module):
#     """ACT wrapper."""

#     def __init__(self, config_dict: dict):
#         super().__init__()
#         self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
#         self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

#     @property
#     def puzzle_emb(self):
#         return self.inner.puzzle_emb

#     def initial_carry(self, batch: Dict[str, torch.Tensor]):
#         batch_size = batch["inputs"].shape[0]

#         return TinyRecursiveReasoningModel_ACTV1Carry(
#             inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
#             steps=torch.zeros((batch_size, ), dtype=torch.int32),
#             halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
#             current_data={k: torch.empty_like(v) for k, v in batch.items()}
#         )
        
#     def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

#         # Update data, carry (removing halted sequences)
#         new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
#         new_steps = torch.where(carry.halted, 0, carry.steps)

#         new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}





#         # ---------- Monte-Carlo wrapper (robust) ----------
#         if self.training and self.config.mc_samples > 1:
#             K = self.config.mc_samples
#             B = new_current_data["inputs"].shape[0]
        
#             batch_expanded = {}
#             for k, v in new_current_data.items():
#                 if k == "puzzle_identifiers":
#                     if v.ndim == 1:
#                         batch_expanded[k] = v.repeat_interleave(K, dim=0)
#                     else:
#                         batch_expanded[k] = v.unsqueeze(1).repeat(1, K, *([1] * (v.ndim - 1))).view(-1, *v.shape[1:])
#                 else:
#                     batch_expanded[k] = v.unsqueeze(1).repeat(1, K, *([1] * (v.ndim - 1))).view(-1, *v.shape[1:])

        
#             # expand carry tensors across particles
#             inner_carry_expanded = TinyRecursiveReasoningModel_ACTV1InnerCarry(
#                 z_H=new_inner_carry.z_H.unsqueeze(1).repeat(1, K, 1, 1).view(-1, *new_inner_carry.z_H.shape[1:]),
#                 z_L=new_inner_carry.z_L.unsqueeze(1).repeat(1, K, 1, 1).view(-1, *new_inner_carry.z_L.shape[1:]),
#             )
        
#             # forward inner model in parallel for the K samples
#             inner_carry_out, logits_mc, q_logits = self.inner(inner_carry_expanded, batch_expanded)
        
#             # reshape and aggregate
#             logits_mc = logits_mc.view(B, K, *logits_mc.shape[1:])
#             q_halt_mc = q_logits[0].view(B, K)
#             q_continue_mc = q_logits[1].view(B, K)
        
#             logits = logits_mc.mean(dim=1)
#             q_halt_logits = q_halt_mc.mean(dim=1)
#             q_continue_logits = q_continue_mc.mean(dim=1)
        
#             # take the first particle's carry state as representative (keep same strategy you used)
#             new_inner_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(
#                 z_H=inner_carry_out.z_H[::K],
#                 z_L=inner_carry_out.z_L[::K],
#             )
#         else:
#             # normal (non-MC) path
#             new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)
#         # ---------------------------------------------------







#         outputs = {
#             "logits": logits,
#             "q_halt_logits": q_halt_logits,
#             "q_continue_logits": q_continue_logits
#         }

#         with torch.no_grad():
#             # Step
#             new_steps = new_steps + 1
#             is_last_step = new_steps >= self.config.halt_max_steps
            
#             halted = is_last_step

#             # if training, and ACT is enabled
#             if self.training and (self.config.halt_max_steps > 1):

#                 # Halt signal
#                 # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
#                 if self.config.no_ACT_continue:
#                     halted = halted | (q_halt_logits > 0)
#                 else:
#                     halted = halted | (q_halt_logits > q_continue_logits)

#                 # Exploration
#                 min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
#                 halted = halted & (new_steps >= min_halt_steps)

#                 if not self.config.no_ACT_continue:
#                     # Compute target Q
#                     # NOTE: No replay buffer and target networks for computing target Q-value.
#                     # As batch_size is large, there're many parallel envs.
#                     # Similar concept as PQN https://arxiv.org/abs/2407.04811
#                     _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
#                     outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

#         return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs




























from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from models.common import trunc_normal_init_
from models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    num_puzzle_identifiers: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense

class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

# ... (ReasoningModule, Inner, and Wrapper classes remain unchanged) ...
# To ensure the code runs, I include the ReasoningModule wrapper below 
# so the block integration is clear.

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
        if self.config.puzzle_emb_ndim > 0:
            # Zero init puzzle embeddings
            self.puzzle_emb = CastedSparseEmbedding(self.config.num_puzzle_identifiers, self.config.puzzle_emb_ndim,
                                                    batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))

            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])

        # Forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),  # Empty is expected, it will be reseted in first pass as all sequences are halted.
            
            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),  # Default to halted
            
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)
        
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {k: torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _, _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
