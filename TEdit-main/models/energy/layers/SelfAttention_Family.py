import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from .utils.masking import TriangularCausalMask, ProbMask
# from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat


class DSAttention(nn.Module):
    '''De-stationary Attention with Soft-Boundary Injection Support'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None,
                soft_mask=None, keys_null=None, values_null=None):
        """
        De-stationary Attention with Soft-Boundary Injection.
        
        Core Formula:
            A_inj = Λ ⊙ A_edit + (I - Λ) ⊙ A_null
            V_inj = Λ ⊙ (A_edit @ V_edit) + (I - Λ) ⊙ (A_null @ V_null)
        
        Args:
            queries: [B, L, H, E] - Query tensor
            keys: [B, S, H, E] - Key tensor (edit condition)
            values: [B, S, H, D] - Value tensor (edit condition)
            attn_mask: Attention mask
            tau: De-stationary factor
            delta: De-stationary delta
            soft_mask: [B, L] or [B, L, 1] - Soft boundary mask
            keys_null: [B, S, H, E] - Keys from null/background condition
            values_null: [B, S, H, D] - Values from null/background condition
        
        Returns:
            V: [B, L, H, D] - Output tensor
            A: [B, H, L, S] or None - Attention map
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)

        # Compute edit condition attention scores with de-stationary factors
        scores_edit = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        # ==================== Innovation: Semantic Isolation ====================
        if soft_mask is not None and keys_null is not None and values_null is not None:
            # Compute null/background condition attention scores
            scores_null = torch.einsum("blhe,bshe->bhls", queries, keys_null) * tau + delta
            
            if self.mask_flag:
                if attn_mask is None:
                    attn_mask = TriangularCausalMask(B, L, device=queries.device)
                scores_edit.masked_fill_(attn_mask.mask, -np.inf)
                scores_null.masked_fill_(attn_mask.mask, -np.inf)

            # Compute softmax attention maps
            A_edit = self.dropout(torch.softmax(scale * scores_edit, dim=-1))
            A_null = self.dropout(torch.softmax(scale * scores_null, dim=-1))

            # Align soft_mask dimensions for broadcasting
            if soft_mask.dim() == 2:
                mask_map = soft_mask.unsqueeze(1).unsqueeze(-1)
            elif soft_mask.dim() == 3:
                mask_map = soft_mask.unsqueeze(1)
            else:
                mask_map = soft_mask

            # Compute value outputs for both trajectories
            out_edit = torch.einsum("bhls,bshd->blhd", A_edit, values)
            out_null = torch.einsum("bhls,bshd->blhd", A_null, values_null)

            # Final output blending (Semantic Isolation)
            V = mask_map * out_edit + (1 - mask_map) * out_null

            # Compute blended attention map for visualization
            A_final = mask_map * A_edit + (1 - mask_map) * A_null

            if self.output_attention:
                return V.contiguous(), A_final
            else:
                return V.contiguous(), None
        # ==================== End Innovation ====================

        # Original logic for backward compatibility
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores_edit.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores_edit, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None,
                soft_mask=None, keys_null=None, values_null=None):
        """
        Modified Forward for Soft-Boundary Injection (Semantic Isolation).
        
        Core Formula:
            A_inj = Λ ⊙ A_edit + (I - Λ) ⊙ A_null
            V_inj = Λ ⊙ (A_edit @ V_edit) + (I - Λ) ⊙ (A_null @ V_null)
        
        This implements the "Commander" block in Figure 2, ensuring that
        edit instructions (e.g., "surge") do not leak into background regions.
        
        Args:
            queries: [B, L, H, E] - Query tensor
            keys: [B, S, H, E] - Key tensor (edit condition)
            values: [B, S, H, D] - Value tensor (edit condition)
            attn_mask: Attention mask
            tau: De-stationary factor
            delta: De-stationary delta
            soft_mask: [B, L] or [B, L, 1] - Soft boundary mask (Green curve in Figure 2)
            keys_null: [B, S, H, E] - Keys from null/background condition
            values_null: [B, S, H, D] - Values from null/background condition
        
        Returns:
            V: [B, L, H, D] - Output tensor
            A: [B, H, L, S] or None - Attention map
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # 1. Compute standard Attention Scores (foreground/edit condition)
        scores_edit = torch.einsum("blhe,bshe->bhls", queries, keys)

        # ==================== Innovation: Semantic Isolation ====================
        if soft_mask is not None and keys_null is not None and values_null is not None:
            # Formula: A_inj = Λ ⊙ A(Q, K_edit) + (I - Λ) ⊙ A(Q, K_null)
            
            # 2. Compute background/null condition Attention Scores
            scores_null = torch.einsum("blhe,bshe->bhls", queries, keys_null)
            
            # Handle Mask (causal mask etc.)
            if self.mask_flag:
                if attn_mask is None:
                    attn_mask = TriangularCausalMask(B, L, device=queries.device)
                scores_edit.masked_fill_(attn_mask.mask, -np.inf)
                scores_null.masked_fill_(attn_mask.mask, -np.inf)

            # 3. Compute Softmax Attention Maps
            A_edit = self.dropout(torch.softmax(scale * scores_edit, dim=-1))
            A_null = self.dropout(torch.softmax(scale * scores_null, dim=-1))

            # 4. Align soft_mask dimensions for broadcasting
            # Input: [B, L] or [B, L, 1] -> Output: [B, 1, L, 1] for [B, H, L, S]
            if soft_mask.dim() == 2:
                mask_map = soft_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, L, 1]
            elif soft_mask.dim() == 3:
                mask_map = soft_mask.unsqueeze(1)  # [B, 1, L, 1]
            else:
                mask_map = soft_mask  # Already [B, 1, L, S] or similar
            
            # 5. Compute Value outputs for both trajectories
            out_edit = torch.einsum("bhls,bshd->blhd", A_edit, values)
            out_null = torch.einsum("bhls,bshd->blhd", A_null, values_null)
            
            # 6. Final Output Blending (Semantic Isolation)
            V = mask_map * out_edit + (1 - mask_map) * out_null
            
            # 7. Compute blended attention map for visualization
            A_final = mask_map * A_edit + (1 - mask_map) * A_null

            if self.output_attention:
                return V.contiguous(), A_final
            else:
                return V.contiguous(), None
        # ==================== End Innovation ====================

        # Original logic for backward compatibility
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores_edit.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores_edit, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None,
                soft_mask=None, keys_null=None, values_null=None):
        """
        Forward pass with Soft-Boundary Injection support.
        
        Args:
            queries: Query tensor [B, L, d_model]
            keys: Key tensor [B, S, d_model] (edit condition)
            values: Value tensor [B, S, d_model] (edit condition)
            attn_mask: Attention mask
            tau: De-stationary factor
            delta: De-stationary delta
            soft_mask: Soft boundary mask [B, L] or [B, L, 1]
            keys_null: Key tensor for null/background condition [B, S, d_model]
            values_null: Value tensor for null/background condition [B, S, d_model]
        
        Returns:
            Tuple of (output tensor, attention map)
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # Project null condition keys/values if provided
        keys_null_proj = None
        values_null_proj = None
        if keys_null is not None and values_null is not None:
            keys_null_proj = self.key_projection(keys_null).view(B, S, H, -1)
            values_null_proj = self.value_projection(values_null).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta,
            soft_mask=soft_mask,
            keys_null=keys_null_proj,
            values_null=values_null_proj
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=configs.output_attention), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=configs.output_attention), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=configs.output_attention), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out
