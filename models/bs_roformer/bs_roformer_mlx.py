# coding: utf-8
"""
Clean-sheet native Apple MLX implementation of BS-RoFormer (Band-Split Rotary Position Embedding Transformer).
Zero third-party wrapper dependencies; utilizes pure mlx.core, mlx.nn, and mlx.fast.
"""

import math
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
import mlx.core as mx
import mlx.nn as nn


def get_rotary_freqs(dim: int, max_seq_len: int, theta: float = 10000.0) -> mx.array:
    """Generates 2D frequency matrix for rotary positional embeddings."""
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
    t = mx.arange(max_seq_len).astype(mx.float32)
    freqs_outer = mx.outer(t, freqs)  # (max_seq_len, dim // 2)
    return mx.repeat(freqs_outer, 2, axis=-1)  # (max_seq_len, dim)


def rotate_half_mlx(x: mx.array) -> mx.array:
    """Rotates half of the feature dimensions for RoPE."""
    d = x.shape[-1]
    x_pairs = x.reshape(x.shape[:-1] + (d // 2, 2))
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]
    rotated = mx.stack([-x2, x1], axis=-1)
    return rotated.reshape(x.shape)


def apply_rotary_mlx(freqs: mx.array, t: mx.array) -> mx.array:
    """Applies rotary positional embeddings to queries or keys."""
    seq_len = t.shape[-2]
    freqs_slice = freqs[:seq_len]
    cos = mx.cos(freqs_slice)[None, None, :, :]
    sin = mx.sin(freqs_slice)[None, None, :, :]
    return (t * cos) + (rotate_half_mlx(t) * sin)


def mlx_stft(x: mx.array, n_fft: int = 2048, hop_length: int = 512, win_length: int = 2048) -> mx.array:
    """
    Computes real STFT matching torch.stft(..., return_complex=True) with reflect padding.
    Args:
        x: (B, T) float32 array
    Returns:
        (B, F, num_frames, 2) real-imag STFT representation
    """
    B, T = x.shape
    pad_amount = n_fft // 2
    left_pad = x[:, 1:pad_amount + 1][:, ::-1]
    right_pad = x[:, -pad_amount - 1:-1][:, ::-1]
    x_padded = mx.concatenate([left_pad, x, right_pad], axis=1)

    # Hann window
    n = mx.arange(win_length).astype(mx.float32)
    window = 0.5 * (1.0 - mx.cos(2.0 * math.pi * n / win_length))

    num_frames = 1 + (x_padded.shape[1] - win_length) // hop_length
    indices = mx.arange(win_length)[None, :] + mx.arange(num_frames)[:, None] * hop_length
    frames = x_padded[:, indices]
    windowed = frames * window

    spec = mx.fft.rfft(windowed, n=n_fft, axis=-1)
    out = mx.stack([spec.real, spec.imag], axis=-1)
    return mx.transpose(out, (0, 2, 1, 3))


def mlx_istft(
    spec_real_imag: mx.array,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int = 2048,
    length: Optional[int] = None,
) -> np.ndarray:
    """
    Computes inverse real STFT matching torch.istft with Hann window overlap-add synthesis.
    Args:
        spec_real_imag: (B, F, num_frames, 2)
    Returns:
        (B, length) numpy array
    """
    B, F, num_frames, _ = spec_real_imag.shape
    spec_c = spec_real_imag[..., 0] + 1j * spec_real_imag[..., 1]
    spec_c = mx.transpose(spec_c, (0, 2, 1))

    frames = mx.fft.irfft(spec_c, n=n_fft, axis=-1)[..., :win_length]
    n = mx.arange(win_length).astype(mx.float32)
    window = 0.5 * (1.0 - mx.cos(2.0 * math.pi * n / win_length))
    windowed_frames = frames * window

    total_samples = (num_frames - 1) * hop_length + win_length
    out_signal = np.zeros((B, total_samples), dtype=np.float32)
    window_sum = np.zeros((total_samples,), dtype=np.float32)

    wf_np = np.array(windowed_frames)
    w_sq = np.array(window) ** 2

    for t_idx in range(num_frames):
        start = t_idx * hop_length
        end = start + win_length
        out_signal[:, start:end] += wf_np[:, t_idx, :]
        window_sum[start:end] += w_sq

    window_sum = np.maximum(window_sum, 1e-7)
    recon = out_signal / window_sum[np.newaxis, :]

    pad_amount = n_fft // 2
    if length is not None:
        return recon[:, pad_amount:pad_amount + length]
    return recon[:, pad_amount:-pad_amount]


class RMSNormMLX(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-12):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        norm = mx.sqrt(mx.sum(x ** 2, axis=-1, keepdims=True) + self.eps)
        return (x / norm) * self.scale * self.gamma


class FeedForwardMLX(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        dim_inner = int(dim * mult)
        self.norm = RMSNormMLX(dim)
        self.linear1 = nn.Linear(dim, dim_inner)
        self.linear2 = nn.Linear(dim_inner, dim)

    def __call__(self, x: mx.array) -> mx.array:
        h = self.norm(x)
        h = self.linear1(h)
        h = nn.gelu(h)
        return self.linear2(h)


class AttentionMLX(nn.Module):
    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head
        self.norm = RMSNormMLX(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Linear(dim_inner, dim, bias=False)

    def __call__(self, x: mx.array, rotary_freqs: Optional[mx.array] = None) -> mx.array:
        B, N, D = x.shape
        h_norm = self.norm(x)
        qkv = self.to_qkv(h_norm)
        qkv = qkv.reshape(B, N, 3, self.heads, self.dim_head)

        q = qkv[:, :, 0].transpose(0, 2, 1, 3)
        k = qkv[:, :, 1].transpose(0, 2, 1, 3)
        v = qkv[:, :, 2].transpose(0, 2, 1, 3)

        if rotary_freqs is not None:
            q = apply_rotary_mlx(rotary_freqs, q)
            k = apply_rotary_mlx(rotary_freqs, k)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        gates = self.to_gates(h_norm).transpose(0, 2, 1)[:, :, :, None]
        out = out * mx.sigmoid(gates)

        out = out.transpose(0, 2, 1, 3).reshape(B, N, self.heads * self.dim_head)
        return self.to_out(out)


class TransformerBlockMLX(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, ff_mult: int = 4):
        super().__init__()
        self.attn = AttentionMLX(dim, heads, dim_head)
        self.ff = FeedForwardMLX(dim, ff_mult)

    def __call__(self, x: mx.array, rotary_freqs: Optional[mx.array] = None) -> mx.array:
        x = x + self.attn(x, rotary_freqs=rotary_freqs)
        x = x + self.ff(x)
        return x


class TransformerMLX(nn.Module):
    def __init__(self, depth: int, dim: int, heads: int, dim_head: int, ff_mult: int = 4):
        super().__init__()
        self.layers = [TransformerBlockMLX(dim, heads, dim_head, ff_mult) for _ in range(depth)]

    def __call__(self, x: mx.array, rotary_freqs: Optional[mx.array] = None) -> mx.array:
        for layer in self.layers:
            x = layer(x, rotary_freqs=rotary_freqs)
        return x


class BandSplitMLX(nn.Module):
    def __init__(self, dim: int, dim_inputs: Tuple[int, ...]):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features_norm = [RMSNormMLX(d) for d in dim_inputs]
        self.to_features_linear = [nn.Linear(d, dim) for d in dim_inputs]

    def __call__(self, x: mx.array) -> mx.array:
        outs = []
        start = 0
        for d_in, norm, lin in zip(self.dim_inputs, self.to_features_norm, self.to_features_linear):
            end = start + d_in
            band_x = x[:, :, start:end]
            outs.append(lin(norm(band_x)))
            start = end
        return mx.stack(outs, axis=2)


class MaskEstimatorMLX(nn.Module):
    def __init__(self, dim: int, dim_inputs: Tuple[int, ...], depth: int = 2, mlp_expansion: int = 4):
        super().__init__()
        self.dim_inputs = dim_inputs
        dim_hidden = dim * mlp_expansion
        self.to_freqs = []
        for d_in in dim_inputs:
            out_dim = d_in * 2
            if depth == 1:
                layers = [nn.Linear(dim, out_dim)]
            else:
                layers = [nn.Linear(dim, dim_hidden), nn.Tanh(), nn.Linear(dim_hidden, out_dim)]
            self.to_freqs.append(layers)

    def __call__(self, x: mx.array) -> mx.array:
        outs = []
        for band_idx, d_in in enumerate(self.dim_inputs):
            band_feat = x[:, :, band_idx, :]
            h = band_feat
            for layer in self.to_freqs[band_idx]:
                if isinstance(layer, nn.Linear):
                    h = layer(h)
                elif isinstance(layer, nn.Tanh):
                    h = mx.tanh(h)
            a = h[:, :, :d_in]
            b = h[:, :, d_in:]
            glu_out = a * mx.sigmoid(b)
            outs.append(glu_out)
        return mx.concatenate(outs, axis=-1)


class BSRoformerMLX(nn.Module):
    """Native MLX Band-Split RoFormer model for audio stem separation."""

    def __init__(
        self,
        dim: int = 384,
        depth: int = 8,
        stereo: bool = True,
        num_stems: int = 4,
        time_transformer_depth: int = 2,
        freq_transformer_depth: int = 2,
        freqs_per_bands: Tuple[int, ...] = (
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2,
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
            12, 12, 12, 12, 12, 12, 12, 12,
            24, 24, 24, 24, 24, 24, 24, 24,
            48, 48, 48, 48, 48, 48, 48, 48,
            128, 129,
        ),
        dim_head: int = 64,
        heads: int = 8,
        stft_n_fft: int = 2048,
        stft_hop_length: int = 512,
        stft_win_length: int = 2048,
        mask_estimator_depth: int = 2,
        mlp_expansion_factor: int = 4,
        zero_dc: bool = True,
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.dim = dim
        self.stft_n_fft = stft_n_fft
        self.stft_hop_length = stft_hop_length
        self.stft_win_length = stft_win_length
        self.zero_dc = zero_dc

        self.time_rotary_freqs = get_rotary_freqs(dim_head, max_seq_len)
        self.freq_rotary_freqs = get_rotary_freqs(dim_head, max_seq_len)

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)
        self.band_split = BandSplitMLX(dim=dim, dim_inputs=freqs_per_bands_with_complex)

        self.blocks = []
        for _ in range(depth):
            time_trans = TransformerMLX(
                depth=time_transformer_depth,
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=4,
            )
            freq_trans = TransformerMLX(
                depth=freq_transformer_depth,
                dim=dim,
                heads=heads,
                dim_head=dim_head,
                ff_mult=4,
            )
            self.blocks.append((time_trans, freq_trans))

        self.final_norm = RMSNormMLX(dim)
        self.mask_estimators = [
            MaskEstimatorMLX(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion=mlp_expansion_factor,
            )
            for _ in range(num_stems)
        ]

    def __call__(self, raw_audio: mx.array) -> np.ndarray:
        """
        Runs separation forward pass on raw audio input.
        Args:
            raw_audio: (B, channels, samples) or (channels, samples)
        Returns:
            numpy array of shape (B, num_stems, channels, samples) or (num_stems, channels, samples)
        """
        squeeze_batch = False
        if raw_audio.ndim == 2:
            raw_audio = raw_audio[None, ...]
            squeeze_batch = True

        B, S, T = raw_audio.shape
        raw_audio_flat = raw_audio.reshape(B * S, T)
        stft_repr = mlx_stft(
            raw_audio_flat,
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop_length,
            win_length=self.stft_win_length,
        )  # (B*S, F, num_frames, 2)

        F_bins = stft_repr.shape[1]
        num_frames = stft_repr.shape[2]

        stft_repr = stft_repr.reshape(B, S, F_bins, num_frames, 2)
        # Transpose and merge (F, S): (B, (F*S), num_frames, 2)
        stft_merged = stft_repr.transpose(0, 2, 1, 3, 4).reshape(B, F_bins * S, num_frames, 2)
        # Reshape to (B, num_frames, (F*S*2))
        x = stft_merged.transpose(0, 2, 1, 3).reshape(B, num_frames, F_bins * S * 2)

        x = self.band_split(x)  # (B, num_frames, num_bands, dim)
        num_bands = x.shape[2]

        for time_trans, freq_trans in self.blocks:
            # Time transformer: reshape to (B * num_bands, num_frames, dim)
            x_time = x.transpose(0, 2, 1, 3).reshape(B * num_bands, num_frames, self.dim)
            x_time = time_trans(x_time, rotary_freqs=self.time_rotary_freqs)
            x = x_time.reshape(B, num_bands, num_frames, self.dim).transpose(0, 2, 1, 3)

            # Freq transformer: reshape to (B * num_frames, num_bands, dim)
            x_freq = x.reshape(B * num_frames, num_bands, self.dim)
            x_freq = freq_trans(x_freq, rotary_freqs=self.freq_rotary_freqs)
            x = x_freq.reshape(B, num_frames, num_bands, self.dim)

        x = self.final_norm(x)

        # Estimate masks for each stem
        masks = [estimator(x) for estimator in self.mask_estimators]
        # masks: list of (B, num_frames, F*S*2)
        masks_stacked = mx.stack(masks, axis=1)  # (B, num_stems, num_frames, F*S*2)
        masks_reshaped = masks_stacked.reshape(B, self.num_stems, num_frames, F_bins * S, 2).transpose(0, 1, 3, 2, 4)
        # (B, num_stems, (F*S), num_frames, 2)

        stft_exp = stft_merged[:, None, ...]  # (B, 1, (F*S), num_frames, 2)

        # Complex multiplication: (R1*R2 - I1*I2, R1*I2 + I1*R2)
        r1, i1 = stft_exp[..., 0], stft_exp[..., 1]
        r2, i2 = masks_reshaped[..., 0], masks_reshaped[..., 1]
        r_out = r1 * r2 - i1 * i2
        i_out = r1 * i2 + i1 * r2
        stft_mod = mx.stack([r_out, i_out], axis=-1)  # (B, num_stems, F*S, num_frames, 2)

        # Separate stereo channels
        stft_mod = stft_mod.reshape(B, self.num_stems, F_bins, S, num_frames, 2).transpose(0, 1, 3, 2, 4, 5)
        # (B, num_stems, S, F_bins, num_frames, 2)

        if self.zero_dc:
            stft_mod = mx.concatenate([mx.zeros_like(stft_mod[:, :, :, :1, :, :]), stft_mod[:, :, :, 1:, :, :]], axis=3)

        stft_flat = stft_mod.reshape(B * self.num_stems * S, F_bins, num_frames, 2)
        recon_flat = mlx_istft(
            stft_flat,
            n_fft=self.stft_n_fft,
            hop_length=self.stft_hop_length,
            win_length=self.stft_win_length,
            length=T,
        )  # (B * num_stems * S, T)

        recon = recon_flat.reshape(B, self.num_stems, S, T)
        if squeeze_batch:
            return recon[0]
        return recon


def load_bs_roformer_mlx_from_ckpt(config: Dict[str, Any], ckpt_path: str) -> BSRoformerMLX:
    """
    Directly constructs BSRoformerMLX and populates weights from a PyTorch .ckpt file.
    No intermediate files, wrappers, or external translation tools needed.
    """
    import torch

    model_cfg = config.get("model", {})
    dim = model_cfg.get("dim", 384)
    depth = model_cfg.get("depth", 8)
    stereo = model_cfg.get("stereo", True)
    num_stems = model_cfg.get("num_stems", 4)
    time_depth = model_cfg.get("time_transformer_depth", 2)
    freq_depth = model_cfg.get("freq_transformer_depth", 2)
    dim_head = model_cfg.get("dim_head", 64)
    heads = model_cfg.get("heads", 8)
    stft_n_fft = model_cfg.get("stft_n_fft", 2048)
    stft_hop = model_cfg.get("stft_hop_length", 512)
    stft_win = model_cfg.get("stft_win_length", 2048)
    mask_depth = model_cfg.get("mask_estimator_depth", 2)
    mlp_expansion = model_cfg.get("mlp_expansion_factor", 4)
    freqs_per_bands = tuple(model_cfg.get("freqs_per_bands", (
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2,
        4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        12, 12, 12, 12, 12, 12, 12, 12,
        24, 24, 24, 24, 24, 24, 24, 24,
        48, 48, 48, 48, 48, 48, 48, 48,
        128, 129,
    )))

    model = BSRoformerMLX(
        dim=dim,
        depth=depth,
        stereo=stereo,
        num_stems=num_stems,
        time_transformer_depth=time_depth,
        freq_transformer_depth=freq_depth,
        freqs_per_bands=freqs_per_bands,
        dim_head=dim_head,
        heads=heads,
        stft_n_fft=stft_n_fft,
        stft_hop_length=stft_hop,
        stft_win_length=stft_win,
        mask_estimator_depth=mask_depth,
        mlp_expansion_factor=mlp_expansion,
    )

    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt_data, dict):
        if "state" in ckpt_data:
            state_dict = ckpt_data["state"]
        elif "state_dict" in ckpt_data:
            state_dict = ckpt_data["state_dict"]
        elif "model_state_dict" in ckpt_data:
            state_dict = ckpt_data["model_state_dict"]
        else:
            state_dict = ckpt_data
    else:
        state_dict = ckpt_data

    # Map PyTorch weights to MLX arrays
    # 1. BandSplit weights
    for band_idx in range(len(freqs_per_bands)):
        norm_key = f"band_split.to_features.{band_idx}.0.gamma"
        lin_w_key = f"band_split.to_features.{band_idx}.1.weight"
        lin_b_key = f"band_split.to_features.{band_idx}.1.bias"

        if norm_key in state_dict:
            model.band_split.to_features_norm[band_idx].gamma = mx.array(state_dict[norm_key].numpy())
        if lin_w_key in state_dict:
            model.band_split.to_features_linear[band_idx].weight = mx.array(state_dict[lin_w_key].numpy())
        if lin_b_key in state_dict and state_dict[lin_b_key] is not None:
            model.band_split.to_features_linear[band_idx].bias = mx.array(state_dict[lin_b_key].numpy())

    # 2. Transformer blocks
    for block_idx in range(depth):
        time_trans, freq_trans = model.blocks[block_idx]

        # Time transformer layers
        for layer_idx in range(time_depth):
            prefix = f"layers.{block_idx}.0.layers.{layer_idx}"
            _load_transformer_layer_weights(time_trans.layers[layer_idx], prefix, state_dict)

        # Freq transformer layers
        for layer_idx in range(freq_depth):
            prefix = f"layers.{block_idx}.1.layers.{layer_idx}"
            _load_transformer_layer_weights(freq_trans.layers[layer_idx], prefix, state_dict)

    # 3. Final norm
    if "final_norm.gamma" in state_dict:
        model.final_norm.gamma = mx.array(state_dict["final_norm.gamma"].numpy())

    # 4. Mask estimators
    for stem_idx in range(num_stems):
        estimator = model.mask_estimators[stem_idx]
        for band_idx in range(len(freqs_per_bands)):
            layers = estimator.to_freqs[band_idx]
            prefix = f"mask_estimators.{stem_idx}.to_freqs.{band_idx}.0"

            if len(layers) == 1:
                w_key = f"{prefix}.0.weight"
                b_key = f"{prefix}.0.bias"
                if w_key in state_dict:
                    layers[0].weight = mx.array(state_dict[w_key].numpy())
                if b_key in state_dict and state_dict[b_key] is not None:
                    layers[0].bias = mx.array(state_dict[b_key].numpy())
            else:
                w0_key = f"{prefix}.0.weight"
                b0_key = f"{prefix}.0.bias"
                w2_key = f"{prefix}.2.weight"
                b2_key = f"{prefix}.2.bias"

                if w0_key in state_dict:
                    layers[0].weight = mx.array(state_dict[w0_key].numpy())
                if b0_key in state_dict and state_dict[b0_key] is not None:
                    layers[0].bias = mx.array(state_dict[b0_key].numpy())
                if w2_key in state_dict:
                    layers[2].weight = mx.array(state_dict[w2_key].numpy())
                if b2_key in state_dict and state_dict[b2_key] is not None:
                    layers[2].bias = mx.array(state_dict[b2_key].numpy())

    return model


def _load_transformer_layer_weights(layer: TransformerBlockMLX, prefix: str, state_dict: Dict[str, Any]):
    """Helper to populate Attention and FeedForward weights from state dict."""
    # Attention
    attn_norm_key = f"{prefix}.0.norm.gamma"
    to_qkv_key = f"{prefix}.0.to_qkv.weight"
    to_gates_w_key = f"{prefix}.0.to_gates.weight"
    to_gates_b_key = f"{prefix}.0.to_gates.bias"
    to_out_w_key = f"{prefix}.0.to_out.0.weight"

    if attn_norm_key in state_dict:
        layer.attn.norm.gamma = mx.array(state_dict[attn_norm_key].numpy())
    if to_qkv_key in state_dict:
        layer.attn.to_qkv.weight = mx.array(state_dict[to_qkv_key].numpy())
    if to_gates_w_key in state_dict:
        layer.attn.to_gates.weight = mx.array(state_dict[to_gates_w_key].numpy())
    if to_gates_b_key in state_dict and state_dict[to_gates_b_key] is not None:
        layer.attn.to_gates.bias = mx.array(state_dict[to_gates_b_key].numpy())
    if to_out_w_key in state_dict:
        layer.attn.to_out.weight = mx.array(state_dict[to_out_w_key].numpy())

    # FeedForward
    ff_norm_key = f"{prefix}.1.net.0.gamma"
    ff_lin1_w_key = f"{prefix}.1.net.1.weight"
    ff_lin1_b_key = f"{prefix}.1.net.1.bias"
    ff_lin2_w_key = f"{prefix}.1.net.4.weight"
    ff_lin2_b_key = f"{prefix}.1.net.4.bias"

    if ff_norm_key in state_dict:
        layer.ff.norm.gamma = mx.array(state_dict[ff_norm_key].numpy())
    if ff_lin1_w_key in state_dict:
        layer.ff.linear1.weight = mx.array(state_dict[ff_lin1_w_key].numpy())
    if ff_lin1_b_key in state_dict and state_dict[ff_lin1_b_key] is not None:
        layer.ff.linear1.bias = mx.array(state_dict[ff_lin1_b_key].numpy())
    if ff_lin2_w_key in state_dict:
        layer.ff.linear2.weight = mx.array(state_dict[ff_lin2_w_key].numpy())
    if ff_lin2_b_key in state_dict and state_dict[ff_lin2_b_key] is not None:
        layer.ff.linear2.bias = mx.array(state_dict[ff_lin2_b_key].numpy())
