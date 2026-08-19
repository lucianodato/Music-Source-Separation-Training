# coding: utf-8
"""
Clean-sheet native Apple MLX implementation of HTDemucs (Hybrid Transformer Demucs v4).
Zero third-party wrapper dependencies; utilizes pure mlx.core and mlx.nn on Apple Silicon Metal.
"""

import sys
import math
import numpy as np
from typing import Tuple, List, Dict, Optional, Any
import mlx.core as mx
import mlx.nn as nn


def mlx_pad1d(x: mx.array, paddings: Tuple[int, int], mode: str = "constant", value: float = 0.0) -> mx.array:
    """1D padding along the last axis."""
    pad_left, pad_right = paddings
    if pad_left == 0 and pad_right == 0:
        return x
    if mode == "constant":
        return mx.pad(x, [(0, 0)] * (x.ndim - 1) + [(pad_left, pad_right)], constant_values=value)
    elif mode == "reflect":
        left = x[..., 1:pad_left + 1][..., ::-1] if pad_left > 0 else mx.zeros((*x.shape[:-1], 0))
        right = x[..., -pad_right - 1:-1][..., ::-1] if pad_right > 0 else mx.zeros((*x.shape[:-1], 0))
        return mx.concatenate([left, x, right], axis=-1)
    else:
        raise NotImplementedError(f"Padding mode {mode} not implemented.")


def mlx_spectro(x: mx.array, n_fft: int = 4096, hop_length: Optional[int] = None) -> mx.array:
    """
    Computes complex STFT matching Demucs spectro with Hann window and reflect padding.
    Args:
        x: (..., length) float32
    Returns:
        (..., freqs, frames, 2) real-imag STFT
    """
    hl = hop_length or (n_fft // 4)
    orig_shape = x.shape
    length = orig_shape[-1]
    x_flat = x.reshape(-1, length)
    B = x_flat.shape[0]

    pad_amount = n_fft // 2
    x_padded = mlx_pad1d(x_flat, (pad_amount, pad_amount), mode="reflect")

    win_length = n_fft
    n = mx.arange(win_length).astype(mx.float32)
    window = 0.5 * (1.0 - mx.cos(2.0 * math.pi * n / win_length))

    num_frames = 1 + (x_padded.shape[-1] - win_length) // hl
    indices = mx.arange(win_length)[None, :] + mx.arange(num_frames)[:, None] * hl
    frames = x_padded[:, indices] * window

    spec = mx.fft.rfft(frames, n=n_fft, axis=-1)
    spec = spec / math.sqrt(n_fft)

    out = mx.stack([spec.real, spec.imag], axis=-1)
    out = mx.transpose(out, (0, 2, 1, 3))  # (B, freqs, num_frames, 2)

    freqs = out.shape[1]
    return out.reshape(*orig_shape[:-1], freqs, num_frames, 2)


def mlx_ispectro(spec_real_imag: mx.array, hop_length: Optional[int] = None, length: Optional[int] = None) -> mx.array:
    """
    Computes inverse real STFT matching Demucs ispectro with vectorized Hann window synthesis.
    Args:
        spec_real_imag: (..., freqs, frames, 2)
    Returns:
        (..., length) float32 mx.array
    """
    orig_shape = spec_real_imag.shape
    freqs = orig_shape[-3]
    num_frames = orig_shape[-2]
    n_fft = 2 * freqs - 2
    hl = hop_length or (n_fft // 4)

    z_flat = spec_real_imag.reshape(-1, freqs, num_frames, 2)
    B = z_flat.shape[0]

    spec_c = z_flat[..., 0] + 1j * z_flat[..., 1]
    spec_c = spec_c * math.sqrt(n_fft)
    spec_c = mx.transpose(spec_c, (0, 2, 1))

    win_length = n_fft
    frames = mx.fft.irfft(spec_c, n=n_fft, axis=-1)[..., :win_length]
    n = mx.arange(win_length).astype(mx.float32)
    window = 0.5 * (1.0 - mx.cos(2.0 * math.pi * n / win_length))
    windowed = frames * window

    total_samples = (num_frames - 1) * hl + win_length
    out_signal = mx.zeros((B, total_samples), dtype=mx.float32)
    window_sum = mx.zeros((total_samples,), dtype=mx.float32)
    w_sq = window ** 2

    K = win_length // hl
    for k in range(K):
        k_frames = windowed[:, k::K, :]
        num_k = k_frames.shape[1]
        if num_k > 0:
            k_sig = k_frames.reshape(B, num_k * win_length)
            k_start = k * hl
            k_end = k_start + num_k * win_length
            pad_left = k_start
            pad_right = total_samples - k_end
            out_signal = out_signal + mx.pad(k_sig, [(0, 0), (pad_left, pad_right)])
            k_w_sq = mx.repeat(w_sq[None, :], num_k, axis=0).reshape(-1)
            window_sum = window_sum + mx.pad(k_w_sq, [(pad_left, pad_right)])

    window_sum = mx.maximum(window_sum, 1e-7)
    recon = out_signal / window_sum[None, :]

    pad_amount = n_fft // 2
    if length is not None:
        out = recon[:, pad_amount:pad_amount + length]
    else:
        out = recon[:, pad_amount:-pad_amount]

    return out.reshape(*orig_shape[:-3], out.shape[-1])


def create_sin_embedding_mlx(length: int, dim: int, shift: int = 0, max_period: float = 10000.0) -> mx.array:
    """Generates 1D sinusoidal positional embeddings."""
    pos = (shift + mx.arange(length)).reshape(-1, 1, 1).astype(mx.float32)
    half_dim = dim // 2
    adim = mx.arange(half_dim).reshape(1, 1, -1).astype(mx.float32)
    phase = pos / (max_period ** (adim / (half_dim - 1)))
    return mx.concatenate([mx.cos(phase), mx.sin(phase)], axis=-1)


def create_2d_sin_embedding_mlx(d_model: int, height: int, width: int, max_period: float = 10000.0) -> mx.array:
    """Generates 2D sinusoidal positional embeddings for frequency-time planes matching Demucs."""
    half_d = d_model // 2
    div_term = mx.exp(mx.arange(0.0, half_d, 2) * -(math.log(max_period) / half_d))

    pos_w = mx.arange(float(width))[:, None]
    pos_h = mx.arange(float(height))[:, None]

    sin_w = mx.sin(pos_w * div_term)
    cos_w = mx.cos(pos_w * div_term)
    sin_h = mx.sin(pos_h * div_term)
    cos_h = mx.cos(pos_h * div_term)

    # Interleave sin and cos
    pe_w = mx.stack([sin_w, cos_w], axis=-1).reshape(width, half_d)
    pe_h = mx.stack([sin_h, cos_h], axis=-1).reshape(height, half_d)

    # Broadcast: (height, width, half_d)
    pe_w_bc = mx.repeat(pe_w[None, :, :], height, axis=0)
    pe_h_bc = mx.repeat(pe_h[:, None, :], width, axis=1)

    pe = mx.concatenate([pe_w_bc, pe_h_bc], axis=-1)  # (height, width, d_model)
    return pe[None, ...]  # (1, height, width, d_model)


class LayerScaleMLX(nn.Module):
    """Learnable affine per-channel scaling."""

    def __init__(self, channels: int, init_value: float = 1e-4):
        super().__init__()
        self.scale = mx.ones((channels,)) * float(init_value)

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.scale


class DConvMLX(nn.Module):
    """Multi-depth dilated 1D convolution residual block in Demucs."""

    def __init__(self, channels: int, compress: float = 4, depth: int = 2, init: float = 1e-4, kernel: int = 3):
        super().__init__()
        self.depth = abs(depth)
        hidden = int(channels / compress)
        self.layers = []

        for d in range(self.depth):
            dilation = 2 ** d if depth > 0 else 1
            padding = dilation * (kernel // 2)
            mods = {
                "conv1": nn.Conv1d(channels, hidden, kernel, dilation=dilation, padding=padding),
                "norm1": nn.GroupNorm(1, hidden, pytorch_compatible=True),
                "conv2": nn.Conv1d(hidden, 2 * channels, 1),
                "norm2": nn.GroupNorm(1, 2 * channels, pytorch_compatible=True),
                "scale": LayerScaleMLX(channels, init),
            }
            self.layers.append(mods)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, L, C)
        for layer in self.layers:
            h = layer["conv1"](x)
            h = layer["norm1"](h)
            h = nn.gelu(h)
            h = layer["conv2"](h)
            h = layer["norm2"](h)
            a, b = mx.split(h, 2, axis=-1)
            h = a * mx.sigmoid(b)  # GLU
            h = layer["scale"](h)
            x = x + h
        return x


class HEncLayerMLX(nn.Module):
    """Hybrid Encoder Layer for 2D spectrogram or 1D waveform."""

    def __init__(
        self,
        chin: int,
        chout: int,
        kernel_size: int = 8,
        stride: int = 4,
        norm_groups: int = 1,
        empty: bool = False,
        freq: bool = True,
        dconv: bool = True,
        norm: bool = True,
        context: int = 0,
        dconv_kw: Dict[str, Any] = {},
        pad: bool = True,
        rewrite: bool = True,
    ):
        super().__init__()
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.norm = norm
        self.pad_val = (kernel_size // 4) if pad else 0

        if freq:
            self.conv = nn.Conv2d(chin, chout, (kernel_size, 1), stride=(stride, 1), padding=(self.pad_val, 0))
        else:
            self.conv = nn.Conv1d(chin, chout, kernel_size, stride=stride, padding=self.pad_val)

        if self.empty:
            return

        self.norm1 = nn.GroupNorm(norm_groups, chout, pytorch_compatible=True) if norm else None
        self.dconv = DConvMLX(chout, **dconv_kw) if dconv else None

        self.rewrite = None
        if rewrite:
            k = 1 + 2 * context
            p = context
            if freq:
                self.rewrite = nn.Conv2d(chout, 2 * chout, (k, k), stride=(1, 1), padding=(p, p))
            else:
                self.rewrite = nn.Conv1d(chout, 2 * chout, k, stride=1, padding=p)
            self.norm2 = nn.GroupNorm(norm_groups, 2 * chout, pytorch_compatible=True) if norm else None

    def __call__(self, x: mx.array, inject: Optional[mx.array] = None) -> mx.array:
        # Frequency mode: x is (B, Fr, T, C)
        # Time mode: x is (B, T, C)
        if not self.freq:
            le = x.shape[1]
            if le % self.stride != 0:
                pad_r = self.stride - (le % self.stride)
                x = mx.pad(x, [(0, 0), (0, pad_r), (0, 0)])

        y = self.conv(x)
        if self.empty:
            return y

        if inject is not None:
            if inject.ndim == 3 and y.ndim == 4:
                inject = inject[:, None, :, :]  # (B, 1, T, C)
            y = y + inject

        if self.norm1 is not None:
            y = self.norm1(y)
        y = nn.gelu(y)

        if self.dconv is not None:
            if self.freq:
                B, Fr, T, C = y.shape
                y_d = y.reshape(B * Fr, T, C)
                y_d = self.dconv(y_d)
                y = y_d.reshape(B, Fr, T, C)
            else:
                y = self.dconv(y)

        if self.rewrite is not None:
            z = self.rewrite(y)
            if self.norm2 is not None:
                z = self.norm2(z)
            a, b = mx.split(z, 2, axis=-1)
            z = a * mx.sigmoid(b)  # GLU
        else:
            z = y
        return z


class HDecLayerMLX(nn.Module):
    """Hybrid Decoder Layer for 2D spectrogram or 1D waveform."""

    def __init__(
        self,
        chin: int,
        chout: int,
        last: bool = False,
        kernel_size: int = 8,
        stride: int = 4,
        norm_groups: int = 1,
        empty: bool = False,
        freq: bool = True,
        dconv: bool = True,
        norm: bool = True,
        context: int = 1,
        dconv_kw: Dict[str, Any] = {},
        pad: bool = True,
        context_freq: bool = True,
        rewrite: bool = True,
    ):
        super().__init__()
        self.pad_val = (kernel_size // 4) if pad else 0
        self.last = last
        self.freq = freq
        self.chin = chin
        self.empty = empty
        self.stride = stride
        self.kernel_size = kernel_size
        self.norm = norm
        self.context_freq = context_freq

        if freq:
            self.conv_tr = nn.ConvTranspose2d(chin, chout, (kernel_size, 1), stride=(stride, 1))
        else:
            self.conv_tr = nn.ConvTranspose1d(chin, chout, kernel_size, stride=stride)

        self.norm2 = nn.GroupNorm(norm_groups, chout, pytorch_compatible=True) if norm else None

        if self.empty:
            return

        self.rewrite = None
        if rewrite:
            k = 1 + 2 * context
            p = context
            if freq:
                if context_freq:
                    self.rewrite = nn.Conv2d(chin, 2 * chin, (k, k), stride=(1, 1), padding=(p, p))
                else:
                    self.rewrite = nn.Conv2d(chin, 2 * chin, (1, k), stride=(1, 1), padding=(0, p))
            else:
                self.rewrite = nn.Conv1d(chin, 2 * chin, k, stride=1, padding=p)
            self.norm1 = nn.GroupNorm(norm_groups, 2 * chin, pytorch_compatible=True) if norm else None

        self.dconv = DConvMLX(chin, **dconv_kw) if dconv else None

    def __call__(self, x: mx.array, skip: Optional[mx.array], length: int) -> Tuple[mx.array, mx.array]:
        if not self.empty:
            if skip is not None:
                x = x + skip
            if self.rewrite is not None:
                y = self.rewrite(x)
                if self.norm1 is not None:
                    y = self.norm1(y)
                a, b = mx.split(y, 2, axis=-1)
                y = a * mx.sigmoid(b)
            else:
                y = x

            if self.dconv is not None:
                if self.freq:
                    B, Fr, T, C = y.shape
                    y_d = y.reshape(B * Fr, T, C)
                    y_d = self.dconv(y_d)
                    y = y_d.reshape(B, Fr, T, C)
                else:
                    y = self.dconv(y)
        else:
            y = x

        z = self.conv_tr(y)
        if self.norm2 is not None:
            z = self.norm2(z)

        if self.freq:
            if self.pad_val > 0:
                z = z[:, self.pad_val:-self.pad_val, :, :]
        else:
            z = z[:, self.pad_val:self.pad_val + length, :]

        if not self.last:
            z = nn.gelu(z)

        return z, y


class MyGroupNormMLX(nn.Module):
    """GroupNorm over (T, C) dimensions for each sample B matching Demucs MyGroupNorm(1, C)."""

    def __init__(self, channels: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((channels,))
        self.bias = mx.zeros((channels,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        mean = mx.mean(x, axis=(1, 2), keepdims=True)
        var = mx.var(x, axis=(1, 2), keepdims=True)
        x_norm = (x - mean) / mx.sqrt(var + self.eps)
        return x_norm * self.weight + self.bias


class MyTransformerEncoderLayerMLX(nn.Module):
    """Self-Attention Transformer Encoder layer."""

    def __init__(self, dim: int, nhead: int = 8, dim_feedforward: int = 1536, layer_scale: bool = True, init_values: float = 1e-4, norm_out: bool = True):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.head_dim = dim // nhead
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm_out = MyGroupNormMLX(dim) if norm_out else None
        self.in_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.gamma1 = LayerScaleMLX(dim, init_values) if layer_scale else None
        self.gamma2 = LayerScaleMLX(dim, init_values) if layer_scale else None

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, T, C)
        h = self.norm1(x)
        B, T, C = h.shape
        qkv = self.in_proj(h)
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.nhead, self.head_dim).transpose(0, 2, 1, 3)

        attn = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(self.head_dim))
        attn = attn.transpose(0, 2, 1, 3).reshape(B, T, C)
        attn = self.out_proj(attn)
        if self.gamma1 is not None:
            attn = self.gamma1(attn)
        x = x + attn

        # FFN
        h = self.norm2(x)
        ff = self.linear2(nn.gelu(self.linear1(h)))
        if self.gamma2 is not None:
            ff = self.gamma2(ff)
        x = x + ff

        if self.norm_out is not None:
            x = self.norm_out(x)
        return x


class CrossTransformerEncoderLayerMLX(nn.Module):
    """Cross-Attention Transformer Encoder layer."""

    def __init__(self, dim: int, nhead: int = 8, dim_feedforward: int = 1536, layer_scale: bool = True, init_values: float = 1e-4, norm_out: bool = True):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.head_dim = dim // nhead
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm_out = MyGroupNormMLX(dim) if norm_out else None

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.gamma1 = LayerScaleMLX(dim, init_values) if layer_scale else None
        self.gamma2 = LayerScaleMLX(dim, init_values) if layer_scale else None

    def __call__(self, q: mx.array, k: mx.array) -> mx.array:
        norm_q = self.norm1(q)
        norm_k = self.norm2(k)
        B, Tq, C = norm_q.shape
        Tk = norm_k.shape[1]

        q_proj = self.q_proj(norm_q).reshape(B, Tq, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        k_proj = self.k_proj(norm_k).reshape(B, Tk, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        v_proj = self.v_proj(norm_k).reshape(B, Tk, self.nhead, self.head_dim).transpose(0, 2, 1, 3)

        attn = mx.fast.scaled_dot_product_attention(q_proj, k_proj, v_proj, scale=1.0 / math.sqrt(self.head_dim))
        attn = attn.transpose(0, 2, 1, 3).reshape(B, Tq, C)
        attn = self.out_proj(attn)
        if self.gamma1 is not None:
            attn = self.gamma1(attn)
        x = q + attn

        # FFN
        h = self.norm3(x)
        ff = self.linear2(nn.gelu(self.linear1(h)))
        if self.gamma2 is not None:
            ff = self.gamma2(ff)
        x = x + ff

        if self.norm_out is not None:
            x = self.norm_out(x)
        return x


class CrossTransformerEncoderMLX(nn.Module):
    """Full CrossTransformer module alternating self and cross attention between spectrum and waveform."""

    def __init__(self, dim: int, hidden_scale: float = 4.0, num_heads: int = 8, num_layers: int = 5, layer_scale: bool = True):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        hidden_dim = int(dim * hidden_scale)

        self.norm_in = nn.LayerNorm(dim)
        self.norm_in_t = nn.LayerNorm(dim)

        self.layers = []
        self.layers_t = []

        for idx in range(num_layers):
            if idx % 2 == 0:
                self.layers.append(MyTransformerEncoderLayerMLX(dim, num_heads, hidden_dim, layer_scale=layer_scale))
                self.layers_t.append(MyTransformerEncoderLayerMLX(dim, num_heads, hidden_dim, layer_scale=layer_scale))
            else:
                self.layers.append(CrossTransformerEncoderLayerMLX(dim, num_heads, hidden_dim, layer_scale=layer_scale))
                self.layers_t.append(CrossTransformerEncoderLayerMLX(dim, num_heads, hidden_dim, layer_scale=layer_scale))

    def __call__(self, x: mx.array, xt: mx.array) -> Tuple[mx.array, mx.array]:
        # x: (B, Fr, T1, C)
        # xt: (B, T2, C)
        B, Fr, T1, C = x.shape
        B, T2, C = xt.shape

        pos_emb_2d = create_2d_sin_embedding_mlx(C, Fr, T1)  # (1, Fr, T1, C)
        pos_emb_2d = pos_emb_2d.transpose(0, 2, 1, 3).reshape(1, T1 * Fr, C)

        x_seq = x.transpose(0, 2, 1, 3).reshape(B, T1 * Fr, C)
        x_seq = self.norm_in(x_seq) + pos_emb_2d

        pos_emb_1d = create_sin_embedding_mlx(T2, C).transpose(1, 0, 2)  # (1, T2, C)
        xt_seq = self.norm_in_t(xt) + pos_emb_1d

        for idx in range(self.num_layers):
            if idx % 2 == 0:
                x_seq = self.layers[idx](x_seq)
                xt_seq = self.layers_t[idx](xt_seq)
            else:
                old_x = x_seq
                x_seq = self.layers[idx](x_seq, xt_seq)
                xt_seq = self.layers_t[idx](xt_seq, old_x)

        x_out = x_seq.reshape(B, T1, Fr, C).transpose(0, 2, 1, 3)
        return x_out, xt_seq


class HTDemucsMLX(nn.Module):
    """
    Clean-sheet native Apple MLX HTDemucs Model.
    """

    def __init__(
        self,
        sources: List[str] = ["drums", "bass", "other", "vocals"],
        audio_channels: int = 2,
        channels: int = 48,
        growth: int = 2,
        nfft: int = 4096,
        depth: int = 4,
        kernel_size: int = 8,
        stride: int = 4,
        time_stride: int = 2,
        norm_starts: int = 4,
        norm_groups: int = 4,
        dconv_depth: int = 2,
        dconv_comp: int = 8,
        dconv_init: float = 1e-3,
        t_layers: int = 5,
        t_heads: int = 8,
        t_hidden_scale: float = 4.0,
        bottom_channels: Optional[int] = None,
        freq_emb: Optional[float] = None,
        emb_scale: float = 10.0,
    ):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.channels = channels
        self.growth = growth
        self.nfft = nfft
        self.hop_length = nfft // 4
        self.depth = depth
        self.bottom_channels = bottom_channels
        self.freq_emb_scale = (freq_emb * emb_scale) if freq_emb else None
        self.freq_emb = nn.Embedding(nfft // 2, channels) if freq_emb else None

        self.encoder = []
        self.tencoder = []
        self.decoder = []
        self.tdecoder = []

        chin = audio_channels
        chin_z = chin * 2  # Complex as Channels
        chout = channels
        chout_z = channels
        freqs = nfft // 2

        for index in range(depth):
            norm = index >= norm_starts
            freq = freqs > 1
            stri = stride
            ker = kernel_size
            if not freq:
                ker = time_stride * 2
                stri = time_stride

            pad = True
            last_freq = False
            if freq and freqs <= kernel_size:
                ker = freqs
                pad = False
                last_freq = True

            dconv_kw = {"depth": dconv_depth, "compress": dconv_comp, "init": dconv_init}

            if last_freq:
                chout_z = max(chout, chout_z)
                chout = chout_z

            enc = HEncLayerMLX(
                chin_z, chout_z, kernel_size=ker, stride=stri, norm_groups=norm_groups,
                freq=freq, dconv=True, norm=norm, context=0, dconv_kw=dconv_kw, pad=pad, rewrite=True
            )
            self.encoder.append(enc)

            if freq:
                tenc = HEncLayerMLX(
                    chin, chout, kernel_size=kernel_size, stride=stride, norm_groups=norm_groups,
                    freq=False, dconv=True, norm=norm, context=0, dconv_kw=dconv_kw, pad=True,
                    rewrite=True, empty=last_freq
                )
                self.tencoder.append(tenc)

            if index == 0:
                chin_out = audio_channels * len(sources)
                chin_z_out = chin_out * 2
            else:
                chin_out = chin
                chin_z_out = chin_z

            dec = HDecLayerMLX(
                chout_z, chin_z_out, last=(index == 0), kernel_size=ker, stride=stri,
                norm_groups=norm_groups, freq=freq, dconv=True, norm=norm, context=1,
                dconv_kw=dconv_kw, pad=pad, rewrite=True
            )
            self.decoder.insert(0, dec)

            if freq:
                tdec = HDecLayerMLX(
                    chout, chin_out, last=(index == 0), kernel_size=kernel_size, stride=stride,
                    norm_groups=norm_groups, freq=False, dconv=True, norm=norm, context=1,
                    dconv_kw=dconv_kw, pad=True, rewrite=True, empty=last_freq
                )
                self.tdecoder.insert(0, tdec)

            chin = chout
            chin_z = chout_z
            chout = int(growth * chout)
            chout_z = int(growth * chout_z)
            if freq:
                if freqs <= kernel_size:
                    freqs = 1
                else:
                    freqs //= stride

        transformer_channels = channels * growth ** (depth - 1)
        if self.bottom_channels:
            self.channel_upsampler = nn.Linear(transformer_channels, self.bottom_channels)
            self.channel_downsampler = nn.Linear(self.bottom_channels, transformer_channels)
            self.channel_upsampler_t = nn.Linear(transformer_channels, self.bottom_channels)
            self.channel_downsampler_t = nn.Linear(self.bottom_channels, transformer_channels)
            transformer_channels = self.bottom_channels

        if t_layers > 0:
            self.crosstransformer = CrossTransformerEncoderMLX(
                dim=transformer_channels,
                hidden_scale=t_hidden_scale,
                num_heads=t_heads,
                num_layers=t_layers,
                layer_scale=True,
            )
        else:
            self.crosstransformer = None

    def __call__(self, mix: mx.array) -> np.ndarray:
        """
        Inference forward pass.
        Args:
            mix: (B, C, L) or (C, L)
        Returns:
            (B, num_sources, C, L) numpy array
        """
        squeeze_batch = False
        if mix.ndim == 2:
            mix = mix[None, ...]
            squeeze_batch = True

        B, S, length = mix.shape
        hl = self.hop_length
        le = int(math.ceil(length / hl))
        pad = hl // 2 * 3

        # Spectrogram representation
        mix_padded = mlx_pad1d(mix, (pad, pad + le * hl - length), mode="reflect")
        z = mlx_spectro(mix_padded, n_fft=self.nfft, hop_length=hl)[..., :-1, :, :]  # remove highest Nyquist bin
        z = z[..., 2:2 + le, :]  # (B, S, Fr, T, 2)

        Fr = z.shape[2]
        T_spec = z.shape[3]
        x = z.transpose(0, 1, 4, 2, 3).reshape(B, S * 2, Fr, T_spec)

        # Normalization
        mean = mx.mean(x, axis=(1, 2, 3), keepdims=True)
        std = mx.std(x, axis=(1, 2, 3), keepdims=True)
        x = (x - mean) / (1e-5 + std)

        # Time branch input
        xt = mix
        meant = mx.mean(xt, axis=(1, 2), keepdims=True)
        stdt = mx.std(xt, axis=(1, 2), keepdims=True)
        xt = (xt - meant) / (1e-5 + stdt)

        # Transpose to MLX NHWC / NLC layout
        x = x.transpose(0, 2, 3, 1)  # (B, Fr, T, C)
        xt = xt.transpose(0, 2, 1)  # (B, T, C)

        saved = []
        saved_t = []
        lengths = []
        lengths_t = []

        # 1. Encoders
        for idx, enc in enumerate(self.encoder):
            lengths.append(x.shape[2])
            inject = None
            if idx < len(self.tencoder):
                lengths_t.append(xt.shape[1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt

            x = enc(x, inject=inject)
            if idx == 0 and self.freq_emb is not None:
                frs = mx.arange(x.shape[1])
                emb = self.freq_emb(frs)[None, :, None, :]  # (1, Fr, 1, C)
                x = x + self.freq_emb_scale * emb
            saved.append(x)

        # 2. CrossTransformer
        if self.crosstransformer is not None:
            if self.bottom_channels:
                x = self.channel_upsampler(x)
                xt = self.channel_upsampler_t(xt)

            x, xt = self.crosstransformer(x, xt)

            if self.bottom_channels:
                x = self.channel_downsampler(x)
                xt = self.channel_downsampler_t(xt)

        # 3. Decoders
        offset = self.depth - len(self.tdecoder)
        for idx, dec in enumerate(self.decoder):
            skip = saved.pop(-1)
            length_spec = lengths.pop(-1)
            x, pre = dec(x, skip, length_spec)

            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    pre_t = pre[:, 0, :, :]  # shape: (B, T, C)
                    xt, _ = tdec(pre_t, None, length_t)
                else:
                    skip_t = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip_t, length_t)

        # 4. Denormalization & Synthesis
        num_sources = len(self.sources)
        x = x.transpose(0, 3, 1, 2)  # (B, C_out, Fr, T)
        x = x * (1e-5 + std) + mean

        x = x.reshape(B, num_sources, S, 2, Fr, T_spec).transpose(0, 1, 2, 4, 5, 3)  # (B, S_src, S_ch, Fr, T, 2)

        x_pad = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, 1), (2, 2), (0, 0)])
        le_ispec = hl * int(math.ceil(length / hl)) + 2 * pad
        x_wav = mlx_ispectro(x_pad, hop_length=hl, length=le_ispec)  # (B, num_sources, S_ch, le_ispec)
        x_wav = x_wav[..., pad:pad + length]

        # Time branch reconstruction
        xt = xt.transpose(0, 2, 1)  # (B, C_out, T)
        xt = xt * (1e-5 + stdt) + meant
        xt = xt.reshape(B, num_sources, S, -1)
        xt = xt[..., :length]

        out = x_wav + xt
        if squeeze_batch:
            return out[0]
        return out


def load_htdemucs_mlx_from_ckpt(config: Dict[str, Any], ckpt_path: str) -> HTDemucsMLX:
    """
    Loads HTDemucs weights from a PyTorch .th / .ckpt file into HTDemucsMLX.
    """
    import torch

    ht_cfg = config.get("htdemucs", {})
    training_cfg = config.get("training", {})
    sources = training_cfg.get("instruments", ["drums", "bass", "other", "vocals"])
    if training_cfg.get("target_instrument"):
        sources = [training_cfg.get("target_instrument")]

    channels = int(ht_cfg.get("channels", 48))
    growth = int(ht_cfg.get("growth", 2))
    nfft = int(ht_cfg.get("nfft", 4096))
    depth = int(ht_cfg.get("depth", 4))
    kernel_size = int(ht_cfg.get("kernel_size", 8))
    stride = int(ht_cfg.get("stride", 4))
    time_stride = int(ht_cfg.get("time_stride", 2))
    norm_starts = int(ht_cfg.get("norm_starts", 4))
    norm_groups = int(ht_cfg.get("norm_groups", 4))
    dconv_depth = int(ht_cfg.get("dconv_depth", 2))
    dconv_comp = int(ht_cfg.get("dconv_comp", 8))
    dconv_init = float(ht_cfg.get("dconv_init", 1e-3))
    t_layers = int(ht_cfg.get("t_layers", 5))
    t_heads = int(ht_cfg.get("t_heads", 8))
    t_hidden_scale = float(ht_cfg.get("t_hidden_scale", 4.0))
    bottom_channels = ht_cfg.get("bottom_channels")
    if bottom_channels is not None:
        bottom_channels = int(bottom_channels)
    freq_emb = float(ht_cfg.get("freq_emb", 0.0)) if ht_cfg.get("freq_emb") else None
    emb_scale = float(ht_cfg.get("emb_scale", 10.0))

    model = HTDemucsMLX(
        sources=sources,
        audio_channels=2,
        channels=channels,
        growth=growth,
        nfft=nfft,
        depth=depth,
        kernel_size=kernel_size,
        stride=stride,
        time_stride=time_stride,
        norm_starts=norm_starts,
        norm_groups=norm_groups,
        dconv_depth=dconv_depth,
        dconv_comp=dconv_comp,
        dconv_init=dconv_init,
        t_layers=t_layers,
        t_heads=t_heads,
        t_hidden_scale=t_hidden_scale,
        bottom_channels=bottom_channels,
        freq_emb=freq_emb,
        emb_scale=emb_scale,
    )

    # Ensure safe unpickling even when third-party 'demucs' or 'openunmix' is not installed
    import types

    class _DummyClass:
        def __new__(cls, *args, **kwargs):
            return super().__new__(cls)

    class _AutoModule(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            cls = type(name, (_DummyClass,), {})
            setattr(self, name, cls)
            return cls

    for mod_name in ['demucs', 'demucs.htdemucs', 'demucs.hdemucs', 'demucs.transformer', 'demucs.states', 'demucs.apply', 'demucs.demucs', 'openunmix', 'openunmix.filtering']:
        if mod_name not in sys.modules or sys.modules[mod_name] is None:
            m = _AutoModule(mod_name)
            sys.modules[mod_name] = m

    if 'demucs' in sys.modules and hasattr(sys.modules['demucs'], '__dict__'):
        for sub in ['htdemucs', 'hdemucs', 'transformer', 'states', 'apply', 'demucs']:
            setattr(sys.modules['demucs'], sub, sys.modules.get(f'demucs.{sub}'))

    ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(ckpt_data, dict):
        if "state" in ckpt_data:
            state_dict = ckpt_data["state"]
        elif "state_dict" in ckpt_data:
            state_dict = ckpt_data["state_dict"]
        elif "model" in ckpt_data:
            state_dict = ckpt_data["model"]
        else:
            state_dict = ckpt_data
    else:
        state_dict = getattr(ckpt_data, "state_dict", lambda: ckpt_data)()

    def _load_dconv(mlx_dconv: DConvMLX, pfx: str):
        for d_idx, layer_mods in enumerate(mlx_dconv.layers):
            l_pfx = f"{pfx}.layers.{d_idx}"
            if f"{l_pfx}.0.weight" in state_dict:
                layer_mods["conv1"].weight = mx.array(state_dict[f"{l_pfx}.0.weight"].numpy().transpose(0, 2, 1))
            if f"{l_pfx}.0.bias" in state_dict and state_dict[f"{l_pfx}.0.bias"] is not None:
                layer_mods["conv1"].bias = mx.array(state_dict[f"{l_pfx}.0.bias"].numpy())

            if f"{l_pfx}.1.weight" in state_dict:
                layer_mods["norm1"].weight = mx.array(state_dict[f"{l_pfx}.1.weight"].numpy())
            if f"{l_pfx}.1.bias" in state_dict:
                layer_mods["norm1"].bias = mx.array(state_dict[f"{l_pfx}.1.bias"].numpy())

            if f"{l_pfx}.3.weight" in state_dict:
                layer_mods["conv2"].weight = mx.array(state_dict[f"{l_pfx}.3.weight"].numpy().transpose(0, 2, 1))
            if f"{l_pfx}.3.bias" in state_dict and state_dict[f"{l_pfx}.3.bias"] is not None:
                layer_mods["conv2"].bias = mx.array(state_dict[f"{l_pfx}.3.bias"].numpy())

            if f"{l_pfx}.4.weight" in state_dict:
                layer_mods["norm2"].weight = mx.array(state_dict[f"{l_pfx}.4.weight"].numpy())
            if f"{l_pfx}.4.bias" in state_dict:
                layer_mods["norm2"].bias = mx.array(state_dict[f"{l_pfx}.4.bias"].numpy())

            if f"{l_pfx}.6.scale" in state_dict:
                layer_mods["scale"].scale = mx.array(state_dict[f"{l_pfx}.6.scale"].numpy())

    # Map Encoders
    for idx, enc in enumerate(model.encoder):
        pfx = f"encoder.{idx}"
        if f"{pfx}.conv.weight" in state_dict:
            enc.conv.weight = mx.array(state_dict[f"{pfx}.conv.weight"].numpy().transpose(0, 2, 3, 1))
        if f"{pfx}.conv.bias" in state_dict and state_dict[f"{pfx}.conv.bias"] is not None:
            enc.conv.bias = mx.array(state_dict[f"{pfx}.conv.bias"].numpy())

        if not enc.empty:
            if f"{pfx}.norm1.weight" in state_dict and enc.norm1 is not None:
                enc.norm1.weight = mx.array(state_dict[f"{pfx}.norm1.weight"].numpy())
            if f"{pfx}.norm1.bias" in state_dict and enc.norm1 is not None:
                enc.norm1.bias = mx.array(state_dict[f"{pfx}.norm1.bias"].numpy())

            if enc.dconv is not None:
                _load_dconv(enc.dconv, f"{pfx}.dconv")

            if enc.rewrite is not None:
                if f"{pfx}.rewrite.weight" in state_dict:
                    enc.rewrite.weight = mx.array(state_dict[f"{pfx}.rewrite.weight"].numpy().transpose(0, 2, 3, 1))
                if f"{pfx}.rewrite.bias" in state_dict and state_dict[f"{pfx}.rewrite.bias"] is not None:
                    enc.rewrite.bias = mx.array(state_dict[f"{pfx}.rewrite.bias"].numpy())
                if f"{pfx}.norm2.weight" in state_dict and enc.norm2 is not None:
                    enc.norm2.weight = mx.array(state_dict[f"{pfx}.norm2.weight"].numpy())
                if f"{pfx}.norm2.bias" in state_dict and enc.norm2 is not None:
                    enc.norm2.bias = mx.array(state_dict[f"{pfx}.norm2.bias"].numpy())

    # Map Time Encoders
    for idx, tenc in enumerate(model.tencoder):
        pfx = f"tencoder.{idx}"
        if f"{pfx}.conv.weight" in state_dict:
            tenc.conv.weight = mx.array(state_dict[f"{pfx}.conv.weight"].numpy().transpose(0, 2, 1))
        if f"{pfx}.conv.bias" in state_dict and state_dict[f"{pfx}.conv.bias"] is not None:
            tenc.conv.bias = mx.array(state_dict[f"{pfx}.conv.bias"].numpy())

        if not tenc.empty:
            if f"{pfx}.norm1.weight" in state_dict and tenc.norm1 is not None:
                tenc.norm1.weight = mx.array(state_dict[f"{pfx}.norm1.weight"].numpy())
            if f"{pfx}.norm1.bias" in state_dict and tenc.norm1 is not None:
                tenc.norm1.bias = mx.array(state_dict[f"{pfx}.norm1.bias"].numpy())

            if tenc.dconv is not None:
                _load_dconv(tenc.dconv, f"{pfx}.dconv")

            if tenc.rewrite is not None:
                if f"{pfx}.rewrite.weight" in state_dict:
                    tenc.rewrite.weight = mx.array(state_dict[f"{pfx}.rewrite.weight"].numpy().transpose(0, 2, 1))
                if f"{pfx}.rewrite.bias" in state_dict and state_dict[f"{pfx}.rewrite.bias"] is not None:
                    tenc.rewrite.bias = mx.array(state_dict[f"{pfx}.rewrite.bias"].numpy())
                if f"{pfx}.norm2.weight" in state_dict and tenc.norm2 is not None:
                    tenc.norm2.weight = mx.array(state_dict[f"{pfx}.norm2.weight"].numpy())
                if f"{pfx}.norm2.bias" in state_dict and tenc.norm2 is not None:
                    tenc.norm2.bias = mx.array(state_dict[f"{pfx}.norm2.bias"].numpy())

    # Map Decoders
    for idx, dec in enumerate(model.decoder):
        pfx = f"decoder.{idx}"
        if f"{pfx}.conv_tr.weight" in state_dict:
            dec.conv_tr.weight = mx.array(state_dict[f"{pfx}.conv_tr.weight"].numpy().transpose(1, 2, 3, 0))
        if f"{pfx}.conv_tr.bias" in state_dict and state_dict[f"{pfx}.conv_tr.bias"] is not None:
            dec.conv_tr.bias = mx.array(state_dict[f"{pfx}.conv_tr.bias"].numpy())

        if dec.norm2 is not None:
            if f"{pfx}.norm2.weight" in state_dict:
                dec.norm2.weight = mx.array(state_dict[f"{pfx}.norm2.weight"].numpy())
            if f"{pfx}.norm2.bias" in state_dict:
                dec.norm2.bias = mx.array(state_dict[f"{pfx}.norm2.bias"].numpy())

        if not dec.empty:
            if dec.rewrite is not None:
                if f"{pfx}.rewrite.weight" in state_dict:
                    dec.rewrite.weight = mx.array(state_dict[f"{pfx}.rewrite.weight"].numpy().transpose(0, 2, 3, 1))
                if f"{pfx}.rewrite.bias" in state_dict and state_dict[f"{pfx}.rewrite.bias"] is not None:
                    dec.rewrite.bias = mx.array(state_dict[f"{pfx}.rewrite.bias"].numpy())
                if f"{pfx}.norm1.weight" in state_dict and dec.norm1 is not None:
                    dec.norm1.weight = mx.array(state_dict[f"{pfx}.norm1.weight"].numpy())
                if f"{pfx}.norm1.bias" in state_dict and dec.norm1 is not None:
                    dec.norm1.bias = mx.array(state_dict[f"{pfx}.norm1.bias"].numpy())

            if dec.dconv is not None:
                _load_dconv(dec.dconv, f"{pfx}.dconv")

    # Map Time Decoders
    for idx, tdec in enumerate(model.tdecoder):
        pfx = f"tdecoder.{idx}"
        if f"{pfx}.conv_tr.weight" in state_dict:
            tdec.conv_tr.weight = mx.array(state_dict[f"{pfx}.conv_tr.weight"].numpy().transpose(1, 2, 0))
        if f"{pfx}.conv_tr.bias" in state_dict and state_dict[f"{pfx}.conv_tr.bias"] is not None:
            tdec.conv_tr.bias = mx.array(state_dict[f"{pfx}.conv_tr.bias"].numpy())

        if tdec.norm2 is not None:
            if f"{pfx}.norm2.weight" in state_dict:
                tdec.norm2.weight = mx.array(state_dict[f"{pfx}.norm2.weight"].numpy())
            if f"{pfx}.norm2.bias" in state_dict:
                tdec.norm2.bias = mx.array(state_dict[f"{pfx}.norm2.bias"].numpy())

        if not tdec.empty:
            if tdec.rewrite is not None:
                if f"{pfx}.rewrite.weight" in state_dict:
                    tdec.rewrite.weight = mx.array(state_dict[f"{pfx}.rewrite.weight"].numpy().transpose(0, 2, 1))
                if f"{pfx}.rewrite.bias" in state_dict and state_dict[f"{pfx}.rewrite.bias"] is not None:
                    tdec.rewrite.bias = mx.array(state_dict[f"{pfx}.rewrite.bias"].numpy())
                if f"{pfx}.norm1.weight" in state_dict and tdec.norm1 is not None:
                    tdec.norm1.weight = mx.array(state_dict[f"{pfx}.norm1.weight"].numpy())
                if f"{pfx}.norm1.bias" in state_dict and tdec.norm1 is not None:
                    tdec.norm1.bias = mx.array(state_dict[f"{pfx}.norm1.bias"].numpy())

            if tdec.dconv is not None:
                _load_dconv(tdec.dconv, f"{pfx}.dconv")

    # Map CrossTransformer
    if model.crosstransformer is not None:
        ct = model.crosstransformer
        pfx = "crosstransformer"
        if f"{pfx}.norm_in.weight" in state_dict:
            ct.norm_in.weight = mx.array(state_dict[f"{pfx}.norm_in.weight"].numpy())
        if f"{pfx}.norm_in.bias" in state_dict:
            ct.norm_in.bias = mx.array(state_dict[f"{pfx}.norm_in.bias"].numpy())
        if f"{pfx}.norm_in_t.weight" in state_dict:
            ct.norm_in_t.weight = mx.array(state_dict[f"{pfx}.norm_in_t.weight"].numpy())
        if f"{pfx}.norm_in_t.bias" in state_dict:
            ct.norm_in_t.bias = mx.array(state_dict[f"{pfx}.norm_in_t.bias"].numpy())

        for idx in range(ct.num_layers):
            l_pfx = f"{pfx}.layers.{idx}"
            t_pfx = f"{pfx}.layers_t.{idx}"

            if idx % 2 == 0:
                for m_layer, lp in [(ct.layers[idx], l_pfx), (ct.layers_t[idx], t_pfx)]:
                    if f"{lp}.norm1.weight" in state_dict:
                        m_layer.norm1.weight = mx.array(state_dict[f"{lp}.norm1.weight"].numpy())
                    if f"{lp}.norm1.bias" in state_dict:
                        m_layer.norm1.bias = mx.array(state_dict[f"{lp}.norm1.bias"].numpy())
                    if f"{lp}.norm2.weight" in state_dict:
                        m_layer.norm2.weight = mx.array(state_dict[f"{lp}.norm2.weight"].numpy())
                    if f"{lp}.norm2.bias" in state_dict:
                        m_layer.norm2.bias = mx.array(state_dict[f"{lp}.norm2.bias"].numpy())

                    if f"{lp}.self_attn.in_proj_weight" in state_dict:
                        m_layer.in_proj.weight = mx.array(state_dict[f"{lp}.self_attn.in_proj_weight"].numpy())
                    if f"{lp}.self_attn.in_proj_bias" in state_dict:
                        m_layer.in_proj.bias = mx.array(state_dict[f"{lp}.self_attn.in_proj_bias"].numpy())
                    if f"{lp}.self_attn.out_proj.weight" in state_dict:
                        m_layer.out_proj.weight = mx.array(state_dict[f"{lp}.self_attn.out_proj.weight"].numpy())
                    if f"{lp}.self_attn.out_proj.bias" in state_dict:
                        m_layer.out_proj.bias = mx.array(state_dict[f"{lp}.self_attn.out_proj.bias"].numpy())

                    if f"{lp}.linear1.weight" in state_dict:
                        m_layer.linear1.weight = mx.array(state_dict[f"{lp}.linear1.weight"].numpy())
                    if f"{lp}.linear1.bias" in state_dict:
                        m_layer.linear1.bias = mx.array(state_dict[f"{lp}.linear1.bias"].numpy())
                    if f"{lp}.linear2.weight" in state_dict:
                        m_layer.linear2.weight = mx.array(state_dict[f"{lp}.linear2.weight"].numpy())
                    if f"{lp}.linear2.bias" in state_dict:
                        m_layer.linear2.bias = mx.array(state_dict[f"{lp}.linear2.bias"].numpy())

                    if f"{lp}.gamma_1.scale" in state_dict and m_layer.gamma1 is not None:
                        m_layer.gamma1.scale = mx.array(state_dict[f"{lp}.gamma_1.scale"].numpy())
                    if f"{lp}.gamma_2.scale" in state_dict and m_layer.gamma2 is not None:
                        m_layer.gamma2.scale = mx.array(state_dict[f"{lp}.gamma_2.scale"].numpy())
                    if f"{lp}.norm_out.weight" in state_dict and m_layer.norm_out is not None:
                        m_layer.norm_out.weight = mx.array(state_dict[f"{lp}.norm_out.weight"].numpy())
                    if f"{lp}.norm_out.bias" in state_dict and m_layer.norm_out is not None:
                        m_layer.norm_out.bias = mx.array(state_dict[f"{lp}.norm_out.bias"].numpy())
            else:
                for m_layer, lp in [(ct.layers[idx], l_pfx), (ct.layers_t[idx], t_pfx)]:
                    if f"{lp}.norm1.weight" in state_dict:
                        m_layer.norm1.weight = mx.array(state_dict[f"{lp}.norm1.weight"].numpy())
                    if f"{lp}.norm1.bias" in state_dict:
                        m_layer.norm1.bias = mx.array(state_dict[f"{lp}.norm1.bias"].numpy())
                    if f"{lp}.norm2.weight" in state_dict:
                        m_layer.norm2.weight = mx.array(state_dict[f"{lp}.norm2.weight"].numpy())
                    if f"{lp}.norm2.bias" in state_dict:
                        m_layer.norm2.bias = mx.array(state_dict[f"{lp}.norm2.bias"].numpy())
                    if f"{lp}.norm3.weight" in state_dict:
                        m_layer.norm3.weight = mx.array(state_dict[f"{lp}.norm3.weight"].numpy())
                    if f"{lp}.norm3.bias" in state_dict:
                        m_layer.norm3.bias = mx.array(state_dict[f"{lp}.norm3.bias"].numpy())

                    if f"{lp}.cross_attn.in_proj_weight" in state_dict:
                        w_qkv = state_dict[f"{lp}.cross_attn.in_proj_weight"].numpy()
                        b_qkv = state_dict[f"{lp}.cross_attn.in_proj_bias"].numpy()
                        dim = m_layer.dim
                        m_layer.q_proj.weight = mx.array(w_qkv[:dim])
                        m_layer.q_proj.bias = mx.array(b_qkv[:dim])
                        m_layer.k_proj.weight = mx.array(w_qkv[dim:2*dim])
                        m_layer.k_proj.bias = mx.array(b_qkv[dim:2*dim])
                        m_layer.v_proj.weight = mx.array(w_qkv[2*dim:])
                        m_layer.v_proj.bias = mx.array(b_qkv[2*dim:])

                    if f"{lp}.cross_attn.out_proj.weight" in state_dict:
                        m_layer.out_proj.weight = mx.array(state_dict[f"{lp}.cross_attn.out_proj.weight"].numpy())
                    if f"{lp}.cross_attn.out_proj.bias" in state_dict:
                        m_layer.out_proj.bias = mx.array(state_dict[f"{lp}.cross_attn.out_proj.bias"].numpy())

                    if f"{lp}.linear1.weight" in state_dict:
                        m_layer.linear1.weight = mx.array(state_dict[f"{lp}.linear1.weight"].numpy())
                    if f"{lp}.linear1.bias" in state_dict:
                        m_layer.linear1.bias = mx.array(state_dict[f"{lp}.linear1.bias"].numpy())
                    if f"{lp}.linear2.weight" in state_dict:
                        m_layer.linear2.weight = mx.array(state_dict[f"{lp}.linear2.weight"].numpy())
                    if f"{lp}.linear2.bias" in state_dict:
                        m_layer.linear2.bias = mx.array(state_dict[f"{lp}.linear2.bias"].numpy())

                    if f"{lp}.gamma_1.scale" in state_dict and m_layer.gamma1 is not None:
                        m_layer.gamma1.scale = mx.array(state_dict[f"{lp}.gamma_1.scale"].numpy())
                    if f"{lp}.gamma_2.scale" in state_dict and m_layer.gamma2 is not None:
                        m_layer.gamma2.scale = mx.array(state_dict[f"{lp}.gamma_2.scale"].numpy())
                    if f"{lp}.norm_out.weight" in state_dict and m_layer.norm_out is not None:
                        m_layer.norm_out.weight = mx.array(state_dict[f"{lp}.norm_out.weight"].numpy())
                    if f"{lp}.norm_out.bias" in state_dict and m_layer.norm_out is not None:
                        m_layer.norm_out.bias = mx.array(state_dict[f"{lp}.norm_out.bias"].numpy())

    if model.bottom_channels:
        if "channel_upsampler.weight" in state_dict:
            model.channel_upsampler.weight = mx.array(state_dict["channel_upsampler.weight"].squeeze(-1).numpy())
        if "channel_upsampler.bias" in state_dict:
            model.channel_upsampler.bias = mx.array(state_dict["channel_upsampler.bias"].numpy())

        if "channel_downsampler.weight" in state_dict:
            model.channel_downsampler.weight = mx.array(state_dict["channel_downsampler.weight"].squeeze(-1).numpy())
        if "channel_downsampler.bias" in state_dict:
            model.channel_downsampler.bias = mx.array(state_dict["channel_downsampler.bias"].numpy())

        if "channel_upsampler_t.weight" in state_dict:
            model.channel_upsampler_t.weight = mx.array(state_dict["channel_upsampler_t.weight"].squeeze(-1).numpy())
        if "channel_upsampler_t.bias" in state_dict:
            model.channel_upsampler_t.bias = mx.array(state_dict["channel_upsampler_t.bias"].numpy())

        if "channel_downsampler_t.weight" in state_dict:
            model.channel_downsampler_t.weight = mx.array(state_dict["channel_downsampler_t.weight"].squeeze(-1).numpy())
        if "channel_downsampler_t.bias" in state_dict:
            model.channel_downsampler_t.bias = mx.array(state_dict["channel_downsampler_t.bias"].numpy())

    if model.freq_emb is not None and "freq_emb.embedding.weight" in state_dict:
        model.freq_emb.weight = mx.array(state_dict["freq_emb.embedding.weight"].numpy())

    return model
