# coding: utf-8
"""
Clean-sheet native Apple MLX implementation of SCNet (Sparse Compression Network for Music Source Separation).
Zero third-party wrapper dependencies; utilizes pure mlx.core and mlx.nn on Apple Silicon Metal.
"""

import math
import numpy as np
from collections import deque
from typing import Tuple, List, Dict, Optional, Any
import mlx.core as mx
import mlx.nn as nn


def mlx_stft_scnet(
    x: mx.array,
    n_fft: int = 4096,
    hop_length: int = 1024,
    win_length: int = 4096,
    normalized: bool = True,
) -> mx.array:
    """
    Computes real STFT matching torch.stft(..., return_complex=True) with reflect padding and rectangular window.
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

    num_frames = 1 + (x_padded.shape[1] - win_length) // hop_length
    indices = mx.arange(win_length)[None, :] + mx.arange(num_frames)[:, None] * hop_length
    frames = x_padded[:, indices]

    spec = mx.fft.rfft(frames, n=n_fft, axis=-1)
    if normalized:
        spec = spec / math.sqrt(n_fft)

    out = mx.stack([spec.real, spec.imag], axis=-1)
    return mx.transpose(out, (0, 2, 1, 3))


def mlx_istft_scnet(
    spec_real_imag: mx.array,
    n_fft: int = 4096,
    hop_length: int = 1024,
    win_length: int = 4096,
    normalized: bool = True,
    length: Optional[int] = None,
) -> np.ndarray:
    """
    Computes inverse real STFT matching torch.istft with rectangular window overlap-add synthesis.
    Args:
        spec_real_imag: (B, F, num_frames, 2)
    Returns:
        (B, length) numpy array
    """
    B, F, num_frames, _ = spec_real_imag.shape
    spec_c = spec_real_imag[..., 0] + 1j * spec_real_imag[..., 1]
    if normalized:
        spec_c = spec_c * math.sqrt(n_fft)
    spec_c = mx.transpose(spec_c, (0, 2, 1))

    frames = mx.fft.irfft(spec_c, n=n_fft, axis=-1)[..., :win_length]

    total_samples = (num_frames - 1) * hop_length + win_length
    out_signal = np.zeros((B, total_samples), dtype=np.float32)
    window_sum = np.zeros((total_samples,), dtype=np.float32)

    wf_np = np.array(frames)
    w_sq = np.ones((win_length,), dtype=np.float32)

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


class ConvolutionModuleMLX(nn.Module):
    """
    Convolution Module in SD block.
    """

    def __init__(self, channels: int, depth: int = 2, compress: float = 4, kernel: int = 3):
        super().__init__()
        self.depth = depth
        hidden_size = int(channels / compress)
        self.layers = []
        for _ in range(depth):
            mods = {
                "norm1": nn.GroupNorm(1, channels, pytorch_compatible=True),
                "conv1": nn.Conv1d(channels, hidden_size * 2, kernel, padding=kernel // 2),
                "conv2": nn.Conv1d(hidden_size, hidden_size, kernel, padding=kernel // 2, groups=hidden_size),
                "norm2": nn.GroupNorm(1, hidden_size, pytorch_compatible=True),
                "conv3": nn.Conv1d(hidden_size, channels, 1),
            }
            self.layers.append(mods)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, L, C)
        for layer in self.layers:
            h = layer["norm1"](x)
            h = layer["conv1"](h)
            a, b = mx.split(h, 2, axis=-1)
            h = a * mx.sigmoid(b)  # GLU
            h = layer["conv2"](h)
            h = layer["norm2"](h)
            h = h * mx.sigmoid(h)  # Swish / SiLU
            h = layer["conv3"](h)
            x = x + h
        return x


class FusionLayerMLX(nn.Module):
    """
    A FusionLayer within the decoder.
    """

    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels * 2, kernel_size, stride=stride, padding=padding)

    def __call__(self, x: mx.array, skip: Optional[mx.array] = None) -> mx.array:
        # x: (B, Fr, T, C)
        if skip is not None:
            x = x + skip
        x = mx.concatenate([x, x], axis=-1)
        x = self.conv(x)
        a, b = mx.split(x, 2, axis=-1)
        return a * mx.sigmoid(b)


class SDlayerMLX(nn.Module):
    """
    Sparse Down-sample Layer for processing frequency bands separately.
    """

    def __init__(self, channels_in: int, channels_out: int, band_configs: Dict[str, Any]):
        super().__init__()
        self.convs = []
        self.strides = []
        self.kernels = []
        for config in band_configs.values():
            conv = nn.Conv2d(channels_in, channels_out, (config["kernel"], 1), stride=(config["stride"], 1))
            self.convs.append(conv)
            self.strides.append(config["stride"])
            self.kernels.append(config["kernel"])

        self.SR_low = band_configs["low"]["SR"]
        self.SR_mid = band_configs["mid"]["SR"]

    def __call__(self, x: mx.array) -> Tuple[List[mx.array], List[int]]:
        # x: (B, Fr, T, C)
        B, Fr, T, C = x.shape
        splits = [
            (0, math.ceil(Fr * self.SR_low)),
            (math.ceil(Fr * self.SR_low), math.ceil(Fr * (self.SR_low + self.SR_mid))),
            (math.ceil(Fr * (self.SR_low + self.SR_mid)), Fr),
        ]

        outputs = []
        original_lengths = []
        for conv, stride, kernel, (start, end) in zip(self.convs, self.strides, self.kernels, splits):
            extracted = x[:, start:end, :, :]
            original_lengths.append(end - start)
            current_length = extracted.shape[1]

            if stride == 1:
                total_padding = kernel - stride
            else:
                total_padding = (stride - current_length % stride) % stride
            pad_left = total_padding // 2
            pad_right = total_padding - pad_left

            padded = mx.pad(extracted, [(0, 0), (pad_left, pad_right), (0, 0), (0, 0)])
            output = conv(padded)
            outputs.append(output)

        return outputs, original_lengths


class SUlayerMLX(nn.Module):
    """
    Sparse Up-sample Layer in decoder.
    """

    def __init__(self, channels_in: int, channels_out: int, band_configs: Dict[str, Any]):
        super().__init__()
        self.convtrs = []
        for config in band_configs.values():
            convtr = nn.ConvTranspose2d(channels_in, channels_out, (config["kernel"], 1), stride=(config["stride"], 1))
            self.convtrs.append(convtr)

    def __call__(self, x: mx.array, lengths: List[int], origin_lengths: List[int]) -> mx.array:
        # x: (B, Fr, T, C)
        splits = [
            (0, lengths[0]),
            (lengths[0], lengths[0] + lengths[1]),
            (lengths[0] + lengths[1], None),
        ]
        outputs = []
        for idx, (convtr, (start, end)) in enumerate(zip(self.convtrs, splits)):
            sub_x = x[:, start:end, :, :] if end is not None else x[:, start:, :, :]
            out = convtr(sub_x)
            current_Fr_length = out.shape[1]
            dist = abs(origin_lengths[idx] - current_Fr_length) // 2
            trimmed = out[:, dist:dist + origin_lengths[idx], :, :]
            outputs.append(trimmed)

        return mx.concatenate(outputs, axis=1)


class SDblockMLX(nn.Module):
    """
    Sparse Down-sample block in encoder.
    """

    def __init__(
        self,
        channels_in: int,
        channels_out: int,
        band_configs: Dict[str, Any] = {},
        conv_config: Dict[str, Any] = {},
        depths: List[int] = [3, 2, 1],
        kernel_size: int = 3,
    ):
        super().__init__()
        self.SDlayer = SDlayerMLX(channels_in, channels_out, band_configs)
        self.conv_modules = [
            ConvolutionModuleMLX(channels_out, depth, **conv_config) for depth in depths
        ]
        self.globalconv = nn.Conv2d(channels_out, channels_out, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def __call__(self, x: mx.array) -> Tuple[mx.array, mx.array, List[int], List[int]]:
        # x: (B, Fr, T, C)
        bands, original_lengths = self.SDlayer(x)

        processed_bands = []
        for conv_mod, band in zip(self.conv_modules, bands):
            B, Fr_b, T, C = band.shape
            h = band.reshape(B * Fr_b, T, C)
            h = conv_mod(h)
            h = nn.gelu(h)
            processed_bands.append(h.reshape(B, Fr_b, T, C))

        lengths = [band.shape[1] for band in processed_bands]
        full_band = mx.concatenate(processed_bands, axis=1)
        skip = full_band
        output = self.globalconv(full_band)
        return output, skip, lengths, original_lengths


class BiLSTM_MLX(nn.Module):
    """Bidirectional LSTM in MLX."""

    def __init__(self, in_features: int, hidden_size: int):
        super().__init__()
        self.fwd = nn.LSTM(in_features, hidden_size)
        self.bwd = nn.LSTM(in_features, hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (N, L, D)
        out_fwd, _ = self.fwd(x)
        x_rev = x[:, ::-1, :]
        out_bwd, _ = self.bwd(x_rev)
        out_bwd = out_bwd[:, ::-1, :]
        return mx.concatenate([out_fwd, out_bwd], axis=-1)


class DualPathRNN_MLX(nn.Module):
    """
    Dual-Path RNN in Separation Network.
    """

    def __init__(self, d_model: int, expand: int = 1):
        super().__init__()
        self.d_model = d_model
        self.hidden_size = d_model * expand

        self.lstm_freq = BiLSTM_MLX(d_model, self.hidden_size)
        self.linear_freq = nn.Linear(self.hidden_size * 2, d_model)
        self.norm_freq = nn.GroupNorm(1, d_model, pytorch_compatible=True)

        self.lstm_time = BiLSTM_MLX(d_model, self.hidden_size)
        self.linear_time = nn.Linear(self.hidden_size * 2, d_model)
        self.norm_time = nn.GroupNorm(1, d_model, pytorch_compatible=True)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, Fr, T, C)
        B, Fr, T, C = x.shape
        orig_x = x

        # 1. Frequency-path
        h = self.norm_freq(x)
        h = h.transpose(0, 2, 1, 3).reshape(B * T, Fr, C)
        h = self.lstm_freq(h)
        h = self.linear_freq(h)
        h = h.reshape(B, T, Fr, C).transpose(0, 2, 1, 3)
        x = orig_x + h

        # 2. Time-path
        orig_x = x
        h = self.norm_time(x)
        h = h.reshape(B * Fr, T, C)
        h = self.lstm_time(h)
        h = self.linear_time(h)
        h = h.reshape(B, Fr, T, C)
        x = orig_x + h

        return x


class FeatureConversionMLX(nn.Module):
    """
    Orthogonal FFT Feature Conversion between Dual-Path layers.
    """

    def __init__(self, channels: int, inverse: bool):
        super().__init__()
        self.channels = channels
        self.inverse = inverse

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, Fr, T, C)
        if self.inverse:
            C_half = self.channels // 2
            x_r = x[..., :C_half]
            x_i = x[..., C_half:]
            comp = x_r + 1j * x_i
            T_half = comp.shape[2]
            n_orig = (T_half - 1) * 2
            out = mx.fft.irfft(comp, n=n_orig, axis=2)
            return out * (n_orig ** 0.5)
        else:
            n_orig = x.shape[2]
            spec = mx.fft.rfft(x, axis=2)
            scale = 1.0 / (n_orig ** 0.5)
            spec_r = spec.real * scale
            spec_i = spec.imag * scale
            return mx.concatenate([spec_r, spec_i], axis=-1)


class SeparationNetMLX(nn.Module):
    """
    Separation Network composed of alternating DualPathRNN and FeatureConversion.
    """

    def __init__(self, channels: int, expand: int = 1, num_layers: int = 6):
        super().__init__()
        self.num_layers = num_layers
        self.dp_modules = [
            DualPathRNN_MLX(channels * (2 if i % 2 == 1 else 1), expand) for i in range(num_layers)
        ]
        self.feature_conversions = [
            FeatureConversionMLX(channels * 2, inverse=False if i % 2 == 0 else True) for i in range(num_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for i in range(self.num_layers):
            x = self.dp_modules[i](x)
            x = self.feature_conversions[i](x)
        return x


class SCNetMLX(nn.Module):
    """
    Native Apple MLX SCNet model for audio stem separation.
    """

    def __init__(
        self,
        sources: List[str] = ["drums", "bass", "other", "vocals"],
        audio_channels: int = 2,
        dims: List[int] = [4, 64, 128, 256],
        nfft: int = 4096,
        hop_size: int = 1024,
        win_size: int = 4096,
        normalized: bool = True,
        band_SR: List[float] = [0.175, 0.392, 0.433],
        band_stride: List[int] = [1, 4, 16],
        band_kernel: List[int] = [3, 4, 16],
        conv_depths: List[int] = [3, 2, 1],
        compress: int = 4,
        conv_kernel: int = 3,
        num_dplayer: int = 6,
        expand: int = 1,
    ):
        super().__init__()
        self.sources = sources
        self.audio_channels = audio_channels
        self.dims = dims
        self.hop_length = hop_size
        self.n_fft = nfft
        self.win_length = win_size
        self.normalized = normalized

        band_keys = ["low", "mid", "high"]
        self.band_configs = {
            band_keys[i]: {"SR": band_SR[i], "stride": band_stride[i], "kernel": band_kernel[i]}
            for i in range(len(band_keys))
        }

        self.conv_config = {"compress": compress, "kernel": conv_kernel}

        self.encoder = []
        self.decoder = []

        for index in range(len(dims) - 1):
            enc = SDblockMLX(
                channels_in=dims[index],
                channels_out=dims[index + 1],
                band_configs=self.band_configs,
                conv_config=self.conv_config,
                depths=conv_depths,
            )
            self.encoder.append(enc)

            dec_fusion = FusionLayerMLX(channels=dims[index + 1])
            dec_su = SUlayerMLX(
                channels_in=dims[index + 1],
                channels_out=dims[index] if index != 0 else dims[index] * len(sources),
                band_configs=self.band_configs,
            )
            self.decoder.insert(0, (dec_fusion, dec_su))

        self.separation_net = SeparationNetMLX(
            channels=dims[-1],
            expand=expand,
            num_layers=num_dplayer,
        )

    def __call__(self, x: mx.array) -> np.ndarray:
        """
        Runs SCNet forward separation on raw audio input.
        Args:
            x: (B, channels, samples) or (channels, samples)
        Returns:
            (B, num_sources, channels, samples) or (num_sources, channels, samples)
        """
        squeeze_batch = False
        if x.ndim == 2:
            x = x[None, ...]
            squeeze_batch = True

        B, S, T_orig = x.shape
        padding = self.hop_length - T_orig % self.hop_length
        if (T_orig + padding) // self.hop_length % 2 == 0:
            padding += self.hop_length
        x = mx.pad(x, [(0, 0), (0, 0), (0, padding)])

        L = x.shape[-1]
        x_flat = x.reshape(B * S, L)

        stft_repr = mlx_stft_scnet(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            normalized=self.normalized,
        )  # (B*S, Fr, num_frames, 2)

        Fr = stft_repr.shape[1]
        num_frames = stft_repr.shape[2]

        stft_repr = stft_repr.reshape(B, S, Fr, num_frames, 2)
        # Transpose to (B, Fr, num_frames, S * 2) -> (B, Fr, T, C)
        x_in = stft_repr.transpose(0, 2, 3, 1, 4).reshape(B, Fr, num_frames, S * 2)

        save_skip = deque()
        save_lengths = deque()
        save_original_lengths = deque()

        # 1. Encoder
        h = x_in
        for sd_layer in self.encoder:
            h, skip, lengths, original_lengths = sd_layer(h)
            save_skip.append(skip)
            save_lengths.append(lengths)
            save_original_lengths.append(original_lengths)

        # 2. Separation Network (DualPath RNN + Feature Conversion)
        h = self.separation_net(h)

        # 3. Decoder
        for fusion_layer, su_layer in self.decoder:
            h = fusion_layer(h, save_skip.pop())
            h = su_layer(h, save_lengths.pop(), save_original_lengths.pop())

        # 4. Output complex reconstruction
        num_sources = len(self.sources)
        h = h.reshape(B, Fr, num_frames, num_sources, S, 2)
        h = h.transpose(0, 3, 4, 1, 2, 5)

        h_flat = h.reshape(B * num_sources * S, Fr, num_frames, 2)
        recon_flat = mlx_istft_scnet(
            h_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            normalized=self.normalized,
            length=L,
        )  # (B * num_sources * S, L)

        recon = recon_flat.reshape(B, num_sources, S, L)
        recon = recon[:, :, :, :-padding] if padding > 0 else recon

        if squeeze_batch:
            return recon[0]
        return recon


def load_scnet_mlx_from_ckpt(config: Dict[str, Any], ckpt_path: str) -> SCNetMLX:
    """
    Constructs SCNetMLX and maps PyTorch state_dict parameters directly into MLX.
    """
    import torch

    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})
    sources = training_cfg.get("instruments", ["drums", "bass", "other", "vocals"])
    if training_cfg.get("target_instrument"):
        sources = [training_cfg.get("target_instrument")]

    dims = model_cfg.get("dims", [4, 64, 128, 256])
    nfft = model_cfg.get("nfft", 4096)
    hop_size = model_cfg.get("hop_size", 1024)
    win_size = model_cfg.get("win_size", 4096)
    normalized = model_cfg.get("normalized", True)
    band_SR = model_cfg.get("band_SR", [0.175, 0.392, 0.433])
    band_stride = model_cfg.get("band_stride", [1, 4, 16])
    band_kernel = model_cfg.get("band_kernel", [3, 4, 16])
    conv_depths = model_cfg.get("conv_depths", [3, 2, 1])
    compress = model_cfg.get("compress", 4)
    conv_kernel = model_cfg.get("conv_kernel", 3)
    num_dplayer = model_cfg.get("num_dplayer", 6)
    expand = model_cfg.get("expand", 1)

    model = SCNetMLX(
        sources=sources,
        audio_channels=2,
        dims=dims,
        nfft=nfft,
        hop_size=hop_size,
        win_size=win_size,
        normalized=normalized,
        band_SR=band_SR,
        band_stride=band_stride,
        band_kernel=band_kernel,
        conv_depths=conv_depths,
        compress=compress,
        conv_kernel=conv_kernel,
        num_dplayer=num_dplayer,
        expand=expand,
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

    # Map weights
    # 1. Encoder SDblocks
    for enc_idx, enc_block in enumerate(model.encoder):
        # SDlayer convs
        for c_idx in range(3):
            w_k = f"encoder.{enc_idx}.SDlayer.convs.{c_idx}.weight"
            b_k = f"encoder.{enc_idx}.SDlayer.convs.{c_idx}.bias"
            if w_k in state_dict:
                enc_block.SDlayer.convs[c_idx].weight = mx.array(state_dict[w_k].numpy().transpose(0, 2, 3, 1))
            if b_k in state_dict and state_dict[b_k] is not None:
                enc_block.SDlayer.convs[c_idx].bias = mx.array(state_dict[b_k].numpy())

        # ConvolutionModules
        for mod_idx, conv_mod in enumerate(enc_block.conv_modules):
            for layer_idx, layer_dict in enumerate(conv_mod.layers):
                pfx = f"encoder.{enc_idx}.conv_modules.{mod_idx}.layers.{layer_idx}"
                if f"{pfx}.0.weight" in state_dict:
                    layer_dict["norm1"].weight = mx.array(state_dict[f"{pfx}.0.weight"].numpy())
                if f"{pfx}.0.bias" in state_dict:
                    layer_dict["norm1"].bias = mx.array(state_dict[f"{pfx}.0.bias"].numpy())

                if f"{pfx}.1.weight" in state_dict:
                    layer_dict["conv1"].weight = mx.array(state_dict[f"{pfx}.1.weight"].numpy().transpose(0, 2, 1))
                if f"{pfx}.1.bias" in state_dict and state_dict[f"{pfx}.1.bias"] is not None:
                    layer_dict["conv1"].bias = mx.array(state_dict[f"{pfx}.1.bias"].numpy())

                if f"{pfx}.3.weight" in state_dict:
                    layer_dict["conv2"].weight = mx.array(state_dict[f"{pfx}.3.weight"].numpy().transpose(0, 2, 1))
                if f"{pfx}.3.bias" in state_dict and state_dict[f"{pfx}.3.bias"] is not None:
                    layer_dict["conv2"].bias = mx.array(state_dict[f"{pfx}.3.bias"].numpy())

                if f"{pfx}.4.weight" in state_dict:
                    layer_dict["norm2"].weight = mx.array(state_dict[f"{pfx}.4.weight"].numpy())
                if f"{pfx}.4.bias" in state_dict:
                    layer_dict["norm2"].bias = mx.array(state_dict[f"{pfx}.4.bias"].numpy())

                if f"{pfx}.6.weight" in state_dict:
                    layer_dict["conv3"].weight = mx.array(state_dict[f"{pfx}.6.weight"].numpy().transpose(0, 2, 1))
                if f"{pfx}.6.bias" in state_dict and state_dict[f"{pfx}.6.bias"] is not None:
                    layer_dict["conv3"].bias = mx.array(state_dict[f"{pfx}.6.bias"].numpy())

        # globalconv
        gw_k = f"encoder.{enc_idx}.globalconv.weight"
        gb_k = f"encoder.{enc_idx}.globalconv.bias"
        if gw_k in state_dict:
            enc_block.globalconv.weight = mx.array(state_dict[gw_k].numpy().transpose(0, 2, 3, 1))
        if gb_k in state_dict and state_dict[gb_k] is not None:
            enc_block.globalconv.bias = mx.array(state_dict[gb_k].numpy())

    # 2. Separation Network
    for dp_idx, dp_mod in enumerate(model.separation_net.dp_modules):
        pfx = f"separation_net.dp_modules.{dp_idx}"

        # LSTM 0 (Frequency)
        if f"{pfx}.lstm_layers.0.weight_ih_l0" in state_dict:
            dp_mod.lstm_freq.fwd.Wx = mx.array(state_dict[f"{pfx}.lstm_layers.0.weight_ih_l0"].numpy())
            dp_mod.lstm_freq.fwd.Wh = mx.array(state_dict[f"{pfx}.lstm_layers.0.weight_hh_l0"].numpy())
            dp_mod.lstm_freq.fwd.bias = mx.array((state_dict[f"{pfx}.lstm_layers.0.bias_ih_l0"] + state_dict[f"{pfx}.lstm_layers.0.bias_hh_l0"]).numpy())

            dp_mod.lstm_freq.bwd.Wx = mx.array(state_dict[f"{pfx}.lstm_layers.0.weight_ih_l0_reverse"].numpy())
            dp_mod.lstm_freq.bwd.Wh = mx.array(state_dict[f"{pfx}.lstm_layers.0.weight_hh_l0_reverse"].numpy())
            dp_mod.lstm_freq.bwd.bias = mx.array((state_dict[f"{pfx}.lstm_layers.0.bias_ih_l0_reverse"] + state_dict[f"{pfx}.lstm_layers.0.bias_hh_l0_reverse"]).numpy())

        # Linear 0
        if f"{pfx}.linear_layers.0.weight" in state_dict:
            dp_mod.linear_freq.weight = mx.array(state_dict[f"{pfx}.linear_layers.0.weight"].numpy())
        if f"{pfx}.linear_layers.0.bias" in state_dict and state_dict[f"{pfx}.linear_layers.0.bias"] is not None:
            dp_mod.linear_freq.bias = mx.array(state_dict[f"{pfx}.linear_layers.0.bias"].numpy())

        # Norm 0
        if f"{pfx}.norm_layers.0.weight" in state_dict:
            dp_mod.norm_freq.weight = mx.array(state_dict[f"{pfx}.norm_layers.0.weight"].numpy())
        if f"{pfx}.norm_layers.0.bias" in state_dict:
            dp_mod.norm_freq.bias = mx.array(state_dict[f"{pfx}.norm_layers.0.bias"].numpy())

        # LSTM 1 (Time)
        if f"{pfx}.lstm_layers.1.weight_ih_l0" in state_dict:
            dp_mod.lstm_time.fwd.Wx = mx.array(state_dict[f"{pfx}.lstm_layers.1.weight_ih_l0"].numpy())
            dp_mod.lstm_time.fwd.Wh = mx.array(state_dict[f"{pfx}.lstm_layers.1.weight_hh_l0"].numpy())
            dp_mod.lstm_time.fwd.bias = mx.array((state_dict[f"{pfx}.lstm_layers.1.bias_ih_l0"] + state_dict[f"{pfx}.lstm_layers.1.bias_hh_l0"]).numpy())

            dp_mod.lstm_time.bwd.Wx = mx.array(state_dict[f"{pfx}.lstm_layers.1.weight_ih_l0_reverse"].numpy())
            dp_mod.lstm_time.bwd.Wh = mx.array(state_dict[f"{pfx}.lstm_layers.1.weight_hh_l0_reverse"].numpy())
            dp_mod.lstm_time.bwd.bias = mx.array((state_dict[f"{pfx}.lstm_layers.1.bias_ih_l0_reverse"] + state_dict[f"{pfx}.lstm_layers.1.bias_hh_l0_reverse"]).numpy())

        # Linear 1
        if f"{pfx}.linear_layers.1.weight" in state_dict:
            dp_mod.linear_time.weight = mx.array(state_dict[f"{pfx}.linear_layers.1.weight"].numpy())
        if f"{pfx}.linear_layers.1.bias" in state_dict and state_dict[f"{pfx}.linear_layers.1.bias"] is not None:
            dp_mod.linear_time.bias = mx.array(state_dict[f"{pfx}.linear_layers.1.bias"].numpy())

        # Norm 1
        if f"{pfx}.norm_layers.1.weight" in state_dict:
            dp_mod.norm_time.weight = mx.array(state_dict[f"{pfx}.norm_layers.1.weight"].numpy())
        if f"{pfx}.norm_layers.1.bias" in state_dict:
            dp_mod.norm_time.bias = mx.array(state_dict[f"{pfx}.norm_layers.1.bias"].numpy())

    # 3. Decoder (FusionLayer + SUlayer)
    for dec_idx, (fusion_layer, su_layer) in enumerate(model.decoder):
        f_w_k = f"decoder.{dec_idx}.0.conv.weight"
        f_b_k = f"decoder.{dec_idx}.0.conv.bias"
        if f_w_k in state_dict:
            fusion_layer.conv.weight = mx.array(state_dict[f_w_k].numpy().transpose(0, 2, 3, 1))
        if f_b_k in state_dict and state_dict[f_b_k] is not None:
            fusion_layer.conv.bias = mx.array(state_dict[f_b_k].numpy())

        for c_idx in range(3):
            su_w_k = f"decoder.{dec_idx}.1.convtrs.{c_idx}.weight"
            su_b_k = f"decoder.{dec_idx}.1.convtrs.{c_idx}.bias"
            if su_w_k in state_dict:
                su_layer.convtrs[c_idx].weight = mx.array(state_dict[su_w_k].numpy().transpose(1, 2, 3, 0))
            if su_b_k in state_dict and state_dict[su_b_k] is not None:
                su_layer.convtrs[c_idx].bias = mx.array(state_dict[su_b_k].numpy())

    return model
