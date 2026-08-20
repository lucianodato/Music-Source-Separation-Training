# coding: utf-8
"""
Apple MLX Accelerated Inference Engine for MSST.

Provides high-performance native Metal execution for RoFormer, HTDemucs, and MDX23C
architectures on Apple Silicon (M1/M2/M3/M4) Macs.
"""

import os
import sys
import time
import platform
from typing import Dict, Any, Tuple, Optional, Union, List
import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    _MLX_BASE_AVAILABLE = platform.system() == "Darwin" and platform.machine() == "arm64"
except ImportError:
    mx = None
    nn = None
    _MLX_BASE_AVAILABLE = False


def is_mlx_available() -> bool:
    """Returns True if Apple MLX framework is installed on Apple Silicon."""
    return _MLX_BASE_AVAILABLE and mx is not None


def can_run_on_mlx(
    model_type: Optional[str] = None,
    config_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Checks whether a given model architecture and checkpoint can run via native Apple MLX.

    Returns:
        (is_supported, reason_or_architecture_name)
    """
    if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
        return False, "System is not macOS on Apple Silicon (arm64)."

    if not is_mlx_available():
        return False, "Apple MLX framework is not installed."

    mt = (model_type or "").lower().strip()

    # Direct model type checks
    supported_keywords = ["bs_roformer", "scnet", "htdemucs", "demucs"]
    unsupported_keywords = ["mel_band", "melband", "bandit", "mamba", "swin", "segm", "torchseg", "conformer_model"]

    for unsup in unsupported_keywords:
        if unsup in mt:
            return False, f"Architecture '{model_type}' (recurrent/custom mel kernels) does not have an MLX port yet."

    for sup in supported_keywords:
        if sup in mt:
            return True, f"Architecture '{model_type}' is natively supported by MLX."

    # Inspect config YAML if available
    if config_path and os.path.exists(config_path):
        try:
            import yaml
            with open(config_path, "r") as f:
                raw_cfg = yaml.unsafe_load(f)
            if isinstance(raw_cfg, dict):
                training_cfg = raw_cfg.get("training", {})
                cfg_mtype = str(training_cfg.get("model_type", "")).lower()
                for unsup in unsupported_keywords:
                    if unsup in cfg_mtype:
                        return False, f"Config model_type '{cfg_mtype}' is not supported in MLX."
                for sup in supported_keywords:
                    if sup in cfg_mtype:
                        return True, f"Config model_type '{cfg_mtype}' is natively supported by MLX."
        except Exception:
            pass

    return False, f"Architecture '{model_type}' is not recognized as MLX-compatible."


def load_mlx_model(
    model_type: str,
    config_path: str,
    checkpoint_path: str,
    use_float16: bool = True,
) -> Tuple[Any, Dict[str, Any], str]:
    """
    Loads and converts model weights natively into an MLX neural network module.

    Args:
        use_float16: If True, cast model weights to float16 for faster inference
                     and lower memory usage on Apple Silicon. Defaults to True.

    Returns:
        (mlx_model, config_dict, resolved_model_type)
    """
    if not is_mlx_available():
        raise RuntimeError("Apple MLX framework is not available in this environment.")

    import yaml
    with open(config_path, "r") as f:
        config = yaml.unsafe_load(f)

    clean_mtype = str(model_type or "").lower().strip()
    if not clean_mtype and isinstance(config, dict):
        clean_mtype = str(config.get("training", {}).get("model_type", "")).lower()

    if "bs_roformer" in clean_mtype or (clean_mtype == "roformer"):
        from models.bs_roformer.bs_roformer_mlx import load_bs_roformer_mlx_from_ckpt
        model = load_bs_roformer_mlx_from_ckpt(config, checkpoint_path)
        resolved_type = "bs_roformer"

    elif "scnet" in clean_mtype:
        from models.scnet.scnet_mlx import load_scnet_mlx_from_ckpt
        model = load_scnet_mlx_from_ckpt(config, checkpoint_path)
        resolved_type = "scnet"

    elif "htdemucs" in clean_mtype or "demucs" in clean_mtype:
        from models.htdemucs_mlx import load_htdemucs_mlx_from_ckpt
        model = load_htdemucs_mlx_from_ckpt(config, checkpoint_path)
        resolved_type = "htdemucs"

    else:
        raise NotImplementedError(f"Native MLX model loader for '{model_type}' is not implemented yet.")

    # Performance optimization: float16 precision
    if use_float16 and resolved_type in ("bs_roformer", "scnet"):
        _apply_float16(model)
        mx.eval(model.parameters())

    return model, config, resolved_type


def _apply_float16(model: Any) -> None:
    """Recursively cast all model parameter arrays to float16 for faster inference."""
    params = model.parameters()
    float16_params = _cast_params_recursive(params)
    model.update(float16_params)


def _cast_params_recursive(params):
    """Recursively walks a nested param structure and casts float32 arrays to float16."""
    if isinstance(params, mx.array):
        if params.dtype == mx.float32:
            return params.astype(mx.float16)
        return params
    elif isinstance(params, dict):
        return {k: _cast_params_recursive(v) for k, v in params.items()}
    elif isinstance(params, list):
        return [_cast_params_recursive(v) for v in params]
    return params


def _get_windowing_array_np(window_size: int, fade_size: int) -> np.ndarray:
    """Generates linear crossfade windowing weights for overlap-add."""
    fadein = np.linspace(0, 1, fade_size, dtype=np.float32)
    fadeout = np.linspace(1, 0, fade_size, dtype=np.float32)
    window = np.ones(window_size, dtype=np.float32)
    window[:fade_size] = fadein
    window[-fade_size:] = fadeout
    return window


def demix_mlx(
    config: Dict[str, Any],
    model: Any,
    mix: np.ndarray,
    model_type: str = "bs_roformer",
    pbar: bool = False,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Performs audio source separation with Apple MLX hardware acceleration.

    Args:
        config: Configuration dictionary loaded from model YAML.
        model: MLX model module (e.g. BSRoformerMLX).
        mix: Audio array of shape (channels, samples), float32.
        model_type: Model type string.
        pbar: Whether to display a tqdm progress bar.
        chunk_size: Optional custom chunk size in samples.
        overlap: Overlap factor (e.g. 2, 4, 8).

    Returns:
        Dict mapping stem names (e.g. 'vocals', 'drums', 'bass', 'other') to separated waveforms (numpy arrays).
    """
    from tqdm.auto import tqdm

    audio_cfg = config.get("audio", {})
    training_cfg = config.get("training", {})
    inference_cfg = config.get("inference", {})
    model_cfg = config.get("model", {})

    target_instrument = training_cfg.get("target_instrument")
    instruments = [target_instrument] if target_instrument else training_cfg.get("instruments", ["vocals", "bass", "drums", "other"])

    # Determine chunk parameters
    if chunk_size is not None:
        eff_chunk_size = chunk_size
    elif "chunk_size" in inference_cfg:
        eff_chunk_size = int(inference_cfg["chunk_size"])
    elif "chunk_size" in audio_cfg:
        eff_chunk_size = int(audio_cfg["chunk_size"])
    else:
        # Calculate from STFT parameters for RoFormer
        stft_hop = model_cfg.get("stft_hop_length", audio_cfg.get("hop_length", 512))
        dim_t = inference_cfg.get("dim_t", 256)
        eff_chunk_size = int(stft_hop) * (int(dim_t) - 1)

    eff_overlap = overlap if overlap is not None else int(inference_cfg.get("num_overlap", 2))
    if eff_overlap <= 0:
        eff_overlap = 2

    step = eff_chunk_size // eff_overlap
    fade_size = eff_chunk_size // 10
    border = eff_chunk_size - step

    # Ensure mix is 2D float32
    if len(mix.shape) == 1:
        mix = np.stack([mix, mix], axis=0)
    mix = mix.astype(np.float32)

    length_init = mix.shape[-1]
    if length_init > 2 * border and border > 0:
        # Reflect pad edges
        padded_mix = np.pad(mix, ((0, 0), (border, border)), mode="reflect")
    else:
        padded_mix = mix
        border = 0

    total_len = padded_mix.shape[1]
    num_instruments = len(instruments)

    # Accumulator arrays on CPU/unified memory
    result = np.zeros((num_instruments, padded_mix.shape[0], total_len), dtype=np.float32)
    counter = np.zeros((num_instruments, padded_mix.shape[0], total_len), dtype=np.float32)

    windowing_array = _get_windowing_array_np(eff_chunk_size, fade_size)

    i = 0
    progress = tqdm(total=total_len, desc="Processing MLX chunks", leave=False) if pbar else None

    while i < total_len:
        part = padded_mix[:, i:i + eff_chunk_size]
        chunk_len = part.shape[-1]

        if chunk_len < eff_chunk_size:
            pad_w = eff_chunk_size - chunk_len
            if chunk_len > eff_chunk_size // 2:
                part = np.pad(part, ((0, 0), (0, pad_w)), mode="reflect")
            else:
                part = np.pad(part, ((0, 0), (0, pad_w)), mode="constant")

        # Convert chunk to MLX array of shape (1, channels, eff_chunk_size)
        mlx_chunk = mx.array(part[np.newaxis, ...])
        
        # Forward pass on MLX GPU
        out_mlx = model(mlx_chunk)
        mx.eval(out_mlx)

        # Output shape is typically (1, num_stems, channels, eff_chunk_size) or (1, channels, eff_chunk_size)
        out_np = np.array(out_mlx.astype(mx.float32))
        if out_np.ndim == 4:
            out_chunk = out_np[0]  # (num_stems, channels, eff_chunk_size)
        elif out_np.ndim == 3 and num_instruments == 1:
            out_chunk = out_np  # (1, channels, eff_chunk_size)
        else:
            out_chunk = out_np

        window = windowing_array.copy()
        if i == 0:
            window[:fade_size] = 1.0
        if i + eff_chunk_size >= total_len:
            window[-fade_size:] = 1.0

        # Broadcast window to (num_stems, channels, seg_len)
        w_slice = window[:chunk_len][np.newaxis, np.newaxis, :]
        result[..., i:i + chunk_len] += out_chunk[..., :chunk_len] * w_slice
        counter[..., i:i + chunk_len] += w_slice

        i += step
        if progress:
            progress.update(step)

    if progress:
        progress.close()

    # Normalize by overlap weights
    counter = np.maximum(counter, 1e-7)
    estimated_sources = result / counter

    # Trim border reflection padding
    if border > 0:
        estimated_sources = estimated_sources[..., border:border + length_init]

    # Return stems dict
    ret_dict = {}
    for idx, inst in enumerate(instruments):
        ret_dict[inst] = estimated_sources[idx]

    return ret_dict


def bigshifts_wrapper_mlx(
    config: Dict[str, Any],
    model: Any,
    mix: np.ndarray,
    model_type: str = "bs_roformer",
    pbar: bool = False,
    bigshifts: int = 1,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """BigShifts wrapper for MLX inference with multi-shift averaging."""
    if bigshifts <= 1:
        return demix_mlx(
            config=config,
            model=model,
            mix=mix,
            model_type=model_type,
            pbar=pbar,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    from tqdm.auto import tqdm
    shift_in_samples = mix.shape[1] // bigshifts
    shifts = [x * shift_in_samples for x in range(bigshifts)]
    results = []

    shifts_iter = tqdm(shifts, desc="BigShifts MLX passes...", leave=False) if pbar else shifts
    for shift in shifts_iter:
        shifted_mix = np.concatenate((mix[:, -shift:], mix[:, :-shift]), axis=-1)
        sources = demix_mlx(
            config=config,
            model=model,
            mix=shifted_mix,
            model_type=model_type,
            pbar=False,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        unshifted = {
            k: np.concatenate((v[..., shift:], v[..., :shift]), axis=-1)
            for k, v in sources.items()
        }
        results.append(unshifted)

    avg_result = {}
    for k in results[0]:
        avg_result[k] = np.mean([r[k] for r in results], axis=0)
    return avg_result
