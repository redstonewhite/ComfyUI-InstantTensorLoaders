from __future__ import annotations

import contextlib
import copy
import json
import logging
import threading
from dataclasses import dataclass
from typing import Any, Iterable

import torch

import comfy.clip_vision
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.sd
import comfy.utils
import folder_paths
import nodes
import yaml


_SAFETENSOR_EXTS = (".safetensors", ".sft")
_PATCH_LOCK = threading.RLock()
_META_DEVICE = torch.device("meta")
_SAFETENSORS_DTYPES = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "C64": torch.complex64,
}
for _safetensors_name, _torch_names in {
    "F8_E4M3": ("float8_e4m3fn",),
    "F8_E5M2": ("float8_e5m2",),
    "U64": ("uint64",),
    "U32": ("uint32",),
    "U16": ("uint16",),
    "F8_E8M0": ("float8_e8m0fnu", "float8_e8m0fnuz"),
    "F4": ("float4_e2m1fn_x2",),
}.items():
    for _torch_name in _torch_names:
        if hasattr(torch, _torch_name):
            _SAFETENSORS_DTYPES[_safetensors_name] = getattr(torch, _torch_name)
            break


def _is_safetensors(path: str) -> bool:
    return path.lower().endswith(_SAFETENSOR_EXTS)


def _cuda_device() -> torch.device:
    device = comfy.model_management.get_torch_device()
    if device.type != "cuda":
        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            return torch.device("cuda", index)
        raise RuntimeError(
            "InstantTensor currently only supports CUDA devices. "
            "Start ComfyUI with a CUDA torch device to use Instant Loaders."
        )
    if device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _without_cpu_device(input_types: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(input_types)
    optional = out.get("optional", {})
    optional.pop("device", None)
    if not optional and "optional" in out:
        out.pop("optional")
    return out


def _instant_load_safetensors(
    path: str,
    device: torch.device,
    return_metadata: bool = False,
) -> Any:
    try:
        from instanttensor import safe_open
    except Exception as exc:
        raise RuntimeError(
            "The instanttensor package is required for CUDA instant loading. "
            "Install it in the ComfyUI environment used to run ComfyUI."
        ) from exc

    state_dict: dict[str, torch.Tensor] = {}
    metadata = None
    with safe_open(path, framework="pt", device=device) as handle:
        if return_metadata:
            try:
                metadata = handle.metadata()
            except TypeError:
                metadata = None
        for key, tensor in handle.tensors():
            # InstantTensor reuses its internal transfer buffer while iterating.
            # The clone is the durable tensor that can be assigned directly into
            # the target module without a second model-sized copy.
            state_dict[key] = tensor.clone().detach()
    return (state_dict, metadata) if return_metadata else state_dict


def _read_safetensors_metadata(path: str) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    header = comfy.utils.safetensors_header(path)
    if header is None:
        raise ValueError(f"Safetensors header is too large or invalid: {path}")
    data = json.loads(header.decode("utf-8"))
    metadata = data.pop("__metadata__", {}) or {}
    return metadata, data


def _meta_state_dict(tensor_metadata: dict[str, dict[str, Any]]) -> dict[str, torch.Tensor]:
    state_dict = {}
    for name, info in tensor_metadata.items():
        dtype_name = info["dtype"]
        dtype = _SAFETENSORS_DTYPES.get(dtype_name)
        if dtype is None:
            raise ValueError(f"Unsupported safetensors dtype {dtype_name!r} for {name}")
        state_dict[name] = torch.empty(tuple(info["shape"]), dtype=dtype, device=_META_DEVICE)
    return state_dict


def _metadata_numel(info: dict[str, Any]) -> int:
    numel = 1
    for dim in info["shape"]:
        numel *= int(dim)
    return numel


def _materialize_quant_metadata_tensors(
    path: str,
    state_dict: dict[str, torch.Tensor],
    tensor_metadata: dict[str, dict[str, Any]],
) -> None:
    # ComfyUI quant loaders parse comfy_quant and call .item() on old scale_input.
    tensor_names = [
        name
        for name, info in tensor_metadata.items()
        if name.endswith(".comfy_quant")
        or (name.endswith(".scale_input") and _metadata_numel(info) == 1)
    ]
    if not tensor_names:
        return

    try:
        from safetensors import safe_open
    except Exception as exc:
        raise RuntimeError(
            "The safetensors package is required to read quantization metadata."
        ) from exc

    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in tensor_names:
            state_dict[name] = handle.get_tensor(name).detach()


def _load_meta_state_dict(path: str) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[int, tuple[str, str]]]:
    metadata, tensor_metadata = _read_safetensors_metadata(path)
    state_dict = _meta_state_dict(tensor_metadata)
    _materialize_quant_metadata_tensors(path, state_dict, tensor_metadata)
    source_ids = {id(tensor): (path, name) for name, tensor in state_dict.items()}
    return state_dict, metadata, source_ids


def _unet_model_options(weight_dtype: str) -> dict[str, Any]:
    model_options: dict[str, Any] = {}
    if weight_dtype == "fp8_e4m3fn":
        model_options["dtype"] = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e4m3fn_fast":
        model_options["dtype"] = torch.float8_e4m3fn
        model_options["fp8_optimizations"] = True
    elif weight_dtype == "fp8_e5m2":
        model_options["dtype"] = torch.float8_e5m2
    return model_options


@dataclass
class _CapturedStateLoad:
    module: torch.nn.Module
    state_dict: dict[str, torch.Tensor]


@dataclass
class _TensorCopyPlan:
    tensor: torch.Tensor
    storage_offset: int
    shape: torch.Size
    stride: tuple[int, ...]


def _source_view_plan(value: torch.Tensor, source_ids: dict[int, tuple[str, str]]) -> tuple[str, str, int, torch.Size, tuple[int, ...]] | None:
    source = source_ids.get(id(value))
    if source is not None:
        return source[0], source[1], 0, value.shape, tuple(value.stride())

    base = getattr(value, "_base", None)
    source = source_ids.get(id(base))
    if source is None:
        return None

    try:
        storage_offset = value.storage_offset()
    except Exception:
        return None
    return source[0], source[1], storage_offset, value.shape, tuple(value.stride())


def _copy_tensor_to_target(
    source: torch.Tensor,
    target: torch.Tensor,
    storage_offset: int,
    shape: torch.Size,
    stride: tuple[int, ...],
) -> None:
    if storage_offset or tuple(source.shape) != tuple(shape) or tuple(source.stride()) != tuple(stride):
        source = source.as_strided(shape, stride, storage_offset)
    if source.shape != target.shape:
        source = source.reshape(target.shape)
    target.copy_(source, non_blocking=True)


def _stream_instanttensor_loads(
    target_device: torch.device,
    captured_loads: list[_CapturedStateLoad],
    source_ids: dict[int, tuple[str, str]],
) -> list[tuple[list[str], list[str]]]:
    plans_by_file: dict[str, dict[str, list[_TensorCopyPlan]]] = {}
    results = []

    for captured in captured_loads:
        target_sd = captured.module.state_dict()
        unexpected = []
        loaded = set()

        for target_name, meta_tensor in captured.state_dict.items():
            if target_name not in target_sd:
                unexpected.append(target_name)
                continue
            view_plan = _source_view_plan(meta_tensor, source_ids)
            if view_plan is None:
                # Small synthetic tensors, for example comfy_quant metadata, are not
                # backed by the safetensors file. Copy them directly if possible.
                if meta_tensor.device.type != "meta":
                    target_sd[target_name].copy_(meta_tensor.to(device=target_sd[target_name].device), non_blocking=True)
                    loaded.add(target_name)
                    continue
                raise RuntimeError(f"Cannot stream transformed tensor {target_name}; refusing clone fallback.")
            source_path, source_name, storage_offset, shape, stride = view_plan
            plans_by_file.setdefault(source_path, {}).setdefault(source_name, []).append(
                _TensorCopyPlan(target_sd[target_name], storage_offset, shape, stride)
            )
            loaded.add(target_name)

        missing = [key for key in target_sd.keys() if key not in loaded and key in captured.state_dict]
        results.append((missing, unexpected))

    try:
        from instanttensor import safe_open
    except Exception as exc:
        raise RuntimeError(
            "The instanttensor package is required for CUDA instant loading. "
            "Install it in the ComfyUI environment used to run ComfyUI."
        ) from exc

    with torch.no_grad():
        for path, plans_by_source in plans_by_file.items():
            with safe_open(path, framework="pt", device=target_device) as handle:
                for source_name, source_tensor in handle.tensors():
                    plans = plans_by_source.get(source_name)
                    if not plans:
                        continue
                    for plan in plans:
                        _copy_tensor_to_target(
                            source_tensor,
                            plan.tensor,
                            plan.storage_offset,
                            plan.shape,
                            plan.stride,
                        )

    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    return results


def _module_load_device(module: torch.nn.Module, fallback: torch.device) -> torch.device:
    factory_kwargs = getattr(module, "factory_kwargs", None)
    if isinstance(factory_kwargs, dict) and factory_kwargs.get("device", None) is not None:
        return torch.device(factory_kwargs["device"])

    bias = getattr(module, "bias", None)
    if torch.is_tensor(bias):
        return bias.device

    return fallback


def _placeholder_for_state_load(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    if value.device.type == "meta":
        return torch.empty_strided(
            tuple(value.shape),
            tuple(value.stride()),
            dtype=value.dtype,
            device=device,
        )
    return value.to(device=device)


def _quant_init_state_dict(
    prefix: str,
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    target_device: torch.device,
) -> dict[str, torch.Tensor] | None:
    comfy_quant_key = f"{prefix}comfy_quant"
    weight_key = f"{prefix}weight"
    if comfy_quant_key not in state_dict or weight_key not in state_dict:
        return None

    load_device = _module_load_device(module, target_device)
    init_sd = {comfy_quant_key: state_dict[comfy_quant_key]}
    for param_name in ("weight", "weight_scale", "weight_scale_2", "input_scale"):
        key = f"{prefix}{param_name}"
        value = state_dict.get(key, None)
        if value is not None:
            init_sd[key] = _placeholder_for_state_load(value, load_device)
    return init_sd


def _prepare_quantized_modules_for_stream(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    target_device: torch.device,
) -> None:
    for module_name, child in module.named_modules():
        if not hasattr(child, "_load_scale_param"):
            continue

        prefix = f"{module_name}." if module_name else ""
        init_sd = _quant_init_state_dict(prefix, child, state_dict, target_device)
        if init_sd is None:
            continue

        missing: list[str] = []
        unexpected: list[str] = []
        errors: list[str] = []
        child._load_from_state_dict(init_sd, prefix, {}, False, missing, unexpected, errors)
        if errors:
            raise RuntimeError("; ".join(errors))


@contextlib.contextmanager
def _capture_state_loads(target_device: torch.device):
    captured: list[_CapturedStateLoad] = []
    original_load_state_dict = torch.nn.Module.load_state_dict
    original_load_models_gpu = comfy.model_management.load_models_gpu

    def patched_load_state_dict(module, state_dict, strict=True, assign=False):
        state_dict = dict(state_dict)
        _prepare_quantized_modules_for_stream(module, state_dict, target_device)
        captured.append(_CapturedStateLoad(module, state_dict))
        return [], []

    def patched_load_models_gpu(*args, **kwargs):
        return None

    torch.nn.Module.load_state_dict = patched_load_state_dict
    comfy.model_management.load_models_gpu = patched_load_models_gpu
    try:
        yield captured
    finally:
        torch.nn.Module.load_state_dict = original_load_state_dict
        comfy.model_management.load_models_gpu = original_load_models_gpu


def _contains_cuda_tensor(value: Any) -> bool:
    if torch.is_tensor(value):
        return value.device.type == "cuda"
    if isinstance(value, dict):
        return any(_contains_cuda_tensor(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_cuda_tensor(v) for v in value)
    return False


@contextlib.contextmanager
def instant_load_context():
    target_device = _cuda_device()
    original_load_torch_file = comfy.utils.load_torch_file
    original_clip_vision_load = comfy.clip_vision.load_torch_file
    original_load_state_dict = torch.nn.Module.load_state_dict
    original_text_encoder_device = comfy.model_management.text_encoder_device
    original_vae_device = comfy.model_management.vae_device

    def patched_load_torch_file(
        ckpt,
        safe_load=False,
        device=None,
        return_metadata=False,
    ):
        if _is_safetensors(str(ckpt)):
            return _instant_load_safetensors(str(ckpt), target_device, return_metadata=return_metadata)
        return original_load_torch_file(
            ckpt,
            safe_load=safe_load,
            device=device,
            return_metadata=return_metadata,
        )

    def patched_load_state_dict(module, state_dict, *args, **kwargs):
        if target_device.type == "cuda" and _contains_cuda_tensor(state_dict):
            args_list = list(args)
            if len(args_list) >= 2:
                args_list[1] = True
            else:
                kwargs["assign"] = True
            args = tuple(args_list)
        return original_load_state_dict(module, state_dict, *args, **kwargs)

    with _PATCH_LOCK:
        comfy.utils.load_torch_file = patched_load_torch_file
        comfy.clip_vision.load_torch_file = patched_load_torch_file
        torch.nn.Module.load_state_dict = patched_load_state_dict
        comfy.model_management.text_encoder_device = lambda: target_device
        comfy.model_management.vae_device = lambda: target_device
        try:
            yield target_device
        finally:
            comfy.utils.load_torch_file = original_load_torch_file
            comfy.clip_vision.load_torch_file = original_clip_vision_load
            torch.nn.Module.load_state_dict = original_load_state_dict
            comfy.model_management.text_encoder_device = original_text_encoder_device
            comfy.model_management.vae_device = original_vae_device


def _iter_model_patchers(value: Any, seen: set[int] | None = None) -> Iterable[comfy.model_patcher.ModelPatcher]:
    if seen is None:
        seen = set()
    if id(value) in seen:
        return
    seen.add(id(value))

    if isinstance(value, comfy.model_patcher.ModelPatcher):
        yield value
        return

    patcher = getattr(value, "patcher", None)
    if isinstance(patcher, comfy.model_patcher.ModelPatcher):
        yield patcher

    control_patcher = getattr(value, "control_model_wrapped", None)
    if isinstance(control_patcher, comfy.model_patcher.ModelPatcher):
        yield control_patcher

    if isinstance(value, dict):
        for item in value.values():
            yield from _iter_model_patchers(item, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_model_patchers(item, seen)


def _register_cache_managed_models(result: Any, target_device: torch.device) -> Any:
    patchers = []
    seen = set()
    for patcher in _iter_model_patchers(result):
        if id(patcher) in seen:
            continue
        seen.add(id(patcher))
        if patcher.load_device.type == "cuda":
            patchers.append(patcher)
        else:
            has_cuda_weight = any(p.device.type == "cuda" for p in patcher.model.parameters(recurse=True))
            if has_cuda_weight:
                logging.warning(
                    "InstantTensor loaded %s weights on CUDA, but the ComfyUI patcher load_device is %s. "
                    "This model will be moved back under ComfyUI's normal policy on first use.",
                    patcher.model.__class__.__name__,
                    patcher.load_device,
                )
    if patchers:
        comfy.model_management.load_models_gpu(patchers, force_full_load=True)
    return result


def _instant_call(callback):
    with instant_load_context() as target_device:
        result = callback()
    return _register_cache_managed_models(result, target_device)


def _cuda_model_options(model_options: dict[str, Any] | None = None) -> dict[str, Any]:
    options = {} if model_options is None else model_options.copy()
    options["load_device"] = _cuda_device()
    options["offload_device"] = comfy.model_management.text_encoder_offload_device()
    return options


def _log_stream_results(label: str, results: list[tuple[list[str], list[str]]]) -> None:
    for missing, unexpected in results:
        if missing:
            logging.warning("%s missing: %s", label, missing)
        if unexpected:
            logging.debug("%s unexpected: %s", label, unexpected)


def _instant_stream_clip(
    ckpt_paths: list[str],
    embedding_directory,
    clip_type,
    model_options: dict[str, Any],
    target_device: torch.device,
):
    clip_data = []
    source_ids: dict[int, tuple[str, str]] = {}
    for path in ckpt_paths:
        sd, metadata, ids = _load_meta_state_dict(path)
        source_ids.update(ids)
        if model_options.get("custom_operations", None) is None:
            sd, metadata = comfy.utils.convert_old_quants(sd, model_prefix="", metadata=metadata)
        clip_data.append(sd)

    with _capture_state_loads(target_device) as captured:
        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_data,
            embedding_directory=embedding_directory,
            clip_type=clip_type,
            model_options=model_options,
        )
    if not captured:
        raise RuntimeError("CLIP loader did not capture any state_dict load calls; refusing clone fallback.")

    results = _stream_instanttensor_loads(target_device, captured, source_ids)
    _log_stream_results("clip", results)
    clip.patcher.cached_patcher_init = (
        comfy.sd.load_clip_model_patcher,
        (ckpt_paths, embedding_directory, clip_type, model_options),
    )
    return clip


def _instant_stream_vae(vae_path: str, target_device: torch.device):
    sd, metadata, source_ids = _load_meta_state_dict(vae_path)
    with _capture_state_loads(target_device) as captured:
        vae = comfy.sd.VAE(sd=sd, metadata=metadata)
    if not captured:
        raise RuntimeError("VAE loader did not capture any state_dict load calls; refusing clone fallback.")
    vae.throw_exception_if_invalid()
    results = _stream_instanttensor_loads(target_device, captured, source_ids)
    _log_stream_results("vae", results)
    return vae


def _instant_stream_checkpoint_guess_config(
    ckpt_path: str,
    output_vae=True,
    output_clip=True,
    output_clipvision=False,
    embedding_directory=None,
    output_model=True,
    model_options: dict[str, Any] | None = None,
    te_model_options: dict[str, Any] | None = None,
    target_device: torch.device | None = None,
):
    model_options = {} if model_options is None else model_options
    te_model_options = {} if te_model_options is None else te_model_options
    target_device = _cuda_device() if target_device is None else target_device

    sd, metadata, source_ids = _load_meta_state_dict(ckpt_path)
    original_unet_initial_load_device = comfy.model_management.unet_inital_load_device
    comfy.model_management.unet_inital_load_device = lambda parameters, dtype: target_device
    try:
        with _capture_state_loads(target_device) as captured:
            out = comfy.sd.load_state_dict_guess_config(
                sd,
                output_vae=output_vae,
                output_clip=output_clip,
                output_clipvision=output_clipvision,
                embedding_directory=embedding_directory,
                output_model=output_model,
                model_options=model_options,
                te_model_options=te_model_options,
                metadata=metadata,
            )
    finally:
        comfy.model_management.unet_inital_load_device = original_unet_initial_load_device
    if out is None:
        raise RuntimeError(f"Could not detect checkpoint model type of: {ckpt_path}")
    if not captured:
        raise RuntimeError("Checkpoint loader did not capture any state_dict load calls; refusing clone fallback.")

    results = _stream_instanttensor_loads(target_device, captured, source_ids)
    _log_stream_results("checkpoint", results)

    model_patcher, clip, vae, _clipvision = out
    if model_patcher is not None:
        model_patcher.model.model_loaded_weight_memory = comfy.model_management.module_size(model_patcher.model)
        model_patcher.model.device = target_device
        model_patcher.cached_patcher_init = (
            comfy.sd.load_checkpoint_guess_config_model_only,
            (ckpt_path, embedding_directory, model_options, te_model_options),
        )
    if clip is not None:
        clip.patcher.cached_patcher_init = (
            comfy.sd.load_checkpoint_guess_config_clip_only,
            (ckpt_path, embedding_directory, model_options, te_model_options),
        )
    return out


def _instant_stream_diffusion_model(
    unet_path: str,
    model_options: dict[str, Any],
    target_device: torch.device,
):
    meta_sd, metadata, source_ids = _load_meta_state_dict(unet_path)

    dtype = model_options.get("dtype", None)
    diffusion_model_prefix = comfy.sd.model_detection.unet_prefix_from_state_dict(meta_sd)
    temp_sd = comfy.utils.state_dict_prefix_replace(meta_sd, {diffusion_model_prefix: ""}, filter_keys=True)
    if len(temp_sd) > 0:
        meta_sd = temp_sd

    custom_operations = model_options.get("custom_operations", None)
    if custom_operations is None:
        meta_sd, metadata = comfy.utils.convert_old_quants(meta_sd, "", metadata=metadata)

    parameters = comfy.utils.calculate_parameters(meta_sd)
    weight_dtype = comfy.utils.weight_dtype(meta_sd)
    model_config = comfy.sd.model_detection.model_config_from_unet(meta_sd, "", metadata=metadata)

    if model_config is not None:
        new_sd = meta_sd
    else:
        new_sd = comfy.sd.model_detection.convert_diffusers_mmdit(meta_sd, "")
        if new_sd is not None:
            model_config = comfy.sd.model_detection.model_config_from_unet(new_sd, "")
            if model_config is None:
                return None
        else:
            model_config = comfy.sd.model_detection.model_config_from_diffusers_unet(meta_sd)
            if model_config is None:
                return None

            diffusers_keys = comfy.utils.unet_to_diffusers(model_config.unet_config)
            new_sd = {}
            for key, mapped_key in diffusers_keys.items():
                if key in meta_sd:
                    new_sd[mapped_key] = meta_sd.pop(key)
                else:
                    logging.warning("%s %s", mapped_key, key)

    unet_weight_dtype = list(model_config.supported_inference_dtypes)
    if model_config.quant_config is not None:
        weight_dtype = None

    if dtype is None:
        unet_dtype = comfy.model_management.unet_dtype(
            model_params=parameters,
            supported_dtypes=unet_weight_dtype,
            weight_dtype=weight_dtype,
        )
    else:
        unet_dtype = dtype

    if model_config.quant_config is not None:
        manual_cast_dtype = comfy.model_management.unet_manual_cast(
            None,
            target_device,
            model_config.supported_inference_dtypes,
        )
    else:
        manual_cast_dtype = comfy.model_management.unet_manual_cast(
            unet_dtype,
            target_device,
            model_config.supported_inference_dtypes,
        )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if custom_operations is not None:
        model_config.custom_operations = custom_operations

    if model_options.get("fp8_optimizations", False):
        model_config.optimizations["fp8"] = True

    model = model_config.get_model(new_sd, "", device=target_device)
    processed_meta_sd = model_config.process_unet_state_dict(dict(new_sd))
    (missing, unexpected), = _stream_instanttensor_loads(
        target_device,
        [_CapturedStateLoad(model.diffusion_model, processed_meta_sd)],
        source_ids,
    )
    if missing:
        logging.warning("unet missing: %s", missing)
    if unexpected:
        logging.warning("unet unexpected: %s", unexpected)

    model_patcher = comfy.model_patcher.CoreModelPatcher(
        model,
        load_device=target_device,
        offload_device=comfy.model_management.unet_offload_device(),
    )
    model_patcher.model.model_loaded_weight_memory = comfy.model_management.module_size(model_patcher.model)
    model_patcher.model.device = target_device
    model_patcher.cached_patcher_init = (comfy.sd.load_diffusion_model, (unet_path, model_options))
    return model_patcher


class InstantCheckpointLoaderSimple(nodes.CheckpointLoaderSimple):
    @classmethod
    def INPUT_TYPES(cls):
        return super().INPUT_TYPES()

    CATEGORY = "loaders/instanttensor"
    FUNCTION = "load_checkpoint"

    def load_checkpoint(self, ckpt_name):
        def load():
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
            if _is_safetensors(ckpt_path):
                out = _instant_stream_checkpoint_guess_config(
                    ckpt_path,
                    output_vae=True,
                    output_clip=True,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    te_model_options=_cuda_model_options(),
                    target_device=_cuda_device(),
                )
                return out[:3]
            out = comfy.sd.load_checkpoint_guess_config(
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                te_model_options=_cuda_model_options(),
            )
            return out[:3]

        return _instant_call(load)


class InstantCheckpointLoader(nodes.CheckpointLoader):
    @classmethod
    def INPUT_TYPES(cls):
        return super().INPUT_TYPES()

    CATEGORY = "advanced/loaders/instanttensor"
    FUNCTION = "load_checkpoint"
    DEPRECATED = True

    def load_checkpoint(self, config_name, ckpt_name):
        def load():
            config_path = folder_paths.get_full_path("configs", config_name)
            ckpt_path = folder_paths.get_full_path_or_raise("checkpoints", ckpt_name)
            if _is_safetensors(ckpt_path):
                model, clip, vae, _ = _instant_stream_checkpoint_guess_config(
                    ckpt_path,
                    output_vae=True,
                    output_clip=True,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    target_device=_cuda_device(),
                )
                with open(config_path, "r") as stream:
                    config = yaml.safe_load(stream)
                model_config_params = config["model"]["params"]
                clip_config = model_config_params["cond_stage_config"]

                if model_config_params.get("parameterization") == "v":
                    patched_model = model.clone()

                    class ModelSamplingAdvanced(comfy.model_sampling.ModelSamplingDiscrete, comfy.model_sampling.V_PREDICTION):
                        pass

                    patched_model.add_object_patch("model_sampling", ModelSamplingAdvanced(model.model.model_config))
                    model = patched_model

                layer_idx = clip_config.get("params", {}).get("layer_idx", None)
                if layer_idx is not None:
                    clip.clip_layer(layer_idx)
                return (model, clip, vae)
            return comfy.sd.load_checkpoint(
                config_path,
                ckpt_path,
                output_vae=True,
                output_clip=True,
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
            )

        return _instant_call(load)


class InstantUNETLoader(nodes.UNETLoader):
    @classmethod
    def INPUT_TYPES(cls):
        return super().INPUT_TYPES()

    CATEGORY = "advanced/loaders/instanttensor"
    FUNCTION = "load_unet"

    def load_unet(self, unet_name, weight_dtype):
        def load():
            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
            if _is_safetensors(unet_path):
                try:
                    model = _instant_stream_diffusion_model(
                        unet_path,
                        _unet_model_options(weight_dtype),
                        _cuda_device(),
                    )
                    if model is not None:
                        return (model,)
                except Exception:
                    logging.exception(
                        "InstantTensor streamed diffusion load failed for %s; "
                        "not falling back to the clone state_dict path because that can double memory.",
                        unet_path,
                    )
                    raise
            return super(InstantUNETLoader, self).load_unet(unet_name, weight_dtype)

        return _instant_call(load)


class InstantCLIPLoader(nodes.CLIPLoader):
    @classmethod
    def INPUT_TYPES(cls):
        return _without_cpu_device(super().INPUT_TYPES())

    CATEGORY = "advanced/loaders/instanttensor"
    FUNCTION = "load_clip"

    def load_clip(self, clip_name, type="stable_diffusion"):
        def load():
            clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
            clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
            model_options = _cuda_model_options()
            if _is_safetensors(clip_path):
                clip = _instant_stream_clip(
                    [clip_path],
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=clip_type,
                    model_options=model_options,
                    target_device=_cuda_device(),
                )
            else:
                clip = comfy.sd.load_clip(
                    ckpt_paths=[clip_path],
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=clip_type,
                    model_options=model_options,
                )
            return (clip,)

        return _instant_call(load)


class InstantDualCLIPLoader(nodes.DualCLIPLoader):
    @classmethod
    def INPUT_TYPES(cls):
        return _without_cpu_device(super().INPUT_TYPES())

    CATEGORY = "advanced/loaders/instanttensor"
    FUNCTION = "load_clip"

    def load_clip(self, clip_name1, clip_name2, type):
        def load():
            clip_type = getattr(comfy.sd.CLIPType, type.upper(), comfy.sd.CLIPType.STABLE_DIFFUSION)
            clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
            clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
            paths = [clip_path1, clip_path2]
            model_options = _cuda_model_options()
            if all(_is_safetensors(path) for path in paths):
                clip = _instant_stream_clip(
                    paths,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=clip_type,
                    model_options=model_options,
                    target_device=_cuda_device(),
                )
            else:
                clip = comfy.sd.load_clip(
                    ckpt_paths=paths,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=clip_type,
                    model_options=model_options,
                )
            return (clip,)

        return _instant_call(load)


class InstantTripleCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name1": (folder_paths.get_filename_list("text_encoders"),),
                "clip_name2": (folder_paths.get_filename_list("text_encoders"),),
                "clip_name3": (folder_paths.get_filename_list("text_encoders"),),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "advanced/loaders/instanttensor"

    def load_clip(self, clip_name1, clip_name2, clip_name3):
        def load():
            paths = [
                folder_paths.get_full_path_or_raise("text_encoders", clip_name1),
                folder_paths.get_full_path_or_raise("text_encoders", clip_name2),
                folder_paths.get_full_path_or_raise("text_encoders", clip_name3),
            ]
            model_options = _cuda_model_options()
            if all(_is_safetensors(path) for path in paths):
                clip = _instant_stream_clip(
                    paths,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,
                    model_options=model_options,
                    target_device=_cuda_device(),
                )
            else:
                clip = comfy.sd.load_clip(
                    ckpt_paths=paths,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    model_options=model_options,
                )
            return (clip,)

        return _instant_call(load)


class InstantQuadrupleCLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_name1": (folder_paths.get_filename_list("text_encoders"),),
                "clip_name2": (folder_paths.get_filename_list("text_encoders"),),
                "clip_name3": (folder_paths.get_filename_list("text_encoders"),),
                "clip_name4": (folder_paths.get_filename_list("text_encoders"),),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "advanced/loaders/instanttensor"

    def load_clip(self, clip_name1, clip_name2, clip_name3, clip_name4):
        def load():
            paths = [
                folder_paths.get_full_path_or_raise("text_encoders", clip_name1),
                folder_paths.get_full_path_or_raise("text_encoders", clip_name2),
                folder_paths.get_full_path_or_raise("text_encoders", clip_name3),
                folder_paths.get_full_path_or_raise("text_encoders", clip_name4),
            ]
            model_options = _cuda_model_options()
            if all(_is_safetensors(path) for path in paths):
                clip = _instant_stream_clip(
                    paths,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,
                    model_options=model_options,
                    target_device=_cuda_device(),
                )
            else:
                clip = comfy.sd.load_clip(
                    ckpt_paths=paths,
                    embedding_directory=folder_paths.get_folder_paths("embeddings"),
                    model_options=model_options,
                )
            return (clip,)

        return _instant_call(load)


class InstantVAELoader(nodes.VAELoader):
    @classmethod
    def INPUT_TYPES(cls):
        return super().INPUT_TYPES()

    CATEGORY = "loaders/instanttensor"
    FUNCTION = "load_vae"

    def load_vae(self, vae_name):
        def load():
            if vae_name not in self.image_taes:
                vae_path = folder_paths.get_full_path("vae", vae_name)
                if vae_path is None:
                    vae_path = folder_paths.get_full_path("vae_approx", vae_name)
                if vae_path is not None and _is_safetensors(vae_path):
                    return (_instant_stream_vae(vae_path, _cuda_device()),)
            return super(InstantVAELoader, self).load_vae(vae_name)

        return _instant_call(load)


class InstantControlNetLoader(nodes.ControlNetLoader):
    @classmethod
    def INPUT_TYPES(cls):
        return super().INPUT_TYPES()

    CATEGORY = "loaders/instanttensor"
    FUNCTION = "load_controlnet"

    def load_controlnet(self, control_net_name):
        def load():
            return super(InstantControlNetLoader, self).load_controlnet(control_net_name)

        return _instant_call(load)


class InstantDiffControlNetLoader(nodes.DiffControlNetLoader):
    @classmethod
    def INPUT_TYPES(cls):
        return super().INPUT_TYPES()

    CATEGORY = "loaders/instanttensor"
    FUNCTION = "load_controlnet"

    def load_controlnet(self, model, control_net_name):
        def load():
            return super(InstantDiffControlNetLoader, self).load_controlnet(model, control_net_name)

        return _instant_call(load)


NODE_CLASS_MAPPINGS = {
    "InstantCheckpointLoaderSimple": InstantCheckpointLoaderSimple,
    "InstantCheckpointLoader": InstantCheckpointLoader,
    "InstantUNETLoader": InstantUNETLoader,
    "InstantCLIPLoader": InstantCLIPLoader,
    "InstantDualCLIPLoader": InstantDualCLIPLoader,
    "InstantTripleCLIPLoader": InstantTripleCLIPLoader,
    "InstantQuadrupleCLIPLoader": InstantQuadrupleCLIPLoader,
    "InstantVAELoader": InstantVAELoader,
    "InstantControlNetLoader": InstantControlNetLoader,
    "InstantDiffControlNetLoader": InstantDiffControlNetLoader,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "InstantCheckpointLoaderSimple": "Instant Load Checkpoint",
    "InstantCheckpointLoader": "Instant Load Checkpoint With Config (DEPRECATED)",
    "InstantUNETLoader": "Instant Load Diffusion Model",
    "InstantCLIPLoader": "Instant Load CLIP",
    "InstantDualCLIPLoader": "Instant Load Dual CLIP",
    "InstantTripleCLIPLoader": "Instant Load Triple CLIP",
    "InstantQuadrupleCLIPLoader": "Instant Load Quadruple CLIP",
    "InstantVAELoader": "Instant Load VAE",
    "InstantControlNetLoader": "Instant Load ControlNet Model",
    "InstantDiffControlNetLoader": "Instant Load ControlNet Model (diff)",
}
