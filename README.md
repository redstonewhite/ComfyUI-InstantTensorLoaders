# ComfyUI InstantTensor Loaders

InstantTensor-backed variants of ComfyUI's built-in model loader nodes.

These nodes are CUDA-only because InstantTensor currently only supports CUDA devices. Generic loaders use `instanttensor.safe_open`, clone each tensor once into CUDA, force `load_state_dict(assign=True)` while constructing the model, and register returned Comfy `ModelPatcher` objects with `comfy.model_management`.

The main safetensors loaders use a lower-memory streaming path: they read the safetensors header to let ComfyUI detect and construct the model, then stream tensors from InstantTensor directly into the destination module with `copy_()`. This avoids holding a full CUDA state dict beside the model while still handing returned `MODEL`, `CLIP`, and `VAE` objects to ComfyUI's model manager.

If this streaming path cannot handle a safetensors model in the diffusion, checkpoint, CLIP, or VAE loaders, it raises an error instead of silently falling back to the clone-state_dict path, because that fallback can recreate the Spark double-memory problem.

ComfyUI must be running with a CUDA torch device. The Instant CLIP nodes intentionally do not expose ComfyUI's CPU device option.

## Nodes

- Instant Load Checkpoint
- Instant Load Checkpoint With Config
- Instant Load Diffusion Model
- Instant Load CLIP / Dual CLIP / Triple CLIP / Quadruple CLIP
- Instant Load VAE
- Instant Load ControlNet Model / ControlNet Model (diff)

## Memory Behavior

InstantTensor tensors point at a reusable transfer buffer while iterating. The generic path clones each tensor once so the tensor can survive buffer reuse. The streaming path instead copies each temporary InstantTensor tensor immediately into the final model parameter or buffer, so the only persistent model-sized allocation is the Comfy-managed model itself.

Returned `MODEL`, `CLIP`, `VAE`, and `CONTROL_NET` patchers are handed to ComfyUI's model manager immediately, so they participate in normal model unloading.
