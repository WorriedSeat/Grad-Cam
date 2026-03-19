import torch
import torch.nn.functional as F
import numpy as np


def _default_target_layer(model):
    """Return default target layer name based on model type."""
    name = type(model).__name__
    if name == "ResEmoteNet":
        return "res_blocks.2"
    return "features.8"


def compute_gradcam(model, input_tensor, target_class, target_layer_name=None):
    """
    Grad-CAM для EfficientNet-B4 и ResEmoteNet.

    EfficientNet-B4: "features.8" (финальный Conv перед GAP)
    ResEmoteNet: "res_blocks.2" (последний Residual block перед AAP)

    Args:
        model:             обученная модель в режиме eval
        input_tensor:      тензор [1, C, H, W], БЕЗ requires_grad
        target_class:      индекс целевого класса
        target_layer_name: имя слоя для визуализации (auto-detect если None)

    Returns:
        cam: тензор [H, W] со значениями 0..1
    """
    if target_layer_name is None:
        target_layer_name = _default_target_layer(model)

    named_modules = dict(model.named_modules())
    if target_layer_name not in named_modules:
        available = [
            k for k in named_modules
            if "conv" in k or "features" in k or "res_blocks" in k
        ][:20]
        raise ValueError(
            f"Layer '{target_layer_name}' not found!\n"
            f"Available layers (sample): {available}"
        )

    target_layer = named_modules[target_layer_name]

    activations_store = [None]
    gradients_store = [None]

    def forward_hook(module, inp, output):
        activations_store[0] = output

    def backward_hook(module, grad_input, grad_output):
        gradients_store[0] = grad_output[0].detach()

    fwd_handle = target_layer.register_forward_hook(forward_hook)
    bwd_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.eval()

        with torch.set_grad_enabled(True):
            output = model(input_tensor)
            score = output[0, target_class]
            model.zero_grad()
            score.backward()

        if gradients_store[0] is None or activations_store[0] is None:
            raise RuntimeError(
                "Градиенты или активации не захвачены. "
                "Убедитесь, что backward проходит через целевой слой."
            )

        grads = gradients_store[0]           # [1, C, h, w]
        acts  = activations_store[0].detach()  # [1, C, h, w]

        weights = grads.mean(dim=[0, 2, 3])  # [C]

        cam = torch.einsum("k, k h w -> h w", weights, acts[0])  # [h, w]

        cam = F.relu(cam)  

        cam_np = cam.cpu().numpy()
        p_low  = np.percentile(cam_np, 1)
        p_high = np.percentile(cam_np, 99)
        if p_high > p_low:
            cam_np = np.clip((cam_np - p_low) / (p_high - p_low), 0, 1)
        else:
            cam_np = np.zeros_like(cam_np)
        cam = torch.tensor(cam_np, dtype=torch.float32, device=acts.device)

        h, w = input_tensor.shape[2], input_tensor.shape[3]
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        ).squeeze()

    finally:
        fwd_handle.remove()
        bwd_handle.remove()

    return cam