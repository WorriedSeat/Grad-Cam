import torch
import numpy as np
import torch.nn as nn

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks
        self.forward_hook = target_layer.register_forward_hook(self.save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # If no label take prediction
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # One-hot for chosen class
        one_hot = torch.zeros_like(output)
        one_hot[:, target_class] = 1.0
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Gradients and activations
        gradients = self.gradients          # [1, C, H, W]
        activations = self.activations     # [1, C, H, W]
        
        # Global average pooling
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [1, C, 1, 1]
        
        # Sum over weighted activations
        cam = torch.sum(weights * activations, dim=1).squeeze(0)   # [H, W]
        
        # ReLU + normalization
        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-8)
        
        return cam.cpu().numpy()

    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()
