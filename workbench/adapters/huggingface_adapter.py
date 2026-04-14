# Copyright (c) 2026 Chris. All rights reserved.
# Licensed under the Business Source License 1.1 — see LICENSE file.

"""
HuggingFaceAdapter — Specialized PyTorch adapter for HuggingFace models.

Auto-detects model architecture (GPT-2, BERT, LLaMA, etc.) and extracts
richer information: tokenizer integration, generation support, architecture-
specific layer naming, and auto-mapping of attention layers.

Usage:
    adapter = HuggingFaceAdapter.from_pretrained("gpt2")
    output = adapter.predict("Hello, world!")
    attention = adapter.get_attention("Hello, world!")
"""

import numpy as np
from typing import Any, Callable, Optional

from .protocol import (
    ModelAdapter, ModelInfo, Capability, LayerActivation,
    AttentionMap, InterventionResult,
)


def _require_transformers():
    try:
        import transformers
        return transformers
    except ImportError:
        raise ImportError(
            "HuggingFaceAdapter requires the transformers library. "
            "Install with: pip install transformers"
        )


class HuggingFaceAdapter(ModelAdapter):
    """
    Tier 2 adapter specialized for HuggingFace Transformers.

    Adds tokenizer integration, generation support, and architecture-aware
    layer inspection on top of the standard PyTorch capabilities.
    """

    def __init__(self, model=None, tokenizer=None, name: str = "HuggingFace Model",
                 device: str = None):
        self._model = model
        self._tokenizer = tokenizer
        self._name = name
        self._device = device or "cpu"
        self._arch_info = None

        if model is not None:
            self._detect_architecture()

    @classmethod
    def from_pretrained(cls, model_name: str, device: str = None,
                        **kwargs) -> "HuggingFaceAdapter":
        """
        Load a model and tokenizer from HuggingFace Hub.

        Example:
            adapter = HuggingFaceAdapter.from_pretrained("gpt2")
            adapter = HuggingFaceAdapter.from_pretrained("bert-base-uncased")
        """
        transformers = _require_transformers()
        import torch

        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, **kwargs)
        model = transformers.AutoModel.from_pretrained(model_name, **kwargs)
        model = model.to(dev)
        model.eval()

        return cls(model=model, tokenizer=tokenizer, name=model_name, device=dev)

    def _detect_architecture(self):
        """Auto-detect model architecture and layer structure."""
        model_type = type(self._model).__name__.lower()
        config = getattr(self._model, 'config', None)

        self._arch_info = {
            "model_type": model_type,
            "num_layers": getattr(config, 'num_hidden_layers', None),
            "num_heads": getattr(config, 'num_attention_heads', None),
            "hidden_size": getattr(config, 'hidden_size', None),
            "vocab_size": getattr(config, 'vocab_size', None),
            "max_position": getattr(config, 'max_position_embeddings', None),
        }

        # Detect attention layer naming convention
        self._attn_layer_pattern = self._find_attention_pattern()

    def _find_attention_pattern(self) -> str:
        """Find how attention layers are named in this model."""
        for name, module in self._model.named_modules():
            module_type = type(module).__name__.lower()
            if 'attention' in module_type or 'multiheadattention' in module_type:
                return name
        return ""

    # --- Tier 1: Required ---

    def predict(self, input_data: Any) -> Any:
        """
        Run inference. Accepts strings (tokenized automatically),
        token IDs, or numpy arrays.
        """
        import torch

        self._model.eval()

        if isinstance(input_data, str):
            if self._tokenizer is None:
                raise ValueError("String input requires a tokenizer")
            encoded = self._tokenizer(input_data, return_tensors="pt",
                                      padding=True, truncation=True)
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            with torch.no_grad():
                output = self._model(**encoded)
        elif isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).to(self._device)
            with torch.no_grad():
                output = self._model(input_tensor)
        else:
            input_tensor = torch.tensor(input_data).to(self._device)
            with torch.no_grad():
                output = self._model(input_tensor)

        # Extract the main output tensor
        if hasattr(output, 'last_hidden_state'):
            return output.last_hidden_state.cpu().numpy()
        elif hasattr(output, 'logits'):
            return output.logits.cpu().numpy()
        elif isinstance(output, tuple):
            return output[0].cpu().numpy()
        else:
            return output.cpu().numpy()

    def get_info(self) -> ModelInfo:
        config = getattr(self._model, 'config', None)
        param_count = sum(p.numel() for p in self._model.parameters())

        arch = "transformer"
        if config:
            model_type = getattr(config, 'model_type', 'unknown')
        else:
            model_type = type(self._model).__name__

        return ModelInfo(
            name=self._name,
            framework="huggingface",
            architecture=arch,
            parameter_count=param_count,
            layer_count=self._arch_info.get("num_layers", 0) if self._arch_info else 0,
            dtype=str(next(self._model.parameters()).dtype),
            device=self._device,
            extra={
                "model_type": model_type,
                "num_heads": self._arch_info.get("num_heads") if self._arch_info else None,
                "hidden_size": self._arch_info.get("hidden_size") if self._arch_info else None,
                "vocab_size": self._arch_info.get("vocab_size") if self._arch_info else None,
                "has_tokenizer": self._tokenizer is not None,
            },
        )

    def capabilities(self) -> Capability:
        return (Capability.PREDICT | Capability.INFO | Capability.ACTIVATIONS |
                Capability.GRADIENTS | Capability.ATTENTION | Capability.INTERVENE |
                Capability.PARAMETERS)

    # --- Tier 2: Deep inspection ---

    def get_activations(self, input_data: Any, layers: list = None) -> list:
        """Extract activations using output_hidden_states=True."""
        import torch

        self._model.eval()

        # Try to get hidden states natively
        if isinstance(input_data, str) and self._tokenizer:
            encoded = self._tokenizer(input_data, return_tensors="pt",
                                      padding=True, truncation=True)
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            with torch.no_grad():
                output = self._model(**encoded, output_hidden_states=True)
        else:
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.from_numpy(input_data).to(self._device)
            else:
                input_tensor = torch.tensor(input_data).to(self._device)
            with torch.no_grad():
                output = self._model(input_tensor, output_hidden_states=True)

        results = []

        if hasattr(output, 'hidden_states') and output.hidden_states is not None:
            for i, hidden in enumerate(output.hidden_states):
                layer_name = f"hidden_state_{i}"
                if layers is not None and layer_name not in layers:
                    continue
                h = hidden.cpu().numpy()
                results.append(LayerActivation(
                    layer_name=layer_name,
                    shape=h.shape,
                    values=h,
                    dtype=str(hidden.dtype),
                    metadata={
                        "layer_index": i,
                        "is_embedding": i == 0,
                        "mean": float(h.mean()),
                        "std": float(h.std()),
                    },
                ))

        # Fallback to hook-based extraction if hidden_states not available
        if not results:
            return self._hook_based_activations(input_data, layers)

        return results

    def _hook_based_activations(self, input_data, layers=None):
        """Fallback: use forward hooks to capture activations."""
        import torch
        captured = {}
        hooks = []

        for name, module in self._model.named_modules():
            if layers is not None and name not in layers:
                continue
            if name == "":
                continue

            def make_hook(layer_name):
                def hook(mod, inp, out):
                    output = out[0] if isinstance(out, tuple) else out
                    if isinstance(output, torch.Tensor):
                        captured[layer_name] = output.detach().cpu()
                return hook

            hooks.append(module.register_forward_hook(make_hook(name)))

        self.predict(input_data)

        for h in hooks:
            h.remove()

        results = []
        for name, tensor in captured.items():
            results.append(LayerActivation(
                layer_name=name,
                shape=tuple(tensor.shape),
                values=tensor.numpy(),
                dtype=str(tensor.dtype),
            ))
        return results

    def get_attention(self, input_data: Any, layers: list = None) -> list:
        """Extract attention weights using output_attentions=True."""
        import torch

        self._model.eval()

        if isinstance(input_data, str) and self._tokenizer:
            encoded = self._tokenizer(input_data, return_tensors="pt",
                                      padding=True, truncation=True)
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            with torch.no_grad():
                output = self._model(**encoded, output_attentions=True)
        else:
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.from_numpy(input_data).to(self._device)
            else:
                input_tensor = torch.tensor(input_data).to(self._device)
            with torch.no_grad():
                output = self._model(input_tensor, output_attentions=True)

        results = []

        if hasattr(output, 'attentions') and output.attentions is not None:
            for i, attn in enumerate(output.attentions):
                layer_name = f"attention_layer_{i}"
                if layers is not None and layer_name not in layers:
                    continue

                w = attn.cpu().numpy()
                num_heads = w.shape[1]

                # Compute per-head entropy
                eps = 1e-8
                entropy = -(w * np.log(w + eps)).sum(axis=-1).mean(axis=(0, 2))

                results.append(AttentionMap(
                    layer_name=layer_name,
                    num_heads=num_heads,
                    weights=w,
                    metadata={
                        "layer_index": i,
                        "head_entropy": entropy.tolist(),
                        "head_max_attn": w.max(axis=-1).mean(axis=(0, 2)).tolist(),
                    },
                ))

        return results

    def get_gradients(self, input_data: Any, target: Any,
                      layers: list = None) -> list:
        """Compute gradients via backpropagation."""
        import torch
        captured = {}
        hooks = []

        for name, module in self._model.named_modules():
            if layers is not None and name not in layers:
                continue
            if name == "":
                continue

            def make_hook(layer_name):
                def hook(mod, grad_in, grad_out):
                    if grad_out[0] is not None:
                        captured[layer_name] = grad_out[0].detach().cpu()
                return hook

            hooks.append(module.register_full_backward_hook(make_hook(name)))

        self._model.train()

        if isinstance(input_data, str) and self._tokenizer:
            encoded = self._tokenizer(input_data, return_tensors="pt",
                                      padding=True, truncation=True)
            encoded = {k: v.to(self._device) for k, v in encoded.items()}
            output = self._model(**encoded)
        else:
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.from_numpy(input_data).float().to(self._device)
            else:
                input_tensor = torch.tensor(input_data).float().to(self._device)
            input_tensor.requires_grad_(True)
            output = self._model(input_tensor)

        # Get main output
        if hasattr(output, 'last_hidden_state'):
            out = output.last_hidden_state
        elif hasattr(output, 'logits'):
            out = output.logits
        elif isinstance(output, tuple):
            out = output[0]
        else:
            out = output

        if isinstance(target, int):
            loss = out.flatten()[target]
        else:
            target_tensor = torch.tensor(target).float().to(self._device)
            loss = (out * target_tensor).sum()

        loss.backward()

        for h in hooks:
            h.remove()

        self._model.eval()

        results = []
        for name, tensor in captured.items():
            results.append(LayerActivation(
                layer_name=name,
                shape=tuple(tensor.shape),
                values=tensor.numpy(),
                dtype=str(tensor.dtype),
                metadata={"type": "gradient"},
            ))
        return results

    def intervene(self, input_data: Any, layer_name: str,
                  fn: Callable) -> InterventionResult:
        """Modify activations at a layer during forward pass."""
        import torch

        original_output = self.predict(input_data)

        hook_handle = None
        for name, module in self._model.named_modules():
            if name == layer_name:
                def make_hook(intervention_fn):
                    def hook(mod, inp, out):
                        if isinstance(out, tuple):
                            tensor = out[0]
                        else:
                            tensor = out
                        if isinstance(tensor, torch.Tensor):
                            modified = intervention_fn(tensor.detach().cpu().numpy())
                            modified_tensor = torch.from_numpy(modified).to(tensor.device).to(tensor.dtype)
                            if isinstance(out, tuple):
                                return (modified_tensor,) + out[1:]
                            return modified_tensor
                        return out
                    return hook

                hook_handle = module.register_forward_hook(make_hook(fn))
                break

        if hook_handle is None:
            raise ValueError(f"Layer '{layer_name}' not found in model")

        modified_output = self.predict(input_data)
        hook_handle.remove()

        return InterventionResult(
            original_output=original_output,
            modified_output=modified_output,
            layer_name=layer_name,
            intervention_fn=str(fn),
        )

    def get_parameters(self, layers: list = None) -> dict:
        params = {}
        for name, module in self._model.named_modules():
            if layers is not None and name not in layers:
                continue
            module_params = {}
            for pname, param in module.named_parameters(recurse=False):
                module_params[pname] = param.detach().cpu().numpy()
            if module_params:
                params[name] = module_params
        return params

    def list_layers(self) -> list:
        layers = []
        for name, module in self._model.named_modules():
            if name == "":
                continue
            param_count = sum(p.numel() for p in module.parameters(recurse=False))
            layers.append({
                "name": name,
                "type": type(module).__name__,
                "parameter_count": param_count,
            })
        return layers

    # --- HuggingFace-specific ---

    def tokenize(self, text: str) -> dict:
        """Tokenize text and return token IDs and metadata."""
        if self._tokenizer is None:
            raise ValueError("No tokenizer attached")
        encoded = self._tokenizer(text, return_tensors="pt")
        tokens = self._tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
        return {
            "input_ids": encoded["input_ids"][0].tolist(),
            "tokens": tokens,
            "num_tokens": len(tokens),
        }

    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs) -> str:
        """Generate text (for causal LM models)."""
        import torch
        if self._tokenizer is None:
            raise ValueError("No tokenizer attached")

        encoded = self._tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(self._device) for k, v in encoded.items()}

        # Check if model supports generate()
        if not hasattr(self._model, 'generate'):
            raise NotImplementedError("This model doesn't support text generation")

        with torch.no_grad():
            output_ids = self._model.generate(
                **encoded, max_new_tokens=max_new_tokens, **kwargs
            )

        return self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
