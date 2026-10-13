import copy
import torch
from peft import LoraConfig, get_peft_model

def set_adapter_trainable(model, adapter_name: str, trainable: bool):
    """
    Control whether the LoRA parameters of a specific adapter are trainable.
    """
    for name, param in model.named_parameters():
        if adapter_name in name:
            param.requires_grad = trainable

@torch.no_grad()
def copy_lora_adapter_weights(model, src_adapter: str, dst_adapter: str):
    """
    Copy LoRA weights from src_adapter to dst_adapter.
    Only LoRA parameters are copied; the base model is untouched.
    """
    named_params = dict(model.named_parameters())

    for name, param in named_params.items():
        if src_adapter not in name:
            continue

        dst_name = name.replace(src_adapter, dst_adapter)
        if dst_name in named_params:
            named_params[dst_name].data.copy_(param.data)



def init_dual_lora_transformer(
    transformer,
    lora_rank=16,
    lora_alpha=16,
    target_modules=None,
    current_adapter_name="current",
    old_adapter_name="old",
    old_init_from_current=True,
):
    """
    Attach two LoRA adapters to the same transformer:
      - current: trainable
      - old: frozen, updated via EMA

    Returns:
      transformer_with_lora
    """
    if target_modules is None:
        target_modules = [
            "feed_forward.w1"
            "feed_forward.w2",
            "feed_forward.w3",
            "attention.to_k",
            "attention.to_q",
            "attention.to_v",
            "attention.to_out.0",
        ]

    # Freeze the base model
    for p in transformer.parameters():
        p.requires_grad = False

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )

    # First create the current adapter
    transformer = get_peft_model(transformer, lora_config, adapter_name=current_adapter_name)

    # Then add the old adapter
    transformer.add_adapter(old_adapter_name, lora_config)

    # Activate current by default
    transformer.set_adapter(current_adapter_name)

    # Train only the current adapter and freeze the old adapter
    set_adapter_trainable(transformer, current_adapter_name, True)
    set_adapter_trainable(transformer, old_adapter_name, False)

    # Optionally initialize old = current
    if old_init_from_current:
        copy_lora_adapter_weights(
            transformer,
            src_adapter=current_adapter_name,
            dst_adapter=old_adapter_name,
        )

    return transformer

@torch.no_grad()
def ema_update_lora_adapter(model, src_adapter: str, dst_adapter: str, ema_decay: float = 0.999):
    """
    Apply EMA update from src_adapter to dst_adapter:
        dst = ema_decay * dst + (1 - ema_decay) * src

    Typically:
      src_adapter = "current"
      dst_adapter = "old"
    """
    named_params = dict(model.named_parameters())

    for name, src_param in named_params.items():
        if src_adapter not in name:
            continue

        dst_name = name.replace(src_adapter, dst_adapter)
        if dst_name not in named_params:
            continue

        dst_param = named_params[dst_name]
        dst_param.data.mul_(ema_decay).add_(src_param.data, alpha=1.0 - ema_decay)
