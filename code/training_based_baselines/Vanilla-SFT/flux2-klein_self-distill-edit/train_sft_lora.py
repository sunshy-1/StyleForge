"""Vanilla SFT-LoRA baseline trained directly on a step-distilled model.

This is the Table-1 LoRA-setting Vanilla SFT baseline: the FLUX.2-klein
distilled model is trained directly with a LoRA adapter and the standard
flow-matching/SFT target. It is not the paper's "SFT + LoRA on distilled"
transfer baseline, where the LoRA is associated with the original multi-step
model before being attached to a distilled model for inference.
"""

import json
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import Flux2KleinPipeline
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


METHOD_ROOT = Path(__file__).resolve().parents[2]
DOPSD_IMPL_DIR = Path(
    os.environ.get(
        "DOPSD_PROJECT_DIR",
        METHOD_ROOT / "D-OPSD" / "flux2-klein_self-distill-edit",
    )
).expanduser().resolve()
if not DOPSD_IMPL_DIR.exists():
    raise FileNotFoundError(f"Cannot find D-OPSD helper directory: {DOPSD_IMPL_DIR}")
sys.path.insert(0, str(DOPSD_IMPL_DIR))

from arguments import parse_args  # noqa: E402
from dataset import AspectBatchSampler, CustomDataLoader, TextImageDataset, parse_ratios  # noqa: E402
from dataset_validate import TextPromptDataset  # noqa: E402
from local_paths import resolve_existing_path  # noqa: E402
from train_dopsd import (  # noqa: E402
    _encode_prompt,
    array2grid,
    compute_empirical_mu,
    create_logger,
    get_active_deepspeed_plugin,
    prepare_batch_image_latents,
)
from utils import create_generator  # noqa: E402


def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def init_single_lora_transformer(transformer, lora_rank, lora_alpha, target_modules, adapter_name="sft"):
    for param in transformer.parameters():
        param.requires_grad = False

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer = get_peft_model(transformer, lora_config, adapter_name=adapter_name)
    transformer.set_adapter(adapter_name)

    for name, param in transformer.named_parameters():
        param.requires_grad = adapter_name in name

    return transformer


def sample_base_images(pipeline, prompts, height, width, num_steps, device, dtype):
    pipeline.vae.to(device, dtype=dtype)
    generator = create_generator(prompts, 2026)
    with torch.no_grad():
        with pipeline.transformer.disable_adapter():
            images = pipeline(
                prompt=prompts,
                height=height,
                width=width,
                num_inference_steps=num_steps,
                guidance_scale=0.0 if num_steps < 10 else 4.0,
                generator=generator,
                output_type="pt",
            )[0]
    return torch.nn.functional.interpolate(
        images,
        size=(height // 2, width // 2),
        mode="bicubic",
        align_corners=False,
    )


def sample_lora_images(pipeline, transformer, prompts, height, width, num_steps, device, dtype):
    pipeline.vae.to(device, dtype=dtype)
    generator = create_generator(prompts, 2026)
    transformer.set_adapter("sft")
    with torch.no_grad():
        images = pipeline(
            prompt=prompts,
            height=height,
            width=width,
            num_inference_steps=num_steps,
            guidance_scale=0.0 if num_steps < 10 else 4.0,
            generator=generator,
            output_type="pt",
        )[0]
    return torch.nn.functional.interpolate(
        images,
        size=(height // 2, width // 2),
        mode="bicubic",
        align_corners=False,
    )


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=logging_dir,
    )

    zero2_plugin = DeepSpeedPlugin(hf_ds_config=args.deepspeed_config)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_config=accelerator_project_config,
        deepspeed_plugins={"z2_a": zero2_plugin},
    )

    os.makedirs(args.output_dir, exist_ok=True)
    save_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_dir = os.path.join(save_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if accelerator.is_main_process:
        with open(os.path.join(save_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=4)
        run_logger = create_logger(save_dir)
        run_logger.info(f"Experiment directory created at {save_dir}")

    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model,
        low_cpu_mem_usage=False,
    )
    num_channels_latents = pipeline.transformer.config.in_channels // 4

    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.transformer.requires_grad_(False)
    tokenizer = pipeline.tokenizer
    pipeline.set_progress_bar_config(disable=True)

    if args.use_lora <= 1:
        raise NotImplementedError("This baseline is LoRA-only. Set --use-lora to > 1.")

    target_modules = [
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
    ]
    pipeline.transformer = init_single_lora_transformer(
        transformer=pipeline.transformer,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        adapter_name="sft",
    )

    if args.vae_dtype == "fp32":
        vae_dtype = torch.float32
    else:
        vae_dtype = inference_dtype
    pipeline.vae.to(accelerator.device, dtype=vae_dtype)
    pipeline.vae.enable_slicing()
    pipeline.text_encoder.to(accelerator.device, dtype=inference_dtype)

    gen_model = pipeline.transformer
    gen_model_trainable_parameters = [param for param in gen_model.parameters() if param.requires_grad]
    if args.enable_gc:
        gen_model.enable_gradient_checkpointing()

    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise ImportError("Install bitsandbytes to use --use-8bit-adam.") from exc
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer_gen = optimizer_cls(
        gen_model_trainable_parameters,
        lr=args.learning_rate_gen,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    all_ratios = [
        "1024x1024 ( 1:1 index_0 )",
        "1152x896 ( 9:7 index_1 )",
        "896x1152 ( 7:9 index_2 )",
        "1152x864 ( 4:3 index_3 )",
        "864x1152 ( 3:4 index_4 )",
        "1248x832 ( 3:2 index_5 )",
        "832x1248 ( 2:3 index_6 )",
        "1280x720 ( 16:9 index_7 )",
        "720x1280 ( 9:16 index_8 )",
        "1344x576 ( 21:9 index_9 )",
        "576x1344 ( 9:21 index_10 )",
    ]
    prompt_keys = [
        "short_en",
        "detailed_en",
        "short_zh",
        "detailed_zh",
        "medium_zh",
        "medium_en",
        "user_prompt_en",
        "user_prompt_zh",
    ]
    test_prompt_keys = ["short_en", "short_zh", "medium_zh", "medium_en", "user_prompt_en", "user_prompt_zh"]
    select_ratio = all_ratios

    dataset_root = DOPSD_IMPL_DIR
    train_jsonl_path = resolve_existing_path(args.data_path_train_jsonl, dataset_root)
    test_jsonl_path = resolve_existing_path(args.data_path_test_jsonl, dataset_root)

    test_h, test_w = 1024, 1024
    if "1024x1024 ( 1:1 index_0 )" not in select_ratio:
        test_ratio = select_ratio[0]
        test_w = int(test_ratio.split("x")[0])
        test_h = int(test_ratio.split("x")[1].split(" ")[0])

    train_dataset = TextImageDataset(
        str(train_jsonl_path),
        target_resolutions=parse_ratios(select_ratio),
        data_root=dataset_root,
    )
    train_sampler = AspectBatchSampler(
        buckets=train_dataset.buckets,
        batch_size=args.batch_size,
        target_resolutions=parse_ratios(select_ratio),
        prompt_keys=prompt_keys,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        shuffle=True,
    )
    train_dataloader = CustomDataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    num_test_samples = args.batch_size_test * accelerator.num_processes
    test_dataset = TextPromptDataset(
        str(test_jsonl_path),
        prompt_keys=test_prompt_keys,
        num_prompts=num_test_samples,
        have_gt=True,
        data_root=dataset_root,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )

    if accelerator.is_main_process:
        run_logger.info(f"Dataset contains {len(train_dataset)} samples")
        run_logger.info(
            "Total batch size: "
            f"{args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps}"
        )
        run_logger.info(
            "Total trainable parameters in gen_model: "
            f"{sum(param.numel() for param in gen_model.parameters() if param.requires_grad)}"
        )
        log_gen = os.path.join(save_dir, "loss_log", "loss_gen_log.jsonl")
        os.makedirs(os.path.dirname(log_gen), exist_ok=True)
        with open(log_gen, "w") as f:
            f.write("loss for Vanilla SFT-LoRA direct-distilled baseline\n")

    assert get_active_deepspeed_plugin(accelerator.state) is zero2_plugin
    gen_model, optimizer_gen, test_dataloader = accelerator.prepare(
        gen_model,
        optimizer_gen,
        test_dataloader,
    )

    global_step = 0
    if accelerator.is_main_process:
        run_logger.info(f"Starting training experiment: {args.exp_name}")

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    test_prompts, _ = next(iter(test_dataloader))
    with torch.no_grad():
        base_images = sample_base_images(
            pipeline=pipeline,
            prompts=test_prompts,
            height=test_h,
            width=test_w,
            num_steps=args.num_training_steps,
            device=accelerator.device,
            dtype=vae_dtype,
        )
        accelerator.wait_for_everyone()
        out_samples = accelerator.gather(base_images.to(torch.float32))
        if accelerator.is_main_process:
            sample_dir = os.path.join(save_dir, "samples")
            os.makedirs(sample_dir, exist_ok=True)
            Image.fromarray(array2grid(out_samples)).save(os.path.join(sample_dir, "samples_original.png"))
            run_logger.info(f"Saved original sample images to {sample_dir}/samples_original.png")

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        for batch in train_dataloader:
            with accelerator.accumulate(gen_model):
                images = batch["pixel_values"].to(device=accelerator.device, dtype=vae_dtype)
                prompts = batch["prompts"]
                train_dtype = inference_dtype
                bsz, _, h, w = images.shape

                seqlen = (h * w) // (16 * 16)
                mu = compute_empirical_mu(seqlen, args.num_training_steps)
                pipeline.scheduler.set_timesteps(
                    args.num_training_steps,
                    device=accelerator.device,
                    mu=mu,
                )
                timesteps = pipeline.scheduler.timesteps

                with torch.no_grad():
                    with accelerator.autocast():
                        prompt_embeds, txt_ids = _encode_prompt(
                            pipeline.text_encoder,
                            tokenizer,
                            prompts,
                            max_sequence_length=512,
                            device=accelerator.device,
                        )
                        image_latents, _ = prepare_batch_image_latents(
                            pipeline=pipeline,
                            images=images,
                            device=accelerator.device,
                            vae_dtype=vae_dtype,
                            latent_dtype=train_dtype,
                            generator=None,
                        )
                        noise_latents, latent_ids = pipeline.prepare_latents(
                            batch_size=bsz,
                            num_latents_channels=num_channels_latents,
                            height=h,
                            width=w,
                            dtype=train_dtype,
                            device=accelerator.device,
                            generator=None,
                            latents=None,
                        )

                        timestep_indices = torch.randint(
                            low=0,
                            high=len(timesteps),
                            size=(bsz,),
                            device=accelerator.device,
                        )
                        t = timesteps[timestep_indices].to(device=accelerator.device, dtype=train_dtype) / 1000
                        t_view = t.reshape(bsz, 1, 1)
                        noisy_latents = (1.0 - t_view) * image_latents + t_view * noise_latents
                        target_velocity = noise_latents - image_latents

                with accelerator.autocast():
                    gen_model.set_adapter("sft")
                    model_pred = gen_model(
                        hidden_states=noisy_latents,
                        timestep=t,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=txt_ids,
                        img_ids=latent_ids,
                        return_dict=False,
                    )[0]
                    model_pred = model_pred[:, :noisy_latents.size(1)]
                    loss_sft = F.mse_loss(model_pred.float(), target_velocity.float(), reduction="mean")

                accelerator.backward(loss_sft)

                grad_norm = None
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(gen_model.parameters(), args.max_grad_norm)

                optimizer_gen.step()
                optimizer_gen.zero_grad(set_to_none=True)

                if accelerator.sync_gradients:
                    global_step += 1
                    progress_bar.update(1)

                    logs = {
                        "loss_sft": accelerator.gather(loss_sft.detach()).mean().item(),
                        "glo_s": global_step,
                        "epoch": epoch,
                        "t_mean": accelerator.gather(t.detach()).float().mean().item(),
                        "grad_n": float(grad_norm) if grad_norm is not None else 0.0,
                    }
                    accelerator.log(logs, step=global_step)
                    progress_bar.set_postfix(**logs)

                    if accelerator.is_main_process:
                        with open(log_gen, "a") as f_log_gen:
                            f_log_gen.write(f"{json.dumps(logs)}\n")

                    if global_step % args.checkpoint_steps == 0 or global_step == args.max_train_steps:
                        if accelerator.is_main_process:
                            lora_dir = os.path.join(checkpoint_dir, f"lora_gen_step_{global_step}")
                            os.makedirs(lora_dir, exist_ok=True)
                            unwrap_model(gen_model, accelerator).save_pretrained(lora_dir)

                    if global_step % args.sample_steps == 0 or global_step == args.max_train_steps:
                        with torch.no_grad():
                            images_s = sample_lora_images(
                                pipeline=pipeline,
                                transformer=gen_model,
                                prompts=test_prompts,
                                height=test_h,
                                width=test_w,
                                num_steps=args.num_training_steps,
                                device=accelerator.device,
                                dtype=vae_dtype,
                            )
                            accelerator.wait_for_everyone()
                            out_samples = accelerator.gather(images_s.to(torch.float32))
                            if accelerator.is_main_process:
                                sample_dir = os.path.join(save_dir, "samples")
                                os.makedirs(sample_dir, exist_ok=True)
                                Image.fromarray(array2grid(out_samples)).save(
                                    os.path.join(sample_dir, f"samples_step_{global_step}_sft.png")
                                )
                                run_logger.info(f"Saved sample images to {sample_dir}")

            if global_step >= args.max_train_steps:
                break
        if global_step >= args.max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        run_logger.info("Training completed.")
    accelerator.end_training()


if __name__ == "__main__":
    main(parse_args())
