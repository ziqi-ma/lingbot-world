import argparse
import json
import logging
import os
import sys
import warnings

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.utils import save_video, str2bool


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Batch fast video generation: load model once, run multiple jobs."
    )
    parser.add_argument("--task", type=str, default="i2v-A14B", choices=list(WAN_CONFIGS.keys()))
    parser.add_argument("--size", type=str, default="1280*720", choices=list(SIZE_CONFIGS.keys()))
    parser.add_argument("--frame_num", type=int, default=None)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--offload_model", type=str2bool, default=None)
    parser.add_argument("--ulysses_size", type=int, default=1)
    parser.add_argument("--t5_fsdp", action="store_true", default=False)
    parser.add_argument("--t5_cpu", action="store_true", default=False)
    parser.add_argument("--dit_fsdp", action="store_true", default=False)
    parser.add_argument("--sample_shift", type=float, default=None)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--convert_model_dtype", action="store_true", default=False)
    parser.add_argument("--max_attention_size", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=3)
    parser.add_argument("--jobs_file", type=str, required=True,
                        help="JSON file with list of {image, prompt, action_path, save_file}.")
    return parser.parse_args()


def generate_batch(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank

    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)

    if args.offload_model is None:
        args.offload_model = False if world_size > 1 else True

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    else:
        assert not (args.t5_fsdp or args.dit_fsdp), "t5_fsdp and dit_fsdp require distributed."
        assert args.ulysses_size == 1, "Sequence parallel requires distributed."

    if args.ulysses_size > 1:
        assert args.ulysses_size == world_size
        init_distributed_group()

    cfg = WAN_CONFIGS[args.task]

    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.frame_num is None:
        args.frame_num = cfg.frame_num

    base_seed = args.base_seed if args.base_seed >= 0 else random.randint(0, sys.maxsize)
    if dist.is_initialized():
        seed_list = [base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(seed_list, src=0)
        base_seed = seed_list[0]

    with open(args.jobs_file) as f:
        jobs = json.load(f)

    logging.info(f"Loading WanI2VFast pipeline from {args.ckpt_dir} ...")
    wan_i2v = wan.WanI2VFast(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=device,
        rank=rank,
        t5_fsdp=args.t5_fsdp,
        dit_fsdp=args.dit_fsdp,
        use_sp=(args.ulysses_size > 1),
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )
    logging.info("Model loaded. Starting batch generation.")
    import time
    batch_start = time.time()
    job_times = []

    for i, job in enumerate(jobs):
        save_file = job["save_file"]

        if dist.is_initialized():
            skip = [os.path.exists(save_file) if rank == 0 else False]
            dist.broadcast_object_list(skip, src=0)
            should_skip = skip[0]
        else:
            should_skip = os.path.exists(save_file)

        if should_skip:
            logging.info(f"[{i+1}/{len(jobs)}] Skipping (exists): {save_file}")
            continue

        logging.info(f"[{i+1}/{len(jobs)}] Generating: {save_file}")
        job_start = time.time()
        img = Image.open(job["image"]).convert("RGB")

        video = wan_i2v.generate(
            job["prompt"],
            img,
            action_path=job["action_path"],
            chunk_size=args.chunk_size,
            max_area=MAX_AREA_CONFIGS[args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            seed=base_seed,
            offload_model=args.offload_model,
            max_attention_size=args.max_attention_size,
        )

        if rank == 0:
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            logging.info(f"Saving {save_file}")
            save_video(
                tensor=video[None],
                save_file=save_file,
                fps=cfg.sample_fps,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
            )
        del video
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()
        elapsed = time.time() - job_start
        job_times.append(elapsed)
        logging.info(f"[{i+1}/{len(jobs)}] Took {elapsed:.1f}s ({save_file})")

    total = time.time() - batch_start
    if job_times:
        avg = sum(job_times) / len(job_times)
        logging.info(f"All jobs done. Generated {len(job_times)} in {total:.1f}s (avg {avg:.1f}s/job).")
    else:
        logging.info(f"All jobs done. Total {total:.1f}s (no new jobs generated).")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    args = _parse_args()
    generate_batch(args)
