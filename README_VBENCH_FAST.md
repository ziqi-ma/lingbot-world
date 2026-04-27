# VBench Fast Inference (LingBot-World-Base + camera control)

This document describes the data format expected by `run_vbench_fast_inference.sh`,
how camera matrices are produced from HY-WorldPlay-style action strings via
`convert_from_hyworld.py`, and how to run the pipeline end-to-end.

## 1. What the script does

`run_vbench_fast_inference.sh`:

1. Generates a `poses.npy` + `intrinsics.npy` for each named camera trajectory
   by calling `convert_from_hyworld.py`.
2. Builds a JSON job list pairing every input category with every pose.
3. Runs `generate_fast_batch.py` once, loading the model a single time and
   sweeping all jobs.

## 2. Input data layout

```
INPUT_DIR=/data/ziqi/data/wmagent/baselines/input/vbench
INPUT_DIR/
  <category>/                # e.g. bird, dogball, kangaroo, personcooking, personcouch, dog
    init_16x9.png            # 832x480 first-frame conditioning image
    prompt.txt               # plain text prompt for this category
```

Categories are listed in the `CATEGORIES` array at the top of the script.

## 3. Pose / intrinsics format (per-trajectory directory)

Each named pose ends up in `${POSE_DIR}/<pose_name>/`:

```
poses.npy       float32, shape (F, 4, 4)   camera-to-world matrices, one per video frame
intrinsics.npy  float32, shape (F, 4)      [fx, fy, cx, cy] per frame
```

- `F = total_latents * 4 + 1` (matches HY-WorldPlay-New's frame count
  convention; with the script default `FRAMES=61` this means 15 latents).
- Intrinsics are baked for 832×480: `fx=502.9, fy=503.1, cx=415.8, cy=239.8`.
- Poses are absolute c2w; LingBot internally normalises against the first
  frame, so no Y-flip / coordinate conversion is applied (applying one breaks
  SLERP interpolation inside `interpolate_camera_poses`).

## 4. Getting camera matrices from action strings

`convert_from_hyworld.py` translates HY-WorldPlay action strings into
LingBot-style frame-level poses.

### Action string syntax

Comma-separated `<action>-<num_latents>` tokens. Each latent expands to
`FRAMES_PER_LATENT = 4` video frames of constant motion.

```
right-11                 # yaw-right for 11 latents (44 frames)
w-4, d-4                 # forward 4 latents, then strafe-right 4 latents
right-7, up-8            # yaw-right, then pitch-up
```

Per-frame deltas (configurable at the top of the file):

```
FORWARD_SPEED = 0.02 m/frame
YAW_SPEED     = 0.75°/frame
PITCH_SPEED   = 0.75°/frame
```

Supported actions: `w/a/s/d` (translate), `up/down/left/right` (rotate),
combined translations (`wd`, `wa`, `sd`, `sa`), combined rotations
(`rightup`, `rightdown`, `leftup`, `leftdown`), and translation+rotation
(`wright`, `wleft`, `sright`, `sleft`, `dright`, `dleft`, `aright`, `aleft`,
`wup`, `wdown`, `sup`, `sdown`).

The actual c2w trajectory is produced by
`generate_camera_trajectory_local` from
`/data/ziqi/Repos/HY-WorldPlay/hyvideo/generate_custom_trajectory.py`
(imported via `sys.path` insert) — that repo must be present at that path.

### Standalone usage

```
python convert_from_hyworld.py "right-7,up-8" --output /tmp/lingbot_vbench_poses/right-up
```

Writes `poses.npy` and `intrinsics.npy` into the output directory.

### Trajectories baked into the script

```
right-up   = right-7,up-8
s-w        = s-7,w-8
up-right   = up-7,right-8
w-left     = w-7,left-8
down-left  = down-7,left-8
w-right    = w-7,right-8
```

(15 latents each → 61 frames, matching `FRAMES=61`.)

## 5. Jobs JSON format

`generate_fast_batch.py` consumes a JSON list of jobs:

```json
[
  {
    "image":       "/.../<category>/init_16x9.png",
    "prompt":      "<prompt text>",
    "action_path": "/tmp/lingbot_vbench_poses/<pose_name>",
    "save_file":   "/.../lingbot/<category>/<pose_name>.mp4"
  }
]
```

`action_path` is the directory containing `poses.npy` and `intrinsics.npy`.
Existing `save_file`s are skipped automatically.

## 6. Running

Edit the path constants at the top of the script if needed:

```
CKPT          model checkpoint dir (lingbot-world-base-cam)
INPUT_DIR     vbench input root (one folder per category)
OUTPUT_BASE   output root (mp4s land at OUTPUT_BASE/<category>/<pose>.mp4)
POSE_DIR      where converted poses are cached
JOBS_FILE     temp jobs JSON path
FRAMES        frames per video (must match action string: latents*4+1)
```

Then:

```
bash run_vbench_fast_inference.sh
```

The script uses 4 GPUs (`CUDA_VISIBLE_DEVICES=0,1,2,3`) with FSDP + Ulysses
context-parallel size 4 at 480×832.
