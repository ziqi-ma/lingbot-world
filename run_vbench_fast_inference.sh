#!/bin/bash
set -e

PYTHON=/data/ziqi/miniconda3/envs/rollout/bin/python
TORCHRUN=/data/ziqi/miniconda3/envs/rollout/bin/torchrun

CKPT=/data/ziqi/data/checkpoints/lingbot-world-base-cam
INPUT_DIR=/data/ziqi/data/wmagent/baselines/input/vbench
OUTPUT_BASE=/data/ziqi/data/wmagent/baselines/lingbot
POSE_DIR=/tmp/lingbot_vbench_poses
JOBS_FILE=/tmp/lingbot_vbench_fast_jobs.json
FRAMES=61  # 15 latents * 4 + 1

CATEGORIES=(bird dogball kangaroo personcooking personcouch dog)

declare -A POSE_ACTIONS
POSE_ACTIONS=(
    [right-up]="right-7,up-8"
    [s-w]="s-7,w-8"
    [up-right]="up-7,right-8"
    [w-left]="w-7,left-8"
    [down-left]="down-7,left-8"
    [w-right]="w-7,right-8"
)

# Generate pose files
echo "======== Generating pose files ========"
for POSE_NAME in "${!POSE_ACTIONS[@]}"; do
    ACTION="${POSE_ACTIONS[$POSE_NAME]}"
    PDIR="${POSE_DIR}/${POSE_NAME}"
    if [ ! -f "${PDIR}/poses.npy" ]; then
        $PYTHON convert_from_hyworld.py "$ACTION" --output "$PDIR"
    else
        echo "Poses already exist for ${POSE_NAME}, skipping."
    fi
done
echo "======== Pose files ready ========"

# Build jobs JSON
echo "======== Building jobs list ========"
$PYTHON - <<EOF
import json, os

categories = "${CATEGORIES[*]}".split()
pose_names = list("${!POSE_ACTIONS[*]}".split())
input_dir = "${INPUT_DIR}"
output_base = "${OUTPUT_BASE}"
pose_dir = "${POSE_DIR}"

jobs = []
for cat in categories:
    prompt_file = os.path.join(input_dir, cat, "prompt.txt")
    with open(prompt_file) as f:
        prompt = f.read().strip()
    for pose in pose_names:
        save_file = os.path.join(output_base, cat, f"{pose}.mp4")
        jobs.append({
            "image": os.path.join(input_dir, cat, "init_16x9.png"),
            "prompt": prompt,
            "action_path": os.path.join(pose_dir, pose),
            "save_file": save_file,
        })

with open("${JOBS_FILE}", "w") as f:
    json.dump(jobs, f, indent=2)
print(f"Wrote {len(jobs)} jobs to ${JOBS_FILE}")
EOF

# Create output dirs
for CATEGORY in "${CATEGORIES[@]}"; do
    mkdir -p "${OUTPUT_BASE}/${CATEGORY}"
done

# Run inference — model loads once, all jobs run in sequence
echo "======== Running inference ========"
CUDA_VISIBLE_DEVICES=0,1,2,3 $TORCHRUN --nproc_per_node=4 generate_fast_batch.py \
    --task i2v-A14B \
    --size 480*832 \
    --ckpt_dir $CKPT \
    --dit_fsdp --t5_fsdp --ulysses_size 4 \
    --frame_num $FRAMES \
    --jobs_file $JOBS_FILE

echo "All done."
