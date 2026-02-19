ACTION_PATH=examples/hyworld_converted
FRAME_NUM=$(python -c "import numpy as np; p=np.load('${ACTION_PATH}/poses.npy'); n=len(p); print(((n-1)//4)*4+1)")

# CUDA_VISIBLE_DEVICES=2,3,4,5,6 torchrun --master_port=29801 --nproc_per_node=5 generate.py \
#   --task i2v-A14B \
#   --size 480*832 \
#   --ckpt_dir /data/ziqi/checkpoints/lingobot \
#   --image /data/ziqi/data/worldstate/predynamic/kitchen3.jpeg \
#   --action_path "${ACTION_PATH}" \
#   --dit_fsdp --t5_fsdp \
#   --ulysses_size 5 \
#   --frame_num "${FRAME_NUM}" \
#   --prompt "" && \
CUDA_VISIBLE_DEVICES=2,3,4,5,6 torchrun --master_port=29801 --nproc_per_node=5 generate.py \
  --task i2v-A14B \
  --size 480*832 \
  --ckpt_dir /data/ziqi/checkpoints/lingobot \
  --image /data/ziqi/data/worldstate/predynamic/kitchen.jpeg \
  --action_path "${ACTION_PATH}" \
  --dit_fsdp --t5_fsdp \
  --ulysses_size 5 \
  --frame_num "${FRAME_NUM}" \
  --prompt "The cat goes down the chair to the left."