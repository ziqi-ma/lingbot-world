#!/bin/bash
IMAGE_DIR="/data/ziqi/data/worldstate/predynamic/initial_frames/cropped"
OUTPUT_DIR="outputs/right11_batch"
mkdir -p "$OUTPUT_DIR"

for img in "$IMAGE_DIR"/*.png; do
    name=$(basename "$img" .png)

    # Determine size based on aspect ratio
    dims=$(python3 -c "from PIL import Image; w,h=Image.open('$img').size; print('480*832' if h>w else '832*480')")

    echo "=== Processing: $name (size: $dims) ==="
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29501 --nproc_per_node=4 generate.py \
        --task i2v-A14B \
        --size "$dims" \
        --ckpt_dir /data/ziqi/checkpoints/lingobot \
        --image "$img" \
        --action_path examples/hyworld_converted \
        --dit_fsdp --t5_fsdp \
        --ulysses_size 4 \
        --frame_num 45 \
        --prompt "" \
        --save_file "$OUTPUT_DIR/${name}.mp4"
    echo "=== Done: $name ==="
done
echo "All done. Videos saved to $OUTPUT_DIR/"
