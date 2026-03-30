#!/bin/bash
SEEDS=(0 1 2)
for SEED in "${SEEDS[@]}"; do
  torchrun --nproc_per_node=4 main.py     --data-path /home/yeom10/workspace/data/imagenet_fixed     --output-dir /home/yeom10/workspace/ckpts/ResNet50_Ordering/bidirectional_seed${SEED}_v3_ddp     --model-type ordered     --ordering-provider /home/yeom10/workspace/models/resnet50_ordering_project/bidirectional_ordering.py     --ordering-factory build_ordering_module     --ordering-mode bidirectional     --insert-stages 1 2 3 4     --pretrained-backbone     --epochs 300     --warmup-epochs 20     --batch-size 16     --num-workers 8     --base-lr 1e-3     --weight-decay 0.05     --label-smoothing 0.1     --grad-clip 5.0     --seed ${SEED}     --use-amp     --amp-dtype bf16
done
