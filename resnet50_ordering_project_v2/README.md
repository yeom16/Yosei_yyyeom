# ResNet-50 Ordering Classification Framework v3 (DDP)

이 프로젝트는 **ResNet-50 backbone + 외부 ordering 모듈 주입 방식**으로
ImageNet 분류 실험을 수행하기 위한 **4 GPU DDP 학습 지원 버전**입니다.

## 핵심 특징
- ResNet-50 backbone 고정
- ordering 방식만 외부 provider로 교체 가능
- `torchrun --nproc_per_node=4` 기반 **DDP 분산 학습 지원**
- AMP 지원 (`bf16`, `fp16`)
- ACC@1 / ACC@5 / Throughput / Latency / Peak Memory 저장
- Warmup + Cosine Scheduler
- RandAugment + Mixup + CutMix
- 다중 seed 실험 스크립트 포함

---

## 1. 프로젝트 구조
```bash
resnet50_ordering_project_v3_ddp/
├─ main.py
├─ config.py
├─ datasets.py
├─ engine.py
├─ losses.py
├─ metrics.py
├─ profiler.py
├─ schedulers.py
├─ utils.py
├─ models/
│  ├─ __init__.py
│  ├─ order_interface.py
│  ├─ backbone_resnet50.py
│  └─ resnet50_ordered_classifier.py
└─ scripts/
   ├─ run_baseline_ddp.sh
   ├─ run_bidirectional_ddp.sh
   └─ run_multi_seed_ddp.sh
```

---

## 2. DDP 학습 실행 예시

### baseline, 4 GPU
```bash
cd /home/yeom10/workspace/models/resnet50_ordering_project_v3_ddp

torchrun --nproc_per_node=4 main.py \
  --data-path /home/yeom10/workspace/data/imagenet_fixed \
  --output-dir /home/yeom10/workspace/ckpts/ResNet50_Ordering/baseline_v3_ddp \
  --model-type baseline \
  --pretrained-backbone \
  --epochs 300 \
  --warmup-epochs 20 \
  --batch-size 16 \
  --num-workers 8 \
  --base-lr 1e-3 \
  --weight-decay 0.05 \
  --label-smoothing 0.1 \
  --grad-clip 5.0 \
  --use-amp \
  --amp-dtype bf16
```

### bidirectional ordering, 4 GPU
```bash
cd /home/yeom10/workspace/models/resnet50_ordering_project_v3_ddp

torchrun --nproc_per_node=4 main.py \
  --data-path /home/yeom10/workspace/data/imagenet_fixed \
  --output-dir /home/yeom10/workspace/ckpts/ResNet50_Ordering/bidirectional_v3_ddp \
  --model-type ordered \
  --ordering-provider /home/yeom10/workspace/models/resnet50_ordering_project/bidirectional_ordering.py \
  --ordering-factory build_ordering_module \
  --ordering-mode bidirectional \
  --insert-stages 1 2 3 4 \
  --pretrained-backbone \
  --epochs 300 \
  --warmup-epochs 20 \
  --batch-size 16 \
  --num-workers 8 \
  --base-lr 1e-3 \
  --weight-decay 0.05 \
  --label-smoothing 0.1 \
  --grad-clip 5.0 \
  --use-amp \
  --amp-dtype bf16
```

> 위 예시에서 `--batch-size 16`은 **GPU당 batch size**입니다.  
> 4 GPU 기준 global batch size는 `16 x 4 = 64`가 됩니다.

---

## 3. ordering 연결 방식
외부 provider 파일에 아래 factory 함수만 있으면 됩니다.

```python
def build_ordering_module(channels: int, stage_idx: int, ordering_mode: str, **kwargs):
    ...
```

예:
- `/home/yeom10/workspace/models/resnet50_ordering_project/bidirectional_ordering.py`

---

## 4. DDP 반영 내용
이번 버전은 아래 항목이 추가되었습니다.

- `torch.distributed.init_process_group`
- `DistributedSampler`
- `DistributedDataParallel(DDP)`
- main process(rank 0)만 checkpoint / log / json / csv 저장
- validation metric을 모든 GPU에서 평균 집계
- profiling은 main process(rank 0)에서만 수행

---

## 5. 저장 결과
출력 폴더에 저장되는 파일:

- `config.json`
- `train_log.csv`
- `latest.pth`
- `best.pth`
- `summary.csv`
- `final_result.json`

저장 지표:
- `acc1`
- `acc5`
- `throughput_img_per_s`
- `latency_ms_per_img`
- `peak_memory_mb`

---

## 6. 권장 사항
- global batch를 64로 유지하려면 `--batch-size 16` 사용
- `bf16`가 잘 되는 GPU면 `--amp-dtype bf16`
- `fp16`이 더 안정적이면 `--amp-dtype fp16`
- 다른 ordering 실험과 비교할 때는
  - epoch
  - warmup
  - batch size
  - optimizer
  - seed
  - profiling 설정
  을 동일하게 유지하는 것이 좋습니다.
