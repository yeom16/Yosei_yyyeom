# ResNet-50 Ordering Classification Framework

이 프로젝트는 **ordering 방식 정의 자체는 비워두고**,  
다른 사람이 만든 ordering 모듈을 **stage 단위로 꽂을 수 있게** 설계한
ImageNet classification 학습/평가 프레임워크입니다.

## 핵심 구조
- Backbone: torchvision ResNet-50
- Stage insertion: layer1, layer2, layer3, layer4 뒤에 ordering module 삽입 가능
- Head: Global Average Pooling + FC classifier
- Metrics:
  - ACC@1
  - ACC@5
  - TP (throughput, img/s)
  - Mem (peak GPU memory, MB)

## 폴더 구조
- `main.py` : 전체 실행 진입점
- `config.py` : 인자 파서
- `datasets.py` : ImageNet dataloader
- `engine.py` : train / validate
- `metrics.py` : ACC@1 / ACC@5
- `profiler.py` : TP / Mem 측정
- `utils.py` : seed, checkpoint, json/csv 저장
- `models/order_interface.py` : ordering 모듈 주입 인터페이스
- `models/backbone_resnet50.py` : ResNet-50 stage backbone
- `models/resnet50_ordered_classifier.py` : 최종 classification 모델

## ordering 모듈 연결 방법
다른 사람이 별도 파일에서 아래 함수만 구현하면 됩니다.

```python
def build_ordering_module(channels: int, stage_idx: int, ordering_mode: str, **kwargs):
    # channels: stage 출력 채널 수 (256, 512, 1024, 2048)
    # stage_idx: 1, 2, 3, 4
    # ordering_mode: 예: "bidirectional", "continuous_2d", "cross_scan" 등
    ...
```

그 후 실행 시:

```bash
python main.py   --data-path /path/to/imagenet   --ordering-provider your_ordering_file.py   --ordering-factory build_ordering_module   --ordering-mode bidirectional   --insert-stages 1 2 3 4   --pretrained-backbone
```

ordering provider를 주지 않으면 자동으로 Identity ordering이 들어갑니다.
