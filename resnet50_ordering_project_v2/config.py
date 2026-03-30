import argparse

def build_parser():
    parser = argparse.ArgumentParser("ResNet50 Ordering Classification v3 DDP")

    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--model-type", type=str, default="ordered", choices=["baseline", "ordered"])
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--zero-init-residual", action="store_true")
    parser.add_argument("--drop-rate", type=float, default=0.0)

    parser.add_argument("--ordering-mode", type=str, default="identity")
    parser.add_argument("--insert-stages", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--ordering-provider", type=str, default="")
    parser.add_argument("--ordering-factory", type=str, default="build_ordering_module")
    parser.add_argument("--ordering-kwargs", type=str, default="{}")

    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--base-lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--betas", nargs=2, type=float, default=[0.9, 0.999])
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--scheduler", type=str, default="cosine_warmup", choices=["cosine_warmup", "step"])
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=5.0)

    parser.add_argument("--randaugment-m", type=int, default=9)
    parser.add_argument("--randaugment-n", type=int, default=2)
    parser.add_argument("--mixup-alpha", type=float, default=0.8)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mix-prob", type=float, default=1.0)
    parser.add_argument("--switch-prob", type=float, default=0.5)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--amp-dtype", type=str, default="bf16", choices=["bf16", "fp16"])

    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--eval-only", action="store_true")

    parser.add_argument("--profile-batch-size", type=int, default=64)
    parser.add_argument("--profile-warmup", type=int, default=50)
    parser.add_argument("--profile-iters", type=int, default=200)

    return parser

def get_args():
    return build_parser().parse_args()
