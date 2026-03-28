import argparse


def build_parser():
    parser = argparse.ArgumentParser("ResNet50 Ordering Classification")

    # data
    parser.add_argument("--data-path", type=str, required=True, help="ImageNet root path containing train/ and val/")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--input-size", type=int, default=224)

    # dataloader
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)

    # training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "step"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-freq", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")

    # model
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--drop-rate", type=float, default=0.0)
    parser.add_argument("--zero-init-residual", action="store_true")

    # ordering injection
    parser.add_argument("--model-type", type=str, default="ordered", choices=["baseline", "ordered"])
    parser.add_argument("--ordering-mode", type=str, default="identity")
    parser.add_argument("--insert-stages", nargs="+", type=int, default=[1, 2, 3, 4], help="Which stages receive ordering module")
    parser.add_argument("--ordering-provider", type=str, default="", help="Path to external python file that defines ordering factory")
    parser.add_argument("--ordering-factory", type=str, default="build_ordering_module", help="Factory function name inside provider file")
    parser.add_argument("--ordering-kwargs", type=str, default="{}", help='JSON string passed to external ordering factory')

    # profiling
    parser.add_argument("--profile-batch-size", type=int, default=64)
    parser.add_argument("--profile-warmup", type=int, default=30)
    parser.add_argument("--profile-iters", type=int, default=100)

    # resume / eval
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--eval-only", action="store_true")

    return parser


def get_args():
    return build_parser().parse_args()
