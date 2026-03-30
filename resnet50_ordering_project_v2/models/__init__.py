from .order_interface import load_external_factory, parse_ordering_kwargs
from .resnet50_ordered_classifier import ResNet50BaselineClassifier, ResNet50OrderedClassifier

def build_model(args):
    if args.model_type == "baseline":
        return ResNet50BaselineClassifier(
            num_classes=args.num_classes,
            pretrained_backbone=args.pretrained_backbone,
            zero_init_residual=args.zero_init_residual,
            drop_rate=args.drop_rate,
        )

    ordering_factory = load_external_factory(args.ordering_provider, args.ordering_factory)
    ordering_kwargs = parse_ordering_kwargs(args.ordering_kwargs)

    return ResNet50OrderedClassifier(
        num_classes=args.num_classes,
        pretrained_backbone=args.pretrained_backbone,
        zero_init_residual=args.zero_init_residual,
        insert_stages=args.insert_stages,
        ordering_factory=ordering_factory,
        ordering_mode=args.ordering_mode,
        ordering_kwargs=ordering_kwargs,
        drop_rate=args.drop_rate,
    )
