"""scripts/phase9/prune_channels.py CLI: new --finetune-* flags parse with sane defaults."""
from scripts.phase9.prune_channels import build_parser


def test_finetune_epochs_defaults_to_zero_off():
    args = build_parser().parse_args(["--model", "alexnet_bottleneck"])
    assert args.finetune_epochs == 0


def test_finetune_flags_are_parsed():
    args = build_parser().parse_args([
        "--model", "alexnet_bottleneck", "--ratio", "0.4",
        "--finetune-epochs", "200", "--finetune-lr", "5e-5", "--finetune-patience", "10",
    ])
    assert args.finetune_epochs == 200
    assert args.finetune_lr == 5e-5
    assert args.finetune_patience == 10
