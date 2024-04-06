import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Any, List

import torch
import torch.nn as nn


_CONFIG = None


def get_config():
    global _CONFIG
    assert _CONFIG is not None
    return _CONFIG


def set_config(args):
    global _CONFIG
    assert _CONFIG is None
    _CONFIG = args


def _parse_args_from_cli():
    parser = argparse.ArgumentParser()
    _add_distributed_args(parser)
    _add_model_args(parser)
    _add_data_args(parser)
    _add_trainer_args(parser)
    args = parser.parse_args()
    return args


def _add_distributed_args(parser):
    group = parser.add_argument_group("distributed")
    group.add_argument("--mesh", nargs="+")
    group.add_argument("--mesh-dim-names", nargs="+")
    group.add_argument("--device", type=str, default="cpu")
    group.add_argument("--consumed-samples", type=int, default=0)


def _add_model_args(parser):
    group = parser.add_argument_group("model")
    group.add_argument("--vocab-size", type=int, default=50272)
    group.add_argument("--max-seq-len", type=int, default=32)
    group.add_argument("--nlayers", type=int, default=24)
    group.add_argument("--model-dim", type=int, default=1024)
    group.add_argument("--nheads", type=int, default=16)
    group.add_argument("--mlp-expansion-scale", type=int, default=4)
    group.add_argument("--use-bias", action="store_true")
    group.add_argument("--attention-dropout", type=float, default=0.)
    group.add_argument("--mha-dropout", type=float, default=0.)
    group.add_argument("--mlp-dropout", type=float, default=0.)
    group.add_argument("--dtype", type=str, default="fp32")


def _add_tokenizer_args(parser):
    group = parser.add_argument_group("tokenizer")
    group.add_argument("--bos-id", type=int, default=0)
    group.add_argument("--eos-id", type=int, default=1)
    group.add_argument("--pad-id", type=int, default=2)


def _add_data_args(parser):
    group = parser.add_argument_group("data")
    group.add_argument("--num-train-samples", type=int, default=1000)
    group.add_argument("--num-workers", type=int, default=0)
    group.add_argument("--no-pinned-memory", action="store_true")
    group.add_argument("--no-drop-last", action="store_true")


def _add_trainer_args(parser):
    group = parser.add_argument_group("trainer")
    group.add_argument("--overfit", action="store_true")
    group.add_argument("--seed", type=int, default=0)
    group.add_argument("--lr", type=float, default=3e-4)
    group.add_argument("--beta1", type=float, default=0.9)
    group.add_argument("--beta2", type=float, default=0.999)
    group.add_argument("--optim-eps", type=float, default=1e-8)
    group.add_argument("--weight-decay", type=float, default=0.)
    group.add_argument("--use-fused-adam", action="store_true")
    group.add_argument("--compile", action="store_true")
    group.add_argument("--constant-lr-schedule-factor", type=float, default=1.)
    group.add_argument("--train-steps", type=int, default=100)
    group.add_argument("--val-steps", type=int, default=100)
    group.add_argument("--test-steps", type=int, default=100)
    group.add_argument("--train-steps-per-val", type=int, default=50)
    group.add_argument("--train-steps-per-checkpoint", type=int, default=10000)
    group.add_argument("--checkpoint-dir", type=str, default=".")
    group.add_argument("--checkpoint-prefix", type=str, default="DEFAULT")
    group.add_argument("--checkpoint-format", type=str, default="pt")
    group.add_argument("--batch-size", type=int, default=8)
    group.add_argument("--optim", type=str, default="torch_adam")
    group.add_argument("--lr-scheduler", type=str, default="constant")
    group.add_argument("--loss-log-interval", type=int, default=5)


def _validate_args(args):
    torch.manual_seed(args.seed)

    if not args.mesh:
        args.mesh = (1,)
    if not args.mesh_dim_names:
        args.mesh_dim_names = ("default",)

    args.mesh = list(map(int, args.mesh))
    assert len(args.mesh) == len(args.mesh_dim_names)
    assert args.device in ["cuda", "cpu"]

    str_dtype_to_dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    args.dtype = str_dtype_to_dtype[args.dtype]

    return args


def initialize_global_config():
    args = _parse_args_from_cli()
    args = _validate_args(args)
    set_config(args)
