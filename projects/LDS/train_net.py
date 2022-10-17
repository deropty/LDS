#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock, guan'an wang
@contact: sherlockliao01@gmail.com, guan.wang0706@gmail.com
"""

import sys
import torch
from torch import nn

sys.path.append('.')
from fastreid.config import get_cfg
from fastreid.engine import default_argument_parser, default_setup, DefaultTrainer, launch
from fastreid.utils.checkpoint import Checkpointer, LDSCheckpointer

from lds import add_ldsreid_config, add_shufflenet_config, LDSTrainer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_shufflenet_config(cfg)
    add_ldsreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        trainer = LDSTrainer(cfg)
        model = trainer.build_model(cfg)
        LDSCheckpointer(model, cfg.OUTPUT_DIR, cfg.MODEL.NUM_MODEL).load(cfg.MODEL.WEIGHTS)
        res = DefaultTrainer.test(cfg, model)
        return res

    trainer = LDSTrainer(cfg)

    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--kd", action="store_true", help="kd training with teacher model guided")
    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )