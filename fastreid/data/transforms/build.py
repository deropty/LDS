# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torchvision.transforms as T

from .transforms import *
from .autoaugment import AutoAugment


def build_transforms(cfg, is_train=True):
    res = []

    if is_train:
        size_train = cfg.INPUT.SIZE_TRAIN

        # augmix augmentation
        do_augmix = cfg.INPUT.DO_AUGMIX
        augmix_prob = cfg.INPUT.AUGMIX_PROB

        # auto augmentation
        do_autoaug = cfg.INPUT.DO_AUTOAUG
        autoaug_prob = cfg.INPUT.AUTOAUG_PROB

        # horizontal filp
        do_flip = cfg.INPUT.DO_FLIP
        flip_prob = cfg.INPUT.FLIP_PROB

        # padding
        do_pad = cfg.INPUT.DO_PAD
        padding = cfg.INPUT.PADDING
        padding_mode = cfg.INPUT.PADDING_MODE

        # color jitter
        do_cj = cfg.INPUT.CJ.ENABLED
        cj_prob = cfg.INPUT.CJ.PROB
        cj_brightness = cfg.INPUT.CJ.BRIGHTNESS
        cj_contrast = cfg.INPUT.CJ.CONTRAST
        cj_saturation = cfg.INPUT.CJ.SATURATION
        cj_hue = cfg.INPUT.CJ.HUE

        # random erasing
        do_rea = cfg.INPUT.REA.ENABLED
        rea_prob = cfg.INPUT.REA.PROB
        rea_value = cfg.INPUT.REA.VALUE

        # random patch
        do_rpt = cfg.INPUT.RPT.ENABLED
        rpt_prob = cfg.INPUT.RPT.PROB

        # random scale
        do_rsl = cfg.INPUT.RSL.ENABLED
        rsl_type = cfg.INPUT.RSL.TYPE
        rsl_prob = cfg.INPUT.RSL.PROB
        rsl_scale = cfg.INPUT.RSL.SCALE
        rsl_threshold = cfg.INPUT.RSL.THRESHOLD

        if do_autoaug:                                                                          # 0
            res.append(T.RandomApply([AutoAugment()], p=autoaug_prob))
        res.append(T.Resize(size_train, interpolation=3))                                       # 1
        if do_flip:
            res.append(T.RandomHorizontalFlip(p=flip_prob))                                     # 2
        if do_pad:
            res.extend([T.Pad(padding, padding_mode=padding_mode), T.RandomCrop(size_train)])   # 3 4
        if do_cj:
            res.append(T.RandomApply([T.ColorJitter(cj_brightness, cj_contrast, cj_saturation, cj_hue)], p=cj_prob))
        if do_augmix:
            res.append(T.RandomApply([AugMix()], p=augmix_prob))
        if do_rsl:
            res.append(eval(rsl_type)(prob_happen=rsl_prob, scale=rsl_scale, threshold=rsl_threshold))

        res.append(ToTensor())                                                                  # 5
        if do_rea:
            res.append(T.RandomErasing(p=rea_prob, value=rea_value))                            # 6
        if do_rpt:
            res.append(RandomPatch(prob_happen=rpt_prob))
    else:
        size_test = cfg.INPUT.SIZE_TEST
        res.append(T.Resize(size_test, interpolation=3))
        res.append(ToTensor())
    return T.Compose(res)
