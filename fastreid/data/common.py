# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torchvision.transforms as T
from fastreid.data.transforms.transforms import *
import torch

from torch.utils.data import Dataset
from .data_utils import read_image


class CommDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, img_items, transform=None, relabel=True):
        self.img_items = img_items
        self.transform = transform
        self.relabel = relabel

        pid_set = set()
        cam_set = set()
        for i in img_items:
            pid_set.add(i[1])
            cam_set.add(i[2])

        self.pids = sorted(list(pid_set))
        self.cams = sorted(list(cam_set))
        if relabel:
            self.pid_dict = dict([(p, i) for i, p in enumerate(self.pids)])
            self.cam_dict = dict([(p, i) for i, p in enumerate(self.cams)])

    def __len__(self):
        return len(self.img_items)

    def __getitem__(self, index):
        img_path, pid, camid = self.img_items[index]
        img = read_image(img_path)
        if self.transform is not None: img = self.transform(img)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
        }

    @property
    def num_classes(self):
        return len(self.pids)

    @property
    def num_cameras(self):
        return len(self.cams)

class LDSDataset(CommDataset):
    def __init__(self, img_items, transform=None, relabel=True, cfg=None):
        self.cfg = cfg
        self.num_models = self.cfg.MODEL.NUM_MODEL
        self.target_transform = []
        for i in self.cfg.DML.TARGET_TRANSFORM:
            if i == "RandomErasing":
                self.target_transform.append("T."+i)
            else:
                self.target_transform.append(i)
        self.num_transforms = len(self.target_transform)
        super().__init__(img_items, transform, relabel)

    def __getitem__(self, index):
        img_path, pid, camid = self.img_items[index]
        img = read_image(img_path)
        if self.transform is not None:
            transform_modes = []
            if self.num_models == 1:
                transform_modes.append(self.transform.transforms)
            elif self.num_models >= 2:
                if self.cfg.DML.VARIATION == "dml":
                    if self.cfg.DML.SYN_TYPE == "heterologous":
                        pass
                    elif self.cfg.DML.SYN_TYPE == "homologous":
                        transform_modes = [self.transform.transforms for i in range(self.num_models)]
                elif self.cfg.DML.VARIATION == "lds":
                    pos = [None] * self.num_transforms + [None]
                    for i in range(len(self.transform.transforms)):
                        if self.transform.transforms[i].__class__.__name__ == 'ToTensor':
                            pos[self.num_transforms] = i
                        for j,k in enumerate(self.target_transform):
                            if isinstance(self.transform.transforms[i], eval(k)):
                                pos[j] = i
                    target_bef = self.transform.transforms[:pos[0]]
                    totensor_bef = self.transform.transforms[:pos[-1]]    # before target transform
                    totensor_tar = self.transform.transforms[pos[-1]:pos[-1]+1] # target transform
                    if self.num_models == 2:
                        assert len(self.target_transform) == 1, "number of target transforms is not match"
                        if self.target_transform[0] == "T.RandomErasing":
                            if self.cfg.DML.SYN_TYPE == "heterologous":
                                img_1 = T.Compose(totensor_bef+totensor_tar)(img)
                                img_2 = T.Compose([self.transform.transforms[pos[0]]])(img_1)
                                img = torch.stack([img_1, img_2], 0)
                            elif self.cfg.DML.SYN_TYPE == "homologous":
                                transform_model1 = totensor_bef + totensor_tar
                                transform_model2 = totensor_bef + totensor_tar + [self.transform.transforms[pos[0]]]
                                transform_modes = [transform_model1, transform_model2]
                        elif self.target_transform[0][:11] == "RandomScale":
                            if self.cfg.DML.SYN_TYPE == "heterologous":
                                img_bef = T.Compose(target_bef)(img)
                                img_1 = T.Compose(totensor_tar)(img_bef)
                                img_2 = T.Compose([self.transform.transforms[pos[0]]]+totensor_tar)(img_bef)
                                img = torch.stack([img_1, img_2], 0)
                            elif self.cfg.DML.SYN_TYPE == "homologous":
                                transform_model1 = target_bef + totensor_tar
                                transform_model2 = target_bef + [self.transform.transforms[pos[0]]] + totensor_tar
                                transform_modes = [transform_model1, transform_model2]
                    elif self.num_models == 3:
                        assert len(self.target_transform) == 2, "number of target transforms is not match"
                        assert self.target_transform[0] == "T.RandomErasing",   "config error"
                        assert self.target_transform[1][:11] == "RandomScale",       "config error"
                        target_bef = self.transform.transforms[:pos[1]]
                        if self.cfg.DML.SYN_TYPE == "heterologous":
                            img_bef = T.Compose(target_bef)(img)
                            img_1 = T.Compose(totensor_tar)(img_bef)
                            img_2 = T.Compose(totensor_tar + [self.transform.transforms[pos[0]]])(img_bef)
                            img_3 = T.Compose([self.transform.transforms[pos[1]]] + totensor_tar)(img_bef)
                            img = torch.stack([img_1, img_2, img_3], 0)
                        elif self.cfg.DML.SYN_TYPE == "homologous":
                            transform_model1 = target_bef + totensor_tar
                            transform_model2 = target_bef + totensor_tar + [self.transform.transforms[pos[0]]]
                            transform_model3 = target_bef + [self.transform.transforms[pos[1]]] + totensor_tar
                            transform_modes = [transform_model1, transform_model2, transform_model3]
                elif self.cfg.DML.VARIATION == "ams":
                    pos = [None] * self.num_transforms + [None]
                    for i in range(len(self.transform.transforms)):
                        if self.transform.transforms[i].__class__.__name__ == 'ToTensor':
                            pos[self.num_transforms] = i
                        for j, k in enumerate(self.target_transform):
                            if isinstance(self.transform.transforms[i], eval(k)):
                                pos[j] = i
                    target_bef = self.transform.transforms[:pos[0]]
                    assert len(self.target_transform) == 1, "number of target transforms is not match"
                    assert self.target_transform[0] == "T.RandomErasing", "config error"

                    img_bef = T.Compose(target_bef)(img)
                    img_1 = img_bef
                    img_2 = T.Compose([self.transform.transforms[pos[0]]])(img_bef)
                    img_3 = T.Compose([self.transform.transforms[pos[0]]])(img_bef)
                    img = torch.stack([img_1, img_2, img_3], 0)


            # exclude heterologous mode
            if self.num_models == 1 or self.num_models >= 2 and self.cfg.DML.VARIATION == 'dml' and self.cfg.DML.SYN_TYPE == "homologous" or \
                    self.num_models >= 2 and self.cfg.DML.VARIATION == 'lds' and self.cfg.DML.SYN_TYPE == "homologous":
                img = torch.stack([T.Compose(i)(img) for i in transform_modes], 0)
            if self.num_models >= 2 and self.cfg.DML.VARIATION == 'dml' and self.cfg.DML.SYN_TYPE == "heterologous":
                img = T.Compose(self.transform.transforms)(img)
                img = torch.stack([img for i in range(self.num_models)], 0)
        if self.relabel:
            pid = self.pid_dict[pid]
            camid = self.cam_dict[camid]
        return {
            "images": img,
            "targets": pid,
            "camids": camid,
            "img_paths": img_path,
        }