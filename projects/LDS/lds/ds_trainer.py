# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from typing import Dict
import numpy as np
from apex import amp
from fastreid.engine import DefaultTrainer
from fastreid.utils.file_io import PathManager
from fastreid.modeling.meta_arch import build_model
from fastreid.solver import build_lr_scheduler, build_optimizer
from fastreid.utils.checkpoint import Checkpointer
from .config import update_model_teacher_config

import fastreid.utils.comm as comm
from fastreid.utils.events import get_event_storage

class DSTrainer(DefaultTrainer):
    """
    A knowledge distillation trainer for person reid of task.
    """

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg)

        self._data_loader_iter = iter(self.data_loader)
        self.num_models = cfg.MODEL.NUM_MODEL
        self.learning = cfg.DML.MUTUAL_MASTER_SERVANT
        self.fp16_enable = cfg.SOLVER.FP16_ENABLED

    def run_step(self):
        """
        Implement the moco training logic described above.
        """
        assert isinstance(self.model, list) and [i.training for i in self.model], \
            "[DSTrainer] base model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)

        data_time = time.perf_counter() - start

        # split data for each model
        datas = []
        for i in range(self.num_models):
            d = {}
            for key in data:
                d[key] = data[key][:, i, :] if key == 'images' else data[key]
            datas.append(d)

        # forward
        outputs = []
        loss_dicts = []
        for i in range(self.num_models):
            output= self.model[i](datas[i])
            # Compute reid loss
            if isinstance(self.model[i], DistributedDataParallel):
                loss_dict = self.model[i].module.losses(output)
            else:
                loss_dict = self.model[i].losses(output)
            outputs.append(output)
            loss_dicts.append(loss_dict)

        metrics_dict = {}
        for i in range(self.num_models): # 交互学习
            q_logits = outputs[i]["outputs"]["pred_class_logits"]
            if self.learning == "mutual":
                for j in range(self.num_models):
                    if i != j:
                        t_logits = outputs[j]["outputs"]["pred_class_logits"].detach()
                        loss_dicts[i]['loss_kl_{}vs{}'.format(i,j)] = self.distill_loss(q_logits, t_logits, t=16) / (self.num_models - 1)
            elif self.learning == "master_servant":
                if i == 0:
                    for j in range(1, self.num_models):
                        t_logits = outputs[j]["outputs"]["pred_class_logits"].detach()
                        loss_dicts[i]['loss_kl_{}vs{}'.format(i, j)] = self.distill_loss(q_logits, t_logits, t=16) / (self.num_models - 1)
                else:
                    t_logits = outputs[0]["outputs"]["pred_class_logits"].detach()
                    loss_dicts[i]['loss_kl_{}vs{}'.format(i, j)] = self.distill_loss(q_logits, t_logits, t=16)
            losses = sum(loss_dicts[i].values())

            for key in loss_dicts[i]:
                metrics_dict['model_{}_'.format(i) + key] = loss_dicts[i][key]

            """
            If you need accumulate gradients or something similar, you can
            wrap the optimizer with your custom `zero_grad()` method.
            """
            self.optimizer[i].zero_grad()
            if self.fp16_enable:
                with amp.scale_loss(losses, self.optimizer[i]) as scaled_loss:
                    scaled_loss.backward()
            else:
                losses.backward()

            """
            If you need gradient clipping/scaling or other processing, you can
            wrap the optimizer with your custom `step()` method.
            """
            self.optimizer[i].step()

        with torch.cuda.stream(torch.cuda.Stream()):
            metrics_dict = loss_dict
            self._write_metrics(metrics_dict, data_time)

    def _write_metrics(self, loss_dict: Dict[str, torch.Tensor], data_time: float):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        device = next(iter(loss_dict.values())).device

        # Use a new stream so these ops don't wait for DDP or backward
        with torch.cuda.stream(torch.cuda.Stream() if device.type == "cuda" else None):
            metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
            metrics_dict["data_time"] = data_time

            # Gather metrics among all workers for logging
            # This assumes we do DDP-style training, which is currently the only
            # supported method in detectron2.
            all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer, iters_per_epoch) -> list:
        lr_schedulers = []
        for o in optimizer:
            lr_schedulers.append(super().build_lr_scheduler(cfg, o, iters_per_epoch))
        lr_scheduler = {}
        for key in lr_schedulers[0]:
            lr_scheduler[key] = [i[key] for i in lr_schedulers]
        return lr_scheduler

    @classmethod
    def build_optimizer(cls, cfg, model) -> list:
        optimizers = []
        for m in model:
            optimizers.append(build_optimizer(cfg, m))
        return optimizers

    @classmethod
    def build_model(cls, cfg) -> list:
        models = []
        for i in range(cfg.MODEL.NUM_MODEL):
            models.append(build_model(cfg))
            logger = logging.getLogger(__name__)
            logger.info("Model_{}:\n{}".format(i, models[i]))
        return models

    @staticmethod
    def pkt_loss(output_net, target_net, eps=0.0000001):
        # Normalize each vector by its norm
        output_net_norm = torch.sqrt(torch.sum(output_net ** 2, dim=1, keepdim=True))
        output_net = output_net / (output_net_norm + eps)
        output_net[output_net != output_net] = 0

        target_net_norm = torch.sqrt(torch.sum(target_net ** 2, dim=1, keepdim=True))
        target_net = target_net / (target_net_norm + eps)
        target_net[target_net != target_net] = 0

        # Calculate the cosine similarity
        model_similarity = torch.mm(output_net, output_net.transpose(0, 1))
        target_similarity = torch.mm(target_net, target_net.transpose(0, 1))

        # Scale cosine similarity to 0..1
        model_similarity = (model_similarity + 1.0) / 2.0
        target_similarity = (target_similarity + 1.0) / 2.0

        # Transform them into probabilities
        model_similarity = model_similarity / torch.sum(model_similarity, dim=1, keepdim=True)
        target_similarity = target_similarity / torch.sum(target_similarity, dim=1, keepdim=True)

        # Calculate the KL-divergence
        loss = torch.mean(target_similarity * torch.log((target_similarity + eps) / (model_similarity + eps)))
        return loss

    @staticmethod
    def distill_loss(y_s, y_t, t=4):
        p_s = F.log_softmax(y_s / t, dim=1)
        p_t = F.softmax(y_t / t, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (t ** 2) / y_s.shape[0]
        return loss
