# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from fastreid.utils.weight_init import weights_init_kaiming, weights_init_classifier
from .build import META_ARCH_REGISTRY


@META_ARCH_REGISTRY.register()
class LDS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._cfg = cfg
        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, -1, 1, 1))
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(1, -1, 1, 1))

        self.in_channel = cfg.FUSION.EMBEDDING_CHANL
        self.in_spatial = cfg.FUSION.EMBEDDING_SPATL

        self.inter_channel = self.in_channel // cfg.FUSION.CHA_RATIO
        self.inter_spatial = self.in_spatial // cfg.FUSION.SPA_RATIO
        self.down_ratio = cfg.FUSION.DOWN_RATIO
        self.con1x1_style = cfg.FUSION.CON1x1_STYLE

        self.use_spatial = cfg.FUSION.USE_SPATIAL
        self.use_channel = cfg.FUSION.USE_CHANNEL

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel * self.con1x1_style, out_channels=self.in_channel,
                      kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.in_channel),
        )

        # head
        self.heads = build_heads(cfg)

        self.fusion.apply(weights_init_kaiming)

        if self.use_spatial:
            self.x_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
            self.y_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
            self.gx_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
            self.gy_spatial = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channel, out_channels=self.inter_channel,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_channel),
                nn.ReLU()
            )
            num_channel_s = 1 + self.in_spatial
            self.Wx_spatial = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s // self.down_ratio,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_channel_s // self.down_ratio),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_channel_s // self.down_ratio, out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )
            self.Wy_spatial = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s // self.down_ratio,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_channel_s // self.down_ratio),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_channel_s // self.down_ratio, out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )
            self.x_spatial.apply(weights_init_kaiming)
            self.y_spatial.apply(weights_init_kaiming)
            self.Wx_spatial.apply(weights_init_kaiming)
            self.Wy_spatial.apply(weights_init_kaiming)
            self.gx_spatial.apply(weights_init_kaiming)
            self.gy_spatial.apply(weights_init_kaiming)

        if self.use_channel:
            self.x_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
            self.y_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
            self.gx_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
            self.gy_channel = nn.Sequential(
                nn.Conv2d(in_channels=self.in_spatial, out_channels=self.inter_spatial,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.inter_spatial),
                nn.ReLU()
            )
            num_channel_s = 1 + self.in_channel
            self.Wx_channel = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s // self.down_ratio,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_channel_s // self.down_ratio),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_channel_s // self.down_ratio, out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )
            self.Wy_channel = nn.Sequential(
                nn.Conv2d(in_channels=num_channel_s, out_channels=num_channel_s // self.down_ratio,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(num_channel_s // self.down_ratio),
                nn.ReLU(),
                nn.Conv2d(in_channels=num_channel_s // self.down_ratio, out_channels=1,
                          kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1)
            )
            self.x_channel.apply(weights_init_kaiming)
            self.y_channel.apply(weights_init_kaiming)
            self.Wx_channel.apply(weights_init_kaiming)
            self.Wy_channel.apply(weights_init_kaiming)
            self.gx_channel.apply(weights_init_kaiming)
            self.gy_channel.apply(weights_init_kaiming)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, x, y):
        if self.training:
            targets = x["targets"]
            x, y = x["outputs"]["feature_map"], y["outputs"]["feature_map"]
        else:
            x, y = x["feature_map"], y["feature_map"]

        b, c, h, w = x.size()

        if self.use_spatial:
            x_spatial = self.x_spatial(x)
            y_spatial = self.y_spatial(y)
            x_spatial = x_spatial.view(b, self.inter_channel, -1)
            x_spatial = x_spatial.permute(0, 2, 1)
            y_spatial = y_spatial.view(b, self.inter_channel, -1)
            Gs = torch.matmul(x_spatial, y_spatial)
            Gs_x = Gs.permute(0, 2, 1).view(b, h*w, h, w)
            Gs_y = Gs.view(b, h*w, h, w)

            gx_spatial = self.gx_spatial(x)
            gx_spatial = torch.mean(gx_spatial, dim=1, keepdim=True)
            gx_spatial = torch.cat((Gs_x, gx_spatial), 1)
            Wx_spatial = self.Wx_spatial(gx_spatial)
            x = torch.sigmoid(Wx_spatial.expand_as(x)) * x

            gy_spatial = self.gy_spatial(y)
            gy_spatial = torch.mean(gy_spatial, dim=1, keepdim=True)
            gy_spatial = torch.cat((Gs_y, gy_spatial), 1)
            Wy_spatial = self.Wy_spatial(gy_spatial)
            y = torch.sigmoid(Wy_spatial.expand_as(y)) * y

        if self.use_channel:
            xc = x.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)
            yc = y.view(b, c, -1).permute(0, 2, 1).unsqueeze(-1)
            x_channel = self.x_channel(xc).squeeze(-1).permute(0, 2, 1)
            y_channel = self.y_channel(yc).squeeze(-1)
            Gc = torch.matmul(x_channel, y_channel)
            Gc_x = Gc.permute(0, 2, 1).unsqueeze(-1)
            Gc_y = Gc.unsqueeze(-1)

            gx_channel = self.gx_channel(xc)
            gx_channel = torch.mean(gx_channel, dim=1, keepdim=True)
            gx_channel = torch.cat((Gc_x, gx_channel), 1)
            Wx_channel = self.Wx_channel(gx_channel).transpose(1,2)
            x = torch.sigmoid(Wx_channel.expand_as(x)) * x

            gy_channel = self.gy_channel(yc)
            gy_channel = torch.mean(gy_channel, dim=1, keepdim=True)
            gy_channel = torch.cat((Gc_y, gy_channel), 1)
            Wy_channel = self.Wy_channel(gy_channel).transpose(1,2)
            y = torch.sigmoid(Wy_channel.expand_as(y)) * y

        if self.con1x1_style == 2:
            feature_joint = torch.cat((x, y), 1)
        elif self.con1x1_style == 1:
            feature_joint = x + y

        features = self.fusion(feature_joint)
        if self.training:
            targets = targets.to(self.device)

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            outputs = self.heads(features, targets)
            return {
                "outputs": outputs,
                "targets": targets,
            }
        else:
            outputs = self.heads(features)
            return outputs

    def losses(self, outs):
        r"""
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # fmt: off
        outputs           = outs["outputs"]
        gt_labels         = outs["targets"]
        # model predictions
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self._cfg.MODEL.LOSSES.NAME

        if "CrossEntropyLoss" in loss_names:
            loss_dict["loss_cls"] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                self._cfg.MODEL.LOSSES.CE.EPSILON,
                self._cfg.MODEL.LOSSES.CE.ALPHA,
            ) * self._cfg.MODEL.LOSSES.CE.SCALE

        if "TripletLoss" in loss_names:
            loss_dict["loss_triplet"] = triplet_loss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.TRI.MARGIN,
                self._cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                self._cfg.MODEL.LOSSES.TRI.HARD_MINING,
            ) * self._cfg.MODEL.LOSSES.TRI.SCALE

        if "CircleLoss" in loss_names:
            loss_dict["loss_circle"] = pairwise_circleloss(
                pred_features,
                gt_labels,
                self._cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                self._cfg.MODEL.LOSSES.CIRCLE.GAMMA,
            ) * self._cfg.MODEL.LOSSES.CIRCLE.SCALE

        return loss_dict
