import torch
import torch.nn as nn
from models.swin_resnet.backbone.swin import SwinTransformer
from mmcv.ops import SyncBatchNorm
from models.swin_resnet.decode_heads.uper_head import UPerHead
from models.swin_resnet.utils import resize

import torch.nn.functional as F

class Swin_UperNet(nn.Module):
    def __init__(self, num_classes=21, sync_bn=True, freeze_bn=False):
        super(Swin_UperNet, self).__init__()

        if sync_bn == True:
            norm_cfg = dict(type='SyncBN', requires_grad=True)
        else:
            norm_cfg = dict(type='BN', requires_grad=True)

        self.backbone = SwinTransformer(
            embed_dims=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
            use_abs_pos_embed=False,
            drop_path_rate=0.3,
            patch_norm=True,
            norm_cfg = dict(type='LN', requires_grad=True),
            pretrained='/workspace/ct/weights/swin_tiny_patch4_window7_224.pth'
        )

        self.decoder = UPerHead(
            in_channels=[96, 192, 384, 768],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=norm_cfg
        )

    def forward(self, img):
        x = self.backbone(img)
        out = self.decoder.forward(x)
        out = F.interpolate(out,size=224)

        return out

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SyncBatchNorm):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()


