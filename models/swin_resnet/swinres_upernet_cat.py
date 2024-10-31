import torch
import torch.nn as nn
from models.swin_resnet.backbone.swin import SwinTransformer
from mmcv.ops import SyncBatchNorm
from models.swin_resnet.decode_heads.uper_head import UPerHead
from models.swin_resnet.backbone.resnet import resnet34, resnet50, resnet101
from models.swin_resnet.modules.LAM import AFModule
import torch.nn.functional as F


# class ConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_relu=True):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels,
#                               kernel_size, stride, padding)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.relu = nn.ReLU()
#         self.with_relu = with_relu

#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         if self.with_relu:
#             x = self.relu(x)
#         return x

# class Bridge(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()

#         # print(in_channels)
#         # print(out_channels)

#         self.bridge = nn.Sequential(
#             ConvBlock(in_channels, out_channels),
#             ConvBlock(out_channels, out_channels)
#         )

#     def forward(self, x):
#         return self.bridge(x)

class SwinRes_UperNet(nn.Module):
    def __init__(self, num_classes=21, sync_bn=True, freeze_bn=False):
        super(SwinRes_UperNet, self).__init__()

        if sync_bn == True:
            norm_cfg = dict(type='SyncBN', requires_grad=True)
        else:
            norm_cfg = dict(type='BN', requires_grad=True)
        self.in_channels = [128, 256, 512, 1024]
        self.backbone1 = SwinTransformer(
            embed_dims=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            use_abs_pos_embed=False,
            drop_path_rate=0.3,
            patch_norm=True,
            pretrained='/root/workspace/data/ctb/swin_base_patch4_window7_224_22k.pth'
        )


        self.backbone2 = resnet50(pretrained=True)


        self.AFMs = nn.ModuleList()
        self.lc_convs = nn.ModuleList()
        for in_channel in self.in_channels:
            AFM = AFModule(inplace=in_channel)
            lc_conv = nn.Conv2d(in_channel * 2, in_channel, 3, padding=1)
            self.AFMs.append(AFM)
            self.lc_convs.append(lc_conv)

        self.decoder = UPerHead(
            in_channels=[128, 256, 512, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=num_classes,
            norm_cfg=norm_cfg
        )

        self.freeze_bn = freeze_bn
        
        # self.bridge2048 = Bridge(2048, 2048)
        # self.bridge1024 = Bridge(1024, 1024)
        # self.bridge512 = Bridge(512, 512)
        # self.bridge256 = Bridge(256, 256)
        # self.bridge128 = Bridge(128, 128)

        self.conv2 = nn.Conv2d(5760, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, 2, kernel_size=1, stride=1, padding=0)

    def _upsample(self, x, y, scale=1):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H // scale, W // scale), mode='bilinear',align_corners=True)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear',align_corners=True) + y



    def forward(self, img):

        # print(img.shape)

        x1 = self.backbone1(img)
        x2 = self.backbone2(img)

        # print(x2[0].shape)
        # print(x2[1].shape)
        # print(x2[2].shape)
        # print(x2[3].shape)

        # x1[0] = self.bridge128(x1[0])
        # x1[1] = self.bridge256(x1[1])
        # x1[2] = self.bridge512(x1[2])
        # x1[3] = self.bridge1024(x1[3])


        # x2[0] = self.bridge256(x2[0])
        # x2[1] = self.bridge512(x2[1])
        # x2[2] = self.bridge1024(x2[2])
        # x2[3] = self.bridge2048(x2[3])

        # print("--------------")
        # print(x1[0].shape)
        # print(x1[1].shape)
        # print(x1[2].shape)
        # print(x1[3].shape)

        # print("--------------")
        # print(x2[0].shape)
        # print(x2[1].shape)
        # print(x2[2].shape)
        # print(x2[3].shape)

        x2[1] = self._upsample(x2[1], x2[0])
        x2[2] = self._upsample(x2[2], x2[0])
        x2[3] = self._upsample(x2[3], x2[0])


        x1[1] = self._upsample(x1[1], x1[0])
        x1[2] = self._upsample(x1[2], x1[0])
        x1[3] = self._upsample(x1[3], x1[0])

        out1 = torch.cat((x2[0], x2[1], x2[2], x2[3]), 1)
        out2 = torch.cat((x1[0], x1[1], x1[2], x1[3]), 1)
        out3 = torch.cat((out1, out2), 1)


        # print(out3.shape)
        out3 = self.conv2(out3)
        out3 = self.relu2(self.bn2(out3))
        out = self.conv3(out3)

        # c2 = [
        #     lc_conv(x2[i])
        #     for i, lc_conv in enumerate(self.lc_convs)
        # ]
        # e = [
        #     AFM(x1[i], c2[i])
        #     for i, AFM in enumerate(self.AFMs)
        # ]

        # print("--------------")
        # print(e[0].shape)
        # print(e[1].shape)
        # print(e[2].shape)
        # print(e[3].shape)
        # print("--------------")


        # out = self.decoder.forward(e)

        # print(out.shape)

        # exit()

        out = F.interpolate(out, scale_factor=4)

        return out


    # def freeze_bn(self):
    #     for m in self.modules():
    #         if isinstance(m, SyncBatchNorm):
    #             m.eval()
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.eval()

if __name__ == '__main__':
    model = SwinRes_UperNet(num_classes=2, sync_bn=False).cuda()
    a = torch.rand(2, 3, 512, 512).cuda()
    b = model(a)
    print(b.shape)

