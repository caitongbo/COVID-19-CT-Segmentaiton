import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.utils.checkpoint as checkpoint

from .swinT import swin_tiny, swin_small, swin_base, swin_large

from .resnet import resnet18_v1b, resnet34_v1b, resnet50_v1b, resnet101_v1b
from .head.seg import SegHead
from .head import *

from .bridge import *
from einops.layers.torch import Rearrange

up_kwargs = {'mode': 'bilinear', 'align_corners': False}


NORM_EPS=1e-5

# class ConvBNReLU(nn.Module):
#     def __init__(
#             self,
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride,
#             groups=1):
#         super(ConvBNReLU, self).__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
#                               padding=1, groups=groups, bias=False)
#         self.norm = nn.BatchNorm2d(out_channels, eps=NORM_EPS)
#         self.act = nn.ReLU(inplace=True)
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.norm(x)
#         x = self.act(x)
#         return x

# class _CAFM(nn.Module):
#     """Coupled attention fusion module"""
#     def __init__(self, channels):
#         super(_CAFM, self).__init__()
#         self.conv_value = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
#         self.conv_query = nn.Conv2d(channels, channels, kernel_size=1)
#         self.conv_key = nn.Conv2d(channels, channels, kernel_size=1)

#         self.softmax = nn.Softmax(dim=2)
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x, y):
#         value = self.conv_value(y)
#         value = value.view(value.size(0), value.size(1), -1)

#         query = self.conv_query(x)
#         key = self.conv_key(y)
#         query = query.view(query.size(0), query.size(1), -1)
#         key = key.view(key.size(0), key.size(1), -1)

#         key_mean = key.mean(2).unsqueeze(2)
#         query_mean = query.mean(2).unsqueeze(2)
#         key -= key_mean
#         query -= query_mean

#         sim_map = torch.bmm(query.transpose(1, 2), key)
#         sim_map = self.softmax(sim_map)
#         out_sim = torch.bmm(sim_map, value.transpose(1, 2))
#         out_sim = out_sim.transpose(1, 2)
#         out_sim = out_sim.view(out_sim.size(0), out_sim.size(1), *x.size()[2:])
#         out_sim = self.gamma * out_sim

#         return out_sim


# class CAFM(nn.Module):
#     """Coupled attention fusion module"""
#     def __init__(self, channels_trans, channels_cnn, channels_fuse):
#         super(CAFM, self).__init__()
#         self.conv_trans = nn.Conv2d(channels_trans, channels_fuse, kernel_size=1)
#         self.conv_cnn = nn.Conv2d(channels_cnn, channels_fuse, kernel_size=1)

#         self._CAFM1 = _CAFM(channels=channels_fuse)
#         self._CAFM2 = _CAFM(channels=channels_fuse)

#         self.fuse_conv = nn.Conv2d(2 * channels_fuse, channels_fuse, kernel_size=1)

#     def forward(self, x, y):
#         """x:transformer, y:cnn"""
#         x = self.conv_trans(x)
#         y = self.conv_cnn(y)
#         local2global = x + self._CAFM1(x, y).contiguous()
#         global2local = y + self._CAFM2(y, x).contiguous()
#         fuse = self.fuse_conv(torch.cat((local2global, global2local), dim=1))

#         return fuse


# class LAFM(nn.Module):
#     def __init__(self, channels_trans, channels_cnn, channels_fuse, residual=False):
#         super(LAFM, self).__init__()
#         self.channels_fuse = channels_fuse
#         self.residual = residual
#         self.conv_trans = nn.Conv2d(channels_trans, channels_fuse, kernel_size=1)
#         self.conv_cnn = nn.Conv2d(channels_cnn, channels_fuse, kernel_size=1)
#         self.conv_fuse = nn.Conv2d(2 * channels_fuse, 2 * channels_fuse, kernel_size=1)
#         self.conv1 = nn.Conv2d(channels_fuse, channels_fuse, kernel_size=1)
#         self.conv2 = nn.Conv2d(channels_fuse, channels_fuse, kernel_size=1)
#         self.softmax = nn.Softmax(dim=0)
#         if residual:
#             self.conv = nn.Conv2d(channels_trans + channels_cnn, channels_fuse, kernel_size=1)

#     def forward(self, x, y):
#         """x:transformer, y:cnn"""
#         if self.residual:
#             residual = self.conv(torch.cat((x, y), dim=1))
#         x = self.conv_trans(x)
#         y = self.conv_cnn(y)
#         x_ori, y_ori = x, y
#         xy_fuse = self.conv_fuse(torch.cat((x, y), 1))
#         xy_split = torch.split(xy_fuse, self.channels_fuse, dim=1)
#         x = torch.sigmoid(self.conv1(xy_split[0]))
#         y = torch.sigmoid(self.conv2(xy_split[1]))
#         weights = self.softmax(torch.stack((x, y), 0))
#         out = weights[0] * x_ori + weights[1] * y_ori
#         if self.residual:
#             out = out + residual

#         return out





# class CNN(nn.Module):
#     def __init__(self, cnn_name, dilated=False, pretrained_base=True, **kwargs):
#         super(CNN, self).__init__()
#         self.pretrained = eval(cnn_name)(pretrained=True, dilated=False, **kwargs)
#         self.conv = nn.Conv2d(3, 64, kernel_size=3, stride=2,
#                               padding=1, groups=1, bias=False)
#         self.norm = nn.BatchNorm2d(64, eps=1e-5)
#         self.act = nn.ReLU(inplace=True)

#     def forward(self, x):
#         """forwarding pre-trained network"""
#         x = self.pretrained.conv1(x)
#         x = self.pretrained.bn1(x)
#         x = self.pretrained.relu(x)
#         x = self.pretrained.maxpool(x)
#         # p1 = self.pretrained.layer1(x)
#         # p2 = self.pretrained.layer2(p1)
#         # p3 = self.pretrained.layer3(p2)
#         # p4 = self.pretrained.layer4(p3)

#         # return p1, p2, p3, p4
#         # x = self.conv(x)
#         # x = self.norm(x)
#         # x = self.act(x)
#         return x

class DilatedParallelConvBlockD2(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DilatedParallelConvBlockD2, self).__init__()
        self.conv0 = nn.Conv2d(in_planes, out_planes, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.conv1 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, dilation=1, groups=out_planes,
                               bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=2, dilation=2, groups=out_planes,
                               bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, input):
        output = self.conv0(input)
        d1 = self.conv1(output)
        d2 = self.conv2(output)
        output = d1 + d2
        output = self.bn(output)
        return output


class CCTNet(nn.Module):

    def __init__(self, transformer_name, cnn_name, nclass, img_size, aux=False, pretrained=False, head='seghead', edge_aux=False):
        super(CCTNet, self).__init__()
        cnn_dict = {
            'resnet18_v1b': 'resnet18_v1b', 'resnet18': 'resnet18_v1b',
            'resnet34_v1b': 'resnet34_v1b', 'resnet34': 'resnet34_v1b',
            'resnet50_v1b': 'resnet50_v1b', 'resnet50': 'resnet50_v1b',
            'resnet101_v1b': 'resnet101_v1b', 'resnet101': 'resnet101_v1b',
        }
        self.aux = aux
        self.edge_aux = edge_aux
        self.head_name = head

        self.model = eval(transformer_name)(nclass=nclass, img_size=img_size, aux=aux, pretrained=pretrained)
        self.transformer_backbone = self.model.backbone
        head_dim = self.model.head_dim

        self.cnn_backbone = eval(cnn_dict[cnn_name])(dilated=False, pretrained=pretrained)

        # # self.cnn_backbone = CNN(cnn_name=cnn_dict[cnn_name], dilated=False, pretrained_base=pretrained)
        if 'resnet18' in cnn_name or 'resnet34' in cnn_name:
            self.cnn_head_dim = [64, 128, 256, 512]
        if 'resnet50' in cnn_name or 'resnet101' in cnn_name:
            self.cnn_head_dim = [256, 512, 1024, 2048]

        self.fuse_dim = head_dim

        # if self.head_name == 'apchead':
        #     self.decode_head = APCHead(in_channels=head_dim[3], num_classes=nclass, in_index=3, channels=512)

        # if self.head_name == 'aspphead':
        #     self.decode_head = ASPPHead(in_channels=head_dim[3], num_classes=nclass, in_index=3)

        # if self.head_name == 'asppplushead':
        #     self.decode_head = ASPPPlusHead(in_channels=head_dim[3], num_classes=nclass, in_index=[0, 3])

        # if self.head_name == 'dahead':
        #     self.decode_head = DAHead(in_channels=head_dim[3], num_classes=nclass, in_index=3)

        # if self.head_name == 'dnlhead':
        #     self.decode_head = DNLHead(in_channels=head_dim[3], num_classes=nclass, in_index=3, channels=512)

        # if self.head_name == 'fcfpnhead':
        #     self.decode_head = FCFPNHead(in_channels=head_dim, num_classes=nclass, in_index=[0, 1, 2, 3], channels=256)

        # if self.head_name == 'cefpnhead':
        #     self.decode_head = CEFPNHead(in_channels=head_dim, num_classes=nclass, in_index=[0, 1, 2, 3], channels=256)

        # if self.head_name == 'fcnhead':
        #     self.decode_head = FCNHead(in_channels=head_dim[3], num_classes=nclass, in_index=3, channels=512)

        # if self.head_name == 'gchead':
        #     self.decode_head = GCHead(in_channels=head_dim[3], num_classes=nclass, in_index=3, channels=512)

        # if self.head_name == 'psahead':
        #     self.decode_head = PSAHead(in_channels=head_dim[3], num_classes=nclass, in_index=3)

        # if self.head_name == 'psphead':
        #     self.decode_head = PSPHead(in_channels=head_dim[3], num_classes=nclass, in_index=3)

        if self.head_name == 'seghead':
            self.decode_head = SegHead(in_channels=self.fuse_dim, num_classes=nclass, in_index=[0, 1, 2, 3])
            self.decode_head_cnn = SegHead(in_channels=self.cnn_head_dim, num_classes=nclass, in_index=[0, 1, 2, 3])
            self.decode_head_trans = SegHead(in_channels=head_dim, num_classes=nclass, in_index=[0, 1, 2, 3])

        # if self.head_name == 'unethead':
        #     self.decode_head = UNetHead(in_channels=self.fuse_dim, num_classes=nclass, in_index=[0, 1, 2, 3])

        # if self.head_name == 'uperhead':
        #     self.decode_head = UPerHead(in_channels=self.fuse_dim, num_classes=nclass)

        # if self.head_name == 'annhead':
        #     self.decode_head = ANNHead(in_channels=head_dim[2:], num_classes=nclass, in_index=[2, 3], channels=512)

        # if self.head_name == 'mlphead':
        #     self.decode_head = MLPHead(in_channels=head_dim, num_classes=nclass, in_index=[0, 1, 2, 3], channels=256)

        if self.aux:
            self.auxiliary_head = FCNHead(num_convs=1, in_channels=self.fuse_dim[2], num_classes=nclass, in_index=2, channels=256)

        if self.edge_aux:
            self.edge_head = EdgeHead(in_channels=head_dim[0:2], in_index=[0, 1], channels=self.fuse_dim[0])

        # self.fuse1 = LAFM(head_dim[0], self.cnn_head_dim[0], self.fuse_dim[0], residual=True)
        # self.fuse2 = LAFM(head_dim[1], self.cnn_head_dim[1], self.fuse_dim[1], residual=True)
        # # self.fuse3 = LAFM(head_dim[2], self.cnn_head_dim[2], self.fuse_dim[2], residual=True)
        # # self.fuse4 = LAFM(head_dim[3], self.cnn_head_dim[3], self.fuse_dim[3], residual=True)

        # # self.fuse1 = CAFM(channels_trans=head_dim[0], channels_cnn=self.cnn_head_dim[0], channels_fuse=self.fuse_dim[0])
        # # self.fuse2 = CAFM(channels_trans=head_dim[1], channels_cnn=self.cnn_head_dim[1], channels_fuse=self.fuse_dim[1])
        # self.fuse3 = CAFM(channels_trans=head_dim[2], channels_cnn=self.cnn_head_dim[2], channels_fuse=self.fuse_dim[2])
        # self.fuse4 = CAFM(channels_trans=head_dim[3], channels_cnn=self.cnn_head_dim[3], channels_fuse=self.fuse_dim[3])


        if transformer_name == 'swin_tiny' or transformer_name == 'swin_small':
            print('Transformer: '+ transformer_name)

            filters = [96, 192, 384, 768]

        elif transformer_name == 'swin_base':
            print('Transformer: '+ transformer_name)

            filters =[128, 256, 512, 1024]

        self.mobile2former = Mobile2Former(dim=filters[0], heads=1, channel=filters[0])
        self.former2mobile = Former2Mobile(dim=filters[0], heads=1, channel=self.cnn_head_dim[0])
        self.mobile2former2 = Mobile2Former(dim=filters[1], heads=1, channel=filters[1])
        self.former2mobile2 = Former2Mobile(dim=filters[1], heads=1, channel=self.cnn_head_dim[1])
        self.mobile2former3 = Mobile2Former(dim=filters[2], heads=1, channel=filters[2])
        self.former2mobile3 = Former2Mobile(dim=filters[2], heads=1, channel=self.cnn_head_dim[2])
        self.mobile2former4 = Mobile2Former(dim=filters[3], heads=1, channel=filters[3])
        self.former2mobile4 = Former2Mobile(dim=filters[3], heads=1, channel=self.cnn_head_dim[3])

        self.conv1 = nn.Conv2d(64,filters[0],1)
        self.conv2 = nn.Conv2d(self.cnn_head_dim[0],filters[1],1)
        self.conv3 = nn.Conv2d(self.cnn_head_dim[1],filters[2],1)
        self.conv4 = nn.Conv2d(self.cnn_head_dim[2],filters[3],1)

        self.norm_layer1 = nn.LayerNorm(filters[0])
        self.norm_layer2 = nn.LayerNorm(filters[1])
        self.norm_layer3 = nn.LayerNorm(filters[2])
        self.norm_layer4 = nn.LayerNorm(filters[3])

        self.rconv1 = nn.Conv2d(filters[0],64,1)
        self.rconv2 = nn.Conv2d(filters[1],self.cnn_head_dim[0],1)
        self.rconv3 = nn.Conv2d(filters[2],self.cnn_head_dim[1],1)
        self.rconv4 = nn.Conv2d(filters[3],self.cnn_head_dim[2],1)


        stem_chs=[64, 32, 64]
        self.stem = nn.Sequential(
            # ConvBNReLU(3, stem_chs[0], kernel_size=3, stride=2),
            # ConvBNReLU(stem_chs[0], stem_chs[1], kernel_size=3, stride=1),
            # ConvBNReLU(stem_chs[1], stem_chs[2], kernel_size=3, stride=1),
            # ConvBNReLU(stem_chs[2], stem_chs[2], kernel_size=3, stride=2),
            nn.Conv2d(3, stem_chs[1], kernel_size=3, stride=2, padding=1, bias=True),
            nn.Conv2d(stem_chs[1], stem_chs[2], kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(stem_chs[2], eps=NORM_EPS),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )


        self.up4_conv4_c = nn.Conv2d(self.cnn_head_dim[3], self.cnn_head_dim[3], 1, stride=1, padding=0)
        self.up4_bn4_c = nn.BatchNorm2d(self.cnn_head_dim[3])
        self.up4_act_c = nn.PReLU(self.cnn_head_dim[3])

        self.up3_conv4_c = DilatedParallelConvBlockD2(self.cnn_head_dim[3], self.cnn_head_dim[2])
        self.up3_conv3_c = nn.Conv2d(self.cnn_head_dim[2], self.cnn_head_dim[2], 1, stride=1, padding=0)
        self.up3_bn3_c = nn.BatchNorm2d(self.cnn_head_dim[2])
        self.up3_act_c = nn.PReLU(self.cnn_head_dim[2])

        self.up2_conv3_c = DilatedParallelConvBlockD2(self.cnn_head_dim[2], self.cnn_head_dim[1])
        self.up2_conv2_c = nn.Conv2d(self.cnn_head_dim[1], self.cnn_head_dim[1], 1, stride=1, padding=0)
        self.up2_bn2_c = nn.BatchNorm2d(self.cnn_head_dim[1])
        self.up2_act_c = nn.PReLU(self.cnn_head_dim[1])

        self.up1_conv2_c = DilatedParallelConvBlockD2(self.cnn_head_dim[1], self.cnn_head_dim[0])
        self.up1_conv1_c = nn.Conv2d(self.cnn_head_dim[0], self.cnn_head_dim[0], 1, stride=1, padding=0)
        self.up1_bn1_c = nn.BatchNorm2d(self.cnn_head_dim[0])
        self.up1_act_c = nn.PReLU(self.cnn_head_dim[0])

##########tran
        self.up4_conv4 = nn.Conv2d(768, 768, 1, stride=1, padding=0)
        self.up4_bn4 = nn.BatchNorm2d(768)
        self.up4_act = nn.PReLU(768)

        self.up3_conv4 = DilatedParallelConvBlockD2(768, 384)
        self.up3_conv3 = nn.Conv2d(384, 384, 1, stride=1, padding=0)
        self.up3_bn3 = nn.BatchNorm2d(384)
        self.up3_act = nn.PReLU(384)

        self.up2_conv3 = DilatedParallelConvBlockD2(384, 192)
        self.up2_conv2 = nn.Conv2d(192, 192, 1, stride=1, padding=0)
        self.up2_bn2 = nn.BatchNorm2d(192)
        self.up2_act = nn.PReLU(192)

        self.up1_conv2 = DilatedParallelConvBlockD2(192, 96)
        self.up1_conv1 = nn.Conv2d(96, 96, 1, stride=1, padding=0)
        self.up1_bn1 = nn.BatchNorm2d(96)
        self.up1_act = nn.PReLU(96)


        self.pred4_c = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(self.cnn_head_dim[3], 2, 1, stride=1, padding=0))
        self.pred3_c = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(self.cnn_head_dim[2], 2, 1, stride=1, padding=0))
        self.pred2_c = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(self.cnn_head_dim[1], 2, 1, stride=1, padding=0))
    
        self.pred4 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(768, 2, 1, stride=1, padding=0))
        self.pred3 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(384, 2, 1, stride=1, padding=0))
        self.pred2 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(192, 2, 1, stride=1, padding=0))
    
        self.pred1_c = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(self.cnn_head_dim[0], 2, 1, stride=1, padding=0))
        self.pred1 = nn.Sequential(nn.Dropout2d(0.01, False), nn.Conv2d(96, 2, 1, stride=1, padding=0))


    def forward(self, x):
        size = x.size()[2:]
        outputs = []


        x1,Wh,Ww = self.transformer_backbone(x)
        # x2 = self.cnn_backbone(x)

        x2 = self.stem(x)

        out1 = []
        out2 = []
###########stage1################### 
        z_hid = self.mobile2former(self.conv1(x2),x1)

        z_out, H, W, x, Wh, Ww = self.transformer_backbone.layers[0](z_hid, Wh, Ww)

        out = Rearrange('b (h w) d -> b d h w', h=H, w=W)(z_out)
        out = self.rconv1(out)
        out = F.interpolate(out,size=x2.shape[-1])

        out = x2+out

        x_hid =  self.cnn_backbone.layer1(out)
        x_out = self.former2mobile(x_hid, z_out)

        out = self.norm_layer1(z_out)
        out = out.view(-1, H, W, self.transformer_backbone.num_features[0]).permute(0, 3, 1, 2).contiguous()

        out1.append(out)    
        out2.append(x_out)
###########stage2################### 
        # exit()
        z_hid = self.mobile2former2(self.conv2(x_out),x)
        z_out, H, W, x, Wh, Ww = self.transformer_backbone.layers[1](z_hid, Wh, Ww)

        out = Rearrange('b (h w) d -> b d h w', h=H, w=W)(z_out)
        out = self.rconv2(out)
        out = F.interpolate(out,size=x_out.shape[-1])
        out = x_out+out

        x_hid =  self.cnn_backbone.layer2(out)
        x_out = self.former2mobile2(x_hid, z_out)

        out = self.norm_layer2(z_out)
        out = out.view(-1, H, W, self.transformer_backbone.num_features[1]).permute(0, 3, 1, 2).contiguous()
        out1.append(out)                               
        out2.append(x_out)
###########stage3################### 
        z_hid = self.mobile2former3(self.conv3(x_out),x)
        z_out, H, W, x, Wh, Ww = self.transformer_backbone.layers[2](z_hid, Wh, Ww)

        out = Rearrange('b (h w) d -> b d h w', h=H, w=W)(z_out)
        out = self.rconv3(out)
        out = F.interpolate(out,size=x_out.shape[-1])
        out = x_out+out

        x_hid =  self.cnn_backbone.layer3(out)
        x_out = self.former2mobile3(x_hid, z_out)

        out = self.norm_layer3(z_out)
        out = out.view(-1, H, W, self.transformer_backbone.num_features[2]).permute(0, 3, 1, 2).contiguous()

        out1.append(out)
        out2.append(x_out)
        # print('pass 3')
###########stage4################### 
        z_hid = self.mobile2former4(self.conv4(x_out),x)
        z_out, H, W, x, Wh, Ww = self.transformer_backbone.layers[3](z_hid, Wh, Ww)

        out = Rearrange('b (h w) d -> b d h w', h=H, w=W)(z_out)
        out = self.rconv4(out)
        out = F.interpolate(out,size=x_out.shape[-1])
        out = x_out+out

        x_hid =  self.cnn_backbone.layer4(out)
        x_out = self.former2mobile4(x_hid, z_out)

        out = self.norm_layer4(z_out)
        out = out.view(-1, H, W, self.transformer_backbone.num_features[3]).permute(0, 3, 1, 2).contiguous()
        out1.append(out)
        out2.append(x_out)

        c1 = out1[0]
        c2 = out1[1]
        c3 = out1[2]
        c4 = out1[3]

        # print(c1.shape)
        # print(c2.shape)
        # print(c3.shape)
        # print(c4.shape)

        p1 = out2[0]
        p2 = out2[1]
        p3 = out2[2]
        p4 = out2[3]

        # x_cp1 = self.fuse1(c1, p1)
        # x_cp2 = self.fuse2(c2, p2)
        # x_cp3 = self.fuse3(c3, p3)
        # x_cp4 = self.fuse4(c4, p4)

        # x_cp1 = c1+p1
        # x_cp2 = c2+p2
        # x_cp3 = c3+p3
        # x_cp4 = c4+p4


        # up4_conv4 = self.up4_bn4(self.up4_conv4(c4))
        # up4 = self.up4_act(up4_conv4)

        # up4 = F.interpolate(up4, c3.size()[2:], mode='bilinear', align_corners=False)
        # up3_conv4 = self.up3_conv4(up4)
        # up3_conv3 = self.up3_bn3(self.up3_conv3(c3))
        # up3 = self.up3_act(up3_conv4 + up3_conv3)

        # up3 = F.interpolate(up3, c2.size()[2:], mode='bilinear', align_corners=False)
        # up2_conv3 = self.up2_conv3(up3)
        # up2_conv2 = self.up2_bn2(self.up2_conv2(c2))
        # up2 = self.up2_act(up2_conv3 + up2_conv2)

        # up2 = F.interpolate(up2, c1.size()[2:], mode='bilinear', align_corners=False)
        # up1_conv2 = self.up1_conv2(up2)
        # up1_conv1 = self.up1_bn1(self.up1_conv1(c1))
        # up1 = self.up1_act(up1_conv2 + up1_conv1)


        # up4_conv4_c = self.up4_bn4_c(self.up4_conv4_c(p4))
        # up4_c = self.up4_act_c(up4_conv4_c)

        # up4_c = F.interpolate(up4_c, p3.size()[2:], mode='bilinear', align_corners=False)
        # up3_conv4_c = self.up3_conv4_c(up4_c)
        # up3_conv3_c = self.up3_bn3_c(self.up3_conv3_c(p3))
        # up3_c = self.up3_act_c(up3_conv4_c + up3_conv3_c)

        # up3_c = F.interpolate(up3_c, p2.size()[2:], mode='bilinear', align_corners=False)
        # up2_conv3_c = self.up2_conv3_c(up3_c)
        # up2_conv2_c = self.up2_bn2_c(self.up2_conv2_c(p2))
        # up2_c = self.up2_act_c(up2_conv3_c + up2_conv2_c)

        # up2_c = F.interpolate(up2_c, p1.size()[2:], mode='bilinear', align_corners=False)
        # up1_conv2_c = self.up1_conv2_c(up2_c)
        # up1_conv1_c = self.up1_bn1_c(self.up1_conv1_c(p1))
        # up1_c = self.up1_act_c(up1_conv2_c + up1_conv1_c)


        # out_backbone = [x_cp1, x_cp2, x_cp3, x_cp4]
        # out_backbone_trans = [up1,up2,up3,up4]
        # out_backbone_cnn = [up1_c,up2_c,up3_c,up4_c]


        # pred4 = torch.cat((self.pred4_c(up4_c), self.pred4(up4)),-1)
        # pred3 = torch.cat((self.pred3_c(up3_c), self.pred3(up3)),-1)
        # pred2 = torch.cat((self.pred2_c(up2_c), self.pred2(up2)),-1)
        # pred1 = torch.cat((self.pred1_c(up1_c), self.pred1(up1)),-1)


        # pred4 = self.pred4_c(up4_c)
        # pred3 = self.pred3_c(up3_c)
        # pred2 = self.pred2_c(up2_c)
        # pred1 = self.pred1_c(up1_c)


        # pred4 = self.pred4(up4)
        # pred3 = self.pred3(up3)
        # pred2 = self.pred2(up2)
        # pred1 = self.pred1(up1)

        # pred4 = F.interpolate(pred4, size, mode='bilinear', align_corners=False)
        # pred3 = F.interpolate(pred3, size, mode='bilinear', align_corners=False)
        # pred2 = F.interpolate(pred2, size, mode='bilinear', align_corners=False)
        # pred1 = F.interpolate(pred1, size, mode='bilinear', align_corners=False)

        # return (pred1, pred2, pred3, pred4, )

        # for i, out in enumerate(out_backbone):
        #     draw_features(out, f'C{i}')

        out_backbone_trans = [c1,c2,c3,c4]
        out_backbone_cnn = [p1,p2,p3,p4]

        # x0 = self.decode_head(out_backbone)
        x1 = self.decode_head_cnn(out_backbone_cnn)
        x2 = self.decode_head_trans(out_backbone_trans)

        # out = torch.cat((x1, x2),-1)
        # x0 = [x0, x1, x2]

        x0 = [x1, x2]


        if isinstance(x0, (list, tuple)):
            for out in x0:
                out = F.interpolate(out, size, **up_kwargs)
                outputs.append(out)
        else:
            x0 = F.interpolate(x0, size, **up_kwargs)
            outputs.append(x0)

        # if self.aux:
        #     x1 = self.auxiliary_head(out_backbone)
        #     x1 = F.interpolate(x1, size, **up_kwargs)
        #     outputs.append(x1)

        # if self.edge_aux:
        #     edge = self.edge_head(out_backbone)
        #     edge = F.interpolate(edge, size, **up_kwargs)
        #     outputs.append(edge)

        # n = len(outputs)
        # print(n)
        # exit()
        # sum=0
        # for i in range(n):
        #     sum+=outputs[i]
        # return sum/n

        return outputs[0]+outputs[1]


if __name__ == '__main__':
    """Notice if torch1.6, try to replace a / b with torch.true_divide(a, b)"""
    from tools.flops_params_fps_count import flops_params_fps

    model = CCTNet(transformer_name='swin_tiny', cnn_name='resnet50', nclass=2, img_size=224, aux=True,
                   edge_aux=False, head='seghead', pretrained=False)
    flops_params_fps(model)



