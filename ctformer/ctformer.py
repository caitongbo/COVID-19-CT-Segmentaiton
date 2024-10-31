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


class CCTNet(nn.Module):

    def __init__(self, transformer_name, cnn_name, nclass, img_size, aux=True, pretrained=True, head='seghead', edge_aux=True):
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

        if self.head_name == 'seghead':
            self.decode_head = SegHead(in_channels=self.fuse_dim, num_classes=nclass, in_index=[0, 1, 2, 3])
            self.decode_head_cnn = SegHead(in_channels=self.cnn_head_dim, num_classes=nclass, in_index=[0, 1, 2, 3])
            self.decode_head_trans = SegHead(in_channels=head_dim, num_classes=nclass, in_index=[0, 1, 2, 3])

        if self.aux:
            self.auxiliary_head_cnn = FCNHead(num_convs=1, in_channels=self.cnn_head_dim[2], num_classes=nclass, in_index=2, channels=256)
            self.auxiliary_head_trans = FCNHead(num_convs=1, in_channels=self.fuse_dim[2], num_classes=nclass, in_index=2, channels=384)

        if self.edge_aux:
            self.edge_head_cnn = EdgeHead(in_channels=self.cnn_head_dim[0:2], in_index=[0, 1], channels=self.cnn_head_dim[0])
            self.edge_head_trans = EdgeHead(in_channels=head_dim[0:2], in_index=[0, 1], channels=self.fuse_dim[0])

            


        if transformer_name == 'swin_tiny' or transformer_name == 'swin_small':
            print('Transformer: '+ transformer_name)

            filters = [96, 192, 384, 768]

        elif transformer_name == 'swin_base':
            print('Transformer: '+ transformer_name)

            filters =[128, 256, 512, 1024]

        self.mobile2former = Mobile2Former(dim=64, heads=1, channel=filters[0])
        self.former2mobile = Former2Mobile(dim=filters[0], heads=1, channel=self.cnn_head_dim[0])
        self.mobile2former2 = Mobile2Former(dim=self.cnn_head_dim[0], heads=1, channel=filters[1])
        self.former2mobile2 = Former2Mobile(dim=filters[1], heads=1, channel=self.cnn_head_dim[1])
        self.mobile2former3 = Mobile2Former(dim=self.cnn_head_dim[1], heads=1, channel=filters[2])
        self.former2mobile3 = Former2Mobile(dim=filters[2], heads=1, channel=self.cnn_head_dim[2])
        self.mobile2former4 = Mobile2Former(dim=self.cnn_head_dim[2], heads=1, channel=filters[3])
        self.former2mobile4 = Former2Mobile(dim=filters[3], heads=1, channel=self.cnn_head_dim[3])

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
            nn.Conv2d(3, stem_chs[1], kernel_size=3, stride=2, padding=1, bias=True),
            nn.Conv2d(stem_chs[1], stem_chs[2], kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(stem_chs[2], eps=NORM_EPS),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(3, 2, 1),
        )


    def forward(self, x):
        size = x.size()[2:]
        outputs = []


        x1,Wh,Ww = self.transformer_backbone(x)
        # x2 = self.cnn_backbone(x)

        x2 = self.stem(x)

        out1 = []
        out2 = []
###########stage1################### 
        z_hid = self.mobile2former(x2,x1)

        z_out, H, W, x, Wh, Ww = self.transformer_backbone.layers[0](z_hid, Wh, Ww)


        out = Rearrange('b (h w) d -> b d h w', h=H, w=W)(z_out)
        out = self.rconv1(out)
        out = F.interpolate(out,size=x2.shape[-1])

        x2 = x2+out

        x_hid =  self.cnn_backbone.layer1(x2)
        x_out = self.former2mobile(x_hid, z_out)

        out = self.norm_layer1(z_out)
        out = out.view(-1, H, W, self.transformer_backbone.num_features[0]).permute(0, 3, 1, 2).contiguous()

        out1.append(out)    
        out2.append(x_out)
###########stage2################### 
        # exit()
        z_hid = self.mobile2former2(x_out,x)

        z_out, H, W, x, Wh, Ww = self.transformer_backbone.layers[1](z_hid, Wh, Ww)

        out = Rearrange('b (h w) d -> b d h w', h=H, w=W)(z_out)
        out = self.rconv2(out)
        out = F.interpolate(out,size=x_out.shape[-1])
        x_out = x_out+out

        x_hid =  self.cnn_backbone.layer2(x_out)
        x_out = self.former2mobile2(x_hid, z_out)

        out = self.norm_layer2(z_out)
        out = out.view(-1, H, W, self.transformer_backbone.num_features[1]).permute(0, 3, 1, 2).contiguous()
        out1.append(out)                               
        out2.append(x_out)
###########stage3################### 
        z_hid = self.mobile2former3(x_out,x)

        z_out, H, W, x, Wh, Ww = self.transformer_backbone.layers[2](z_hid, Wh, Ww)


        out = Rearrange('b (h w) d -> b d h w', h=H, w=W)(z_out)
        out = self.rconv3(out)
        out = F.interpolate(out,size=x_out.shape[-1])
        x_out = x_out+out

        x_hid =  self.cnn_backbone.layer3(x_out)
        x_out = self.former2mobile3(x_hid, z_out)

        out = self.norm_layer3(z_out)
        out = out.view(-1, H, W, self.transformer_backbone.num_features[2]).permute(0, 3, 1, 2).contiguous()

        out1.append(out)
        out2.append(x_out)
        # print('pass 3')
###########stage4################### 
        z_hid = self.mobile2former4(x_out,x)

        z_out, H, W, x, Wh, Ww = self.transformer_backbone.layers[3](z_hid, Wh, Ww)

        out = Rearrange('b (h w) d -> b d h w', h=H, w=W)(z_out)
        out = self.rconv4(out)
        out = F.interpolate(out,size=x_out.shape[-1])
        x_out = x_out+out

        x_hid =  self.cnn_backbone.layer4(x_out)
        x_out = self.former2mobile4(x_hid, z_out)

        out = self.norm_layer4(z_out)
        out = out.view(-1, H, W, self.transformer_backbone.num_features[3]).permute(0, 3, 1, 2).contiguous()
        out1.append(out)
        out2.append(x_out)

        c1 = out1[0]
        c2 = out1[1]
        c3 = out1[2]
        c4 = out1[3]

        p1 = out2[0]
        p2 = out2[1]
        p3 = out2[2]
        p4 = out2[3]

        out_backbone_trans = [c1,c2,c3,c4]
        out_backbone_cnn = [p1,p2,p3,p4]

        # x0 = self.decode_head(out_backbone)
        x1 = self.decode_head_cnn(out_backbone_cnn)
        x2 = self.decode_head_trans(out_backbone_trans)


        x0 = [x1, x2]

        if isinstance(x0, (list, tuple)):
            for out in x0:
                out = F.interpolate(out, size, **up_kwargs)
                outputs.append(out)
        else:
            x0 = F.interpolate(x0, size, **up_kwargs)
            outputs.append(x0)

        if self.aux:
            x1 = self.auxiliary_head_cnn(out_backbone_cnn)
            x2 = self.auxiliary_head_trans(out_backbone_trans)

            x1 = F.interpolate(x1, size, **up_kwargs)
            x2 = F.interpolate(x2, size, **up_kwargs)

            outputs.append(x1+x2)


        if self.edge_aux:
            edge_cnn = self.edge_head_cnn(out_backbone_cnn)
            edge_trans = self.edge_head_trans(out_backbone_trans)

            edge_cnn = F.interpolate(edge_cnn, size, **up_kwargs)
            edge_trans = F.interpolate(edge_trans, size, **up_kwargs)

            outputs.append(edge_cnn+edge_trans)

        # return outputs[0]+outputs[1]
        return outputs


if __name__ == '__main__':
    """Notice if torch1.6, try to replace a / b with torch.true_divide(a, b)"""
    from tools.flops_params_fps_count import flops_params_fps

    model = CCTNet(transformer_name='swin_tiny', cnn_name='resnet50', nclass=2, img_size=224, aux=True,
                   edge_aux=False, head='seghead', pretrained=False)
    flops_params_fps(model)



