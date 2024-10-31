import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.functional as F
from .Encoder import All2Cross
from .Decoder import ConvUpsample, SegmentationHead


class HiFormer(nn.Module):
    def __init__(self, config, img_size=224, in_chans=3, n_classes=9):
        super().__init__()
        self.img_size = img_size
        self.patch_size = [4, 8, 16, 32]
        self.n_classes = n_classes
        self.All2Cross = All2Cross(config = config, img_size= img_size, in_chans=in_chans)
        
        #
        self.ConvUp_1 = ConvUpsample(in_chans=96, upsample=False)
        self.ConvUp_2 = ConvUpsample(in_chans=192, upsample=False)
        self.ConvUp_3 = ConvUpsample(in_chans=384, upsample=False)
        self.ConvUp_4 = ConvUpsample(in_chans=768, out_chans=[128,128,128,128], upsample=True)

        ## swin b
        # self.ConvUp_1 = ConvUpsample(in_chans=128, upsample=False)
        # self.ConvUp_2 = ConvUpsample(in_chans=256, upsample=False)
        # self.ConvUp_3 = ConvUpsample(in_chans=512, upsample=False)
        # self.ConvUp_4 = ConvUpsample(in_chans=1024, out_chans=[128,128,128,128], upsample=True)
    
        self.segmentation_head = SegmentationHead(
            in_channels=16,
            out_channels=n_classes,
            kernel_size=3,
        )    

        self.conv_pred = nn.Sequential(
            nn.Conv2d(
                128, 16,
                kernel_size=1, stride=1,
                padding=0, bias=True),
            # nn.GroupNorm(8, 16), 
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        )
    
    def forward(self, x):
        xs = self.All2Cross(x)
        embeddings = [x[:, 1:] for x in xs]
        reshaped_embed = []
        for i, embed in enumerate(embeddings):
            
            embed = Rearrange('b (h w) d -> b d h w', h=(self.img_size//self.patch_size[i]), w=(self.img_size//self.patch_size[i]))(embed)
            if i==0:
                embed = self.ConvUp_1(embed)
            elif i==1:
                embed = self.ConvUp_2(embed)
            elif i==2:
                embed = self.ConvUp_3(embed)
            elif i==3:
                embed = self.ConvUp_4(embed)

            reshaped_embed.append(embed)

        C = F.interpolate(reshaped_embed[0],size=112) +  F.interpolate(reshaped_embed[1],size=112) +F.interpolate(reshaped_embed[2],size=112) + reshaped_embed[3]
        # C = reshaped_embed[3]
        # C = F.interpolate(reshaped_embed[0],size=112) + F.interpolate(reshaped_embed[2],size=112)

        # C = F.interpolate(reshaped_embed[0],size=112) +  F.interpolate(reshaped_embed[1],size=112) +F.interpolate(reshaped_embed[2],size=112)

        C = self.conv_pred(C)

        out = self.segmentation_head(C)
        
        return out  