import imp
import torch
from torch import nn

################################################
from models.miniseg import MiniSeg
from models.segnet import SegNet
from models.unet import Unet
from models.deeplab2 import DeepLabv3_plus
from models.pspnet import PSPNet
from models.UNet_3Plus import UNet_3Plus_DeepSup
from models.ENet import ENet
from models.GCN import GCN
from models.ResUNet import ResUnet
from models.ResUNetpp import ResUnetPlusPlus
from models.TransUNet import TransUnet
from models.nnunet import initialize_network
# from models.vit_adapter import ViTAdapter
from models.swin_resnet import build_model
from models.transfuse.TransFuse import TransFuse_S
import models.archs as archs 
from models.tmunet.TransMUNet import TransMUNet
from models.vision_transformer import SwinUnet as ViT_seg
from config import get_config
from models.MedT import MedT 
from models.FCN import FCN8s
from models.attention_Unet import AttUNet
from models.UNetPlusPlus import NestedUNet, U_Net, R2U_Net, AttU_Net, R2AttU_Net
from models.UNetPlusPlus import NestedUNet
from models.utnet import UTNet
from models.attunet import AttU_Net
from models.R2U_Net import R2U_Net
from models.R2AttU_Net import R2AttU_Net
from models.InfNet_Res2Net50 import Inf_Net
from models.PraNet_Res2Net50 import PraNet
from models.DANet import DANet
from models.EMANet import EMANet
from models.DenseASPP import DenseASPP
from models.CCNet import CCNet
from models.OCNet import OCNet
from models.PSANet import PSANet
from models.CaraNet import caranet
from models.multiresunet import MultiResUnet
from models.bisenetv2 import BiSeNetV2
from models.fpn import FPN
from models.cctnet.cctnet import CCTNet
from models.danet import get_danet
from models.davit import davit

from EdgeNeXt.models.model import edgenext_small
################################################


import ml_collections
import os
import wget 
# HiFormer-B Configs

# # HiFormer-S Configs
def get_hiformer_s_configs():
    
    cfg = ml_collections.ConfigDict()

    # Swin Transformer Configs
    cfg.swin_pyramid_fm = [96, 192, 384, 768]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 2
    if not os.path.isfile('./weights/swin_tiny_patch4_window7_224.pth'):
        print('Downloading Swin-transformer model ...')
        wget.download("https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth", "./weights/swin_tiny_patch4_window7_224.pth")    
    cfg.swin_pretrained_path = './weights/swin_tiny_patch4_window7_224.pth'

    # CNN Configs
    cfg.cnn_backbone = "resnet50"
    # cfg.cnn_pyramid_fm  = [64, 128, 256, 512] #resnet34
    cfg.cnn_pyramid_fm  = [256, 512, 1024, 2048]

    cfg.resnet_pretrained = True

    # DLF Configs
    cfg.depth = [[1, 1, 1, 1]]
    cfg.num_heads = (3,3,3,3)
    cfg.mlp_ratio=(1., 1., 1., 1)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg
# HiFormer-B Configs
# def get_hiformer_b_configs():

#     cfg = ml_collections.ConfigDict()
    
#     # Swin Transformer Configs
#     cfg.swin_pyramid_fm = [96, 192, 384]
#     cfg.image_size = 224
#     cfg.patch_size = 4
#     cfg.num_classes = 2
#     if not os.path.isfile('/workspace/ct/weights/cswin_base_224.pth'):
#         print('Downloading Swin-transformer model ...')
#         wget.download("https://ghproxy.com/https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth", "./weights/swin_tiny_patch4_window7_224.pth")    
#     cfg.swin_pretrained_path = '/workspace/ct/weights/cswin_base_224.pth'

#     # CNN Configs
#     cfg.cnn_backbone = "resnet50"
#     cfg.cnn_pyramid_fm  = [256, 512, 1024]
#     cfg.resnet_pretrained = True

#     # DLF Configs
#     cfg.depth = [[2, 4, 32]]
#     cfg.num_heads = (4, 8,16)
#     cfg.mlp_ratio=(2., 2., 1.)
#     cfg.drop_rate = 0.
#     cfg.attn_drop_rate = 0.
#     cfg.drop_path_rate = 0.
#     cfg.qkv_bias = True
#     cfg.qk_scale = None
#     cfg.cross_pos_embed = True

#     return cfg

#cswin config
# def get_hiformer_s_configs():
    
#     cfg = ml_collections.ConfigDict()

#     # Swin Transformer Configs
#     cfg.swin_pyramid_fm = [96, 192, 384, 768]

#     cfg.image_size = 224
#     cfg.patch_size = 4
#     cfg.num_classes = 2
#     if not os.path.isfile('./weights/cswin_tiny_224.pth'):
#         print('Downloading Swin-transformer model ...')
#         wget.download("https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth", "./weights/swin_tiny_patch4_window7_224.pth")    
#     cfg.swin_pretrained_path = './weights/cswin_tiny_224.pth'

#     # CNN Configs
#     cfg.cnn_backbone = "resnet50"
#     cfg.cnn_pyramid_fm  = [256, 512, 1024, 2048]
#     cfg.resnet_pretrained = True

#     # DLF Configs
#     cfg.depth = [[1,2,21,1]]
#     cfg.num_heads = (2,4,8,16)
#     cfg.mlp_ratio=(4., 4., 4., 4.)
#     cfg.drop_rate = 0.
#     cfg.attn_drop_rate = 0.
#     cfg.drop_path_rate = 0.
#     cfg.qkv_bias = True
#     cfg.qk_scale = None
#     cfg.cross_pos_embed = True

#     return cfg

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='../0.data/npz_h5_data/train_npz', help='root dir for data')
parser.add_argument('--valid_dataset_path', type=str,
                    default='../0.data/npz_h5_data/test_vol_h5', help='root dir for validation  data')
parser.add_argument('--dataset', type=str,
                    default='Osteosarcoma', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='./lists/lists_Osteosarcoma', help='list dir')
parser.add_argument('--num_classes', type=int,
                    default=2, help='output channel of network')
parser.add_argument('--output_dir', type=str,  default='./results', help='output dir')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum iterations number per epoch to train')
parser.add_argument('--max_epochs', type=int,
                    default=500, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=24, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--max_iou', type=float, default=0.00,
                    help='max iou')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--model', type=str, default='Swin_Unet', help='selected model for training')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--cfg', type=str, default='configs/swin_tiny_patch4_window7_224_lite.yaml', metavar="FILE", help='path to config file', )
parser.add_argument('--checkpoint_path', type=str, default=None, help='Get trained parameters')
parser.add_argument('--start_epoch', type=int, default=1, help='epoch at start')
parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')

parser.add_argument("--use_edge", type=int, default=1)
parser.add_argument("--use_mixup", type=int, default=0)
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument("--head", type=str, default='seghead')
parser.add_argument("--trans_cnn", type=str, nargs='+', default=['cswin_tiny', 'resnet50'], help='transformer, cnn')
parser.add_argument("--crop_size", type=int, nargs='+', default=[224, 224], help='H, W')

args = parser.parse_args()
config = get_config(args)


def net_factory(model,in_chns=3,class_num=2,img_size=224):

    if model == 'SegNet':
        model = SegNet(input_nbr=in_chns, label_nbr=class_num)

    elif model == 'UNet':
        model = Unet(in_channels=in_chns, classes=class_num)

    elif model == 'MiniSeg':
        model = MiniSeg(in_input=in_chns, classes=class_num)

    elif model == 'PSPNet':
        model = PSPNet(in_class=in_chns, n_classes=class_num)

    elif model == 'AttUNet':
        # ATTU_Net 512*512*1
        model = AttU_Net(img_ch=in_chns, output_ch=class_num)

    elif model == 'R2UNet':
        # R2U_Net 512*512*1
        model = R2U_Net(img_ch=in_chns, output_ch=class_num)

    elif model == 'R2AttUNet':
        # R2AttU_Net 265*256*1
        model = R2AttU_Net(img_ch=in_chns, output_ch=class_num)

    elif model == 'DeepLabv3':
        model = DeepLabv3_plus(nInputChannels=in_chns, n_classes=class_num)

    elif model == 'UNet3p':
        model = UNet_3Plus_DeepSup(in_channels=in_chns, n_classes=class_num)

    elif model == 'MulResUNet':
        model = MultiResUnet(channels=in_chns, nclasses=class_num)

    elif model == 'ENet':
        model = ENet(in_channels=in_chns, num_classes=class_num)

    elif model == 'GCN':
        model = GCN(in_channels=in_chns, num_classes=class_num)

    elif model == 'ResUNet':
        model = ResUnet(channel=in_chns, num_classes=class_num)

    elif model == 'ResUNetpp':
        model = ResUnetPlusPlus(channel=in_chns, num_classes=class_num)

    elif model == 'TransUNet':
        model = TransUnet(img_dim=img_size, in_channels=in_chns, classes=class_num, patch_size=1)

    elif model == 'SwinUNet':
        model = ViT_seg(config, num_classes=args.num_classes)

    elif model == 'MedT':
        model = MedT(img_size=img_size, num_classes=class_num, imgchan=in_chns)

    elif model == "nnUNet":
        model = initialize_network(num_classes=class_num).cuda()

    elif model == 'CaraNet':
        model = caranet(channel = in_chns, num_classes=class_num)

    elif model == 'PraNet':
        model = PraNet(num_classes=class_num)

    elif model == 'DANet':
        model = DANet(nclass=class_num)

    elif model == 'InfNet':
        model = Inf_Net(n_class=class_num)

    elif model == 'EMANet':
        model = EMANet(n_classes=class_num, n_layers=101)

    elif model == 'DenseASPP':
        model = DenseASPP(n_class=class_num)

    elif model == 'CCNet':
        model = CCNet(num_classes=class_num, recurrence=2)

    elif model == 'OCNet':
        model = OCNet(num_classes=class_num)

    elif model == 'PSANet':
        crop_h = crop_w = 880
        mask_h = 2 * ((crop_h - 1) // (8 * 2) + 1) - 1
        mask_w = 2 * ((crop_w - 1) // (8 * 2) + 1) - 1
        model = PSANet(classes=class_num, mask_h=mask_h, mask_w=mask_w)

    elif model == 'BiSeNetv2':
        model = BiSeNetV2(nclass=class_num)

    elif model == 'ViTAdapter':
        model = ViTAdapter()

    elif model == 'UNeXt':
        model = archs.__dict__['UNext'](2,3,False)
        
    elif model == 'Swin_Resnet':
        model = build_model('SwinRes_UperNet', num_classes=class_num, sync_bn=False)

    elif model == 'TransFuse_S':
        model = TransFuse_S(pretrained=True)

    elif model == 'TransMUNet':
        model = TransMUNet(n_classes = class_num)

    elif model == 'FCN':
        model =  FCN8s(input_channel=in_chns,n_class=class_num)

    elif model == 'FPN':
        model = FPN(nclass=class_num)

    elif model == 'UNet++':
        model =  NestedUNet(in_ch=3, out_ch=2)
 
    elif model == 'UTNet':
        model = UTNet(3, 32, 2, reduce_size=2,
                    block_list='1234', num_blocks=[1, 1, 1, 1], num_heads=[4, 4, 4, 4],
                    projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False,
                    maxpool=True)

    elif model == 'cctnet':
        model = CCTNet(transformer_name='swin_tiny', cnn_name='resnet50', nclass=class_num, img_size=img_size,
                       pretrained=True, aux=True, head=args.head, edge_aux=args.use_edge)
# 
        # model = CCTNet(transformer_name='swin_small', cnn_name='resnet50', nclass=class_num, img_size=img_size,
        #                pretrained=True, aux=True, head=args.head, edge_aux=args.use_edge)

        # model = CCTNet(transformer_name='swin_base', cnn_name='resnet50', nclass=class_num, img_size=img_size,
        #                pretrained=True, aux=True, head=args.head, edge_aux=args.use_edge)


    elif model == 'beit':
        from models.cctnet.beit import beit_base as beit
        model = beit(nclass=class_num, img_size=img_size, pretrained=True, aux=True, head=args.head, edge_aux=args.use_edge)
    elif model == 'cswin':
        from models.cctnet.cswin2 import cswin_tiny as cswin
        # from models.cctnet.cswin2 import cswin_base as cswin
        model = cswin(nclass=class_num, img_size=img_size, pretrained=True, aux=True, head=args.head, edge_aux=args.use_edge)
    elif model == 'volo':
        from  models.cctnet.volo import volo_d1 as volo
        model = volo(nclass=class_num, img_size=img_size, pretrained=True, aux=True, head=args.head, edge_aux=args.use_edge)
    elif model == 'banet':
        from  models.cctnet.banet import BANet
        model = BANet(nclass=class_num)
    elif model == 'segbase':
        from  models.cctnet.segbase import SegBase
        model = SegBase(nclass=class_num, backbone='resnet50', pretrained_base=True)
    elif model == 'swinT':
        # from  models.cctnet.swinseg import swin_tiny
        # model = swin_tiny(nclass=class_num, aux=True)
        from models.swin_resnet.swin_upernet import Swin_UperNet
        model = Swin_UperNet(num_classes=class_num, sync_bn=False)
    elif model == 'resT':
        from  models.cctnet.resT import rest_tiny as resT
        model = resT(nclass=class_num, pretrained=True, aux=True, head=args.head, edge_aux=args.use_edge)     

    elif model == 'edgenext':
        from timm.models import create_model
        model = create_model(
        'edgenext_small',
        pretrained=True,
        num_classes=class_num,
        drop_path_rate=0.1,
        layer_scale_init_value=1e-6,
        head_init_scale=1.0,
        input_res=224,
        classifier_dropout=0.0,
    )
    elif model=='parcnet':
        from ParC_Net.cvnets import get_model, EMA
        from ParC_Net.options.opts import get_training_arguments
        opts = get_training_arguments()

        model = get_model(opts)
    elif model == "cmt":
        from models.cmt import cmt_b
        model = cmt_b(pretrained=True)
    elif model == "convnext":
        from timm.models.convnext import convnext_base
        model = convnext_base(pretrained=True)

    elif model == "convmixer":
        from timm.models.convmixer import convmixer_768_32
        model = convmixer_768_32(pretrained=False,num_classes=class_num)

    elif model == "nextvit":
        from Next_ViT.next_vit import nextvit_small
        model = nextvit_small()

    elif model == "uniformer":
        from models.uniformer import uniformer_small
        model = uniformer_small(pretrained=True)
    elif model =='edgevit':
        from models.edgevit import edgevit_xs
        model = edgevit_xs(pretrained=True)
    elif model == "hiformer":
        from hiformer.HiFormer import HiFormer
        
        model = HiFormer(config=get_hiformer_s_configs(), img_size=img_size, n_classes=class_num)

    elif model == "poolformer":
        from models.poolformer import poolformer_m36
        
        model = poolformer_m36(pretrained=True)
    elif model == "resnet":
        from models.resnet import resnet50_v1b
        model = resnet50_v1b(pretrained=True)
    elif model == 'hrvit':
        from models.hrvit import HRViT_b2_224
        model = HRViT_b2_224(pretrained=False)
    elif model == 'scaleformer':
        from models.ScaleFormer import ScaleFormer
        model = ScaleFormer(n_classes=class_num)

    elif model =='ctformer_t':
        from ctformer.ctformer import CCTNet
        model = CCTNet(transformer_name='swin_tiny', cnn_name='resnet18', nclass=class_num, img_size=img_size,
                       pretrained=True, aux=True, head=args.head, edge_aux=args.use_edge)
    elif model =='ctformer_s':
        from ctformer.ctformer import CCTNet
        model = CCTNet(transformer_name='swin_tiny', cnn_name='resnet34', nclass=class_num, img_size=img_size,
                       pretrained=True, aux=True, head=args.head, edge_aux=args.use_edge)
    elif model =='ctformer_b':
        from ctformer.ctformer import CCTNet
        model = CCTNet(transformer_name='swin_tiny', cnn_name='resnet50', nclass=class_num, img_size=img_size,
                       pretrained=True, aux=True, head=args.head, edge_aux=args.use_edge)

        # model = CCTNet(transformer_name='swin_base', cnn_name='resnet50', nclass=class_num, img_size=img_size,
        #                pretrained=True, aux=True, head=args.head, edge_aux=args.use_edge)

    elif model == 'danet':
        model = get_danet()
    elif model == 'davit':
        model = davit(pretrained=True)

    else:
        model = None
        print('ERROR: No such model')

    return model

