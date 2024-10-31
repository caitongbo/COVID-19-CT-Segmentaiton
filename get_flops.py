# from models.cctnet.cctnet import CCTNet
from ctformer.ctformer import CCTNet

from hiformer_swin.configs.HiFormer_configs import  get_hiformer_s_configs

import ml_collections
import os
import wget 
from mmcv.cnn import get_model_complexity_info

def get_hiformer_s_configs():
    
    cfg = ml_collections.ConfigDict()

    # Swin Transformer Configs
    cfg.swin_pyramid_fm = [96, 192, 384, 768]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 2
    if not os.path.isfile('/workspace/ct/weights/swin_tiny_patch4_window7_224.pth'):
        print('Downloading Swin-transformer model ...')
        wget.download("https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth", "./weights/swin_tiny_patch4_window7_224.pth")    
    cfg.swin_pretrained_path = '/workspace/ct/weights/swin_tiny_patch4_window7_224.pth'

    # CNN Configs
    cfg.cnn_backbone = "resnet50"
    # cfg.cnn_pyramid_fm  = [64, 128, 256, 512] #resnet34
    cfg.cnn_pyramid_fm  = [256, 512, 1024, 2048]

    cfg.resnet_pretrained = True

    # DLF Configs
    cfg.depth = [[1, 1, 1, 1]]
    cfg.num_heads = (2,4,8,16)
    cfg.mlp_ratio=(1., 1., 1., 1)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg

if __name__ == '__main__':
    """Notice if torch1.6, try to replace a / b with torch.true_divide(a, b)"""
    from tools.flops_params_fps_count import flops_params_fps

    model = CCTNet(transformer_name='swin_tiny', cnn_name='resnet18', nclass=2, img_size=224, aux=True,
                   edge_aux=True, head='seghead', pretrained=True)

    # from models.cctnet.cctnet_cswin import CCTNet
    # model = CCTNet(transformer_name='cswin_tiny', cnn_name='resnet50', nclass=2, img_size=224, aux=False,
    #                edge_aux=False, head='seghead', pretrained=True)

    # from models.resnet import resnet50_v1b
    # model = resnet50_v1b(pretrained=True)

    # from models.swin_resnet.swin_upernet import Swin_UperNet
    # model = Swin_UperNet(num_classes=2, sync_bn=False)

    # from  ctformer.swinT import swin_tiny
    # model = swin_tiny(nclass=2, aux=True)


    # from hiformer.HiFormer import HiFormer
    # model = HiFormer(config=get_hiformer_s_configs(), img_size=224, n_classes=2)

    # from hiformer_swin.HiFormer import HiFormer
    # model = HiFormer(config=get_hiformer_s_configs(), img_size=224, n_classes=2)
    # from Next_ViT.next_vit import nextvit_small

    # model = nextvit_small()

    flops, params = get_model_complexity_info(model, (3,224,224))
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {(3,224,224)}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')
