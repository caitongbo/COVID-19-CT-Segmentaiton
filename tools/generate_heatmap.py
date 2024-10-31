from pyexpat import model
import cv2
import os
import sys
# sys.path.append('../')
import numpy as np
import torch
import tifffile as tiff
from PIL import Image
# from utils import label_to_RGB
from torchvision import transforms
from models.net_factory import net_factory
from argparse import ArgumentParser

def label_to_RGB(image):
    RGB = np.zeros(shape=[image.shape[0], image.shape[1], 3], dtype=np.uint8)
    index = image == 0
    RGB[index] = np.array([255, 255, 255])
    index = image == 1
    RGB[index] = np.array([0, 0, 255])
    index = image == 2
    RGB[index] = np.array([0, 255, 255])
    index = image == 3
    RGB[index] = np.array([0, 255, 0])
    index = image == 4
    RGB[index] = np.array([255, 255, 0])
    index = image == 5
    RGB[index] = np.array([255, 0, 0])
    return RGB


def init_model(args):
    model = net_factory(model=args.model_name, in_chns=3, class_num=args.classes,img_size=args.width)

    pretrained = '/root/workspace/data/ctb/COVID-19-CT/semi_ct/results_MiniSeg_crossVal_mod300/Inf/ctformer_t/model_ctformer_t_crossVal1_best.pth'

    state_dict = torch.load(pretrained)
    model.load_state_dict(state_dict,True)

    return model


def read_img(save_dir):
    img_dir = '/root/workspace/data/ctb/COVID-19-CT/semi_ct/datasets/COVID-19-Inf/TestingSet/LungInfection-Test/Imgs/6.jpg'
    # image = tiff.imread(img_dir)
    image = Image.open(img_dir)
    image = np.array(image)
    image = image[1000:1000 + 512, 0:0+512, 0:3]
    cv2.imwrite(os.path.join(save_dir, 'ori_image.png'), image[..., ::-1])

    return image
    
    
def read_label(save_dir):
    img_dir = '/root/workspace/data/ctb/COVID-19-CT/semi_ct/datasets/COVID-19-Inf/TestingSet/LungInfection-Test/GT/6.png'
    image = Image.open(img_dir)
    image = np.array(image)
    image = image[1000:1000 + 512, 0:0+512, 0:3]
    cv2.imwrite(os.path.join(save_dir, 'ori_label.png'), image[..., ::-1])

    return image


def to_tensor(image):
    image = torch.from_numpy(image).permute(2, 0, 1).float().div(255)
    normalize = transforms.Normalize((.485, .456, .406), (.229, .224, .225))
    image = normalize(image).unsqueeze(0)

    return image


def main(args):
    save_img_dir = os.path.join(save_path, 'origin_img')
    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)
    save_out_dir = os.path.join(save_path, 'output')
    if not os.path.exists(save_out_dir):
        os.mkdir(save_out_dir)

    image = read_img(save_img_dir)
    image = to_tensor(image).cuda()
    model = init_model(args).cuda().eval()
    with torch.no_grad():
        output = model(image)
    output = torch.argmax(output[0], dim=1)
    output = output.squeeze()
    output = output.cpu().numpy()
    output = output.astype(np.uint8)
    output = label_to_RGB(output)
    cv2.imwrite(os.path.join(save_out_dir, 'out.png'), output[..., ::-1])


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--data_dir', default="./datasets", help='Data directory')
    parser.add_argument('--width', type=int, default=224, help='Width of RGB image')
    parser.add_argument('--height', type=int, default=224, help='Height of RGB image')
    parser.add_argument('--savedir', default='./outputs', help='directory to save the results')
    parser.add_argument('--model_name', default='ViTAdapter', help='Model name')
    parser.add_argument('--data_name', default='CT100', help='Model name')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU')
    parser.add_argument('--pretrained', default='./results_MiniSeg_crossVal_mod300/', help='Pretrained model')
    parser.add_argument('--classes', default=2, type=int, help='Number of classes in the dataset')
    parser.add_argument('--crossVal', type=int,  default=1, help='random seed')

    args = parser.parse_args()

    args.data_name = 'Inf'
    args.width = 224
    args.height = 224
# ['cctnet','TransUNet','ViTAdapter','UNeXt','FCN','UTNet','InfNet','MedT','SegNet','UNet','SwinUNet','UNet++','MiniSeg','PSPNet','AttUNet','DeepLabv3','ENet','GCN','ResUNet','ResUNetpp','DANet','PraNet','EMANet','DenseASPP','CaraNet','cswin','volo','resT','banet','segbase','R2UNet','R2AttUNet','UNet3p', 'nnUNet','PSANet','BiSeNetv2','FPN']
    # models = ['CCNet','OCNet','ViTAdapter']
    # models =['cctnet','TransUNet','UNeXt','FCN','UTNet','InfNet','MedT','SegNet','UNet','SwinUNet','UNet++','MiniSeg','PSPNet','AttUNet','DeepLabv3','ENet','GCN','ResUNet','ResUNetpp','DANet','PraNet','EMANet','DenseASPP','CaraNet','cswin','volo','resT','banet','segbase','R2UNet','R2AttUNet','UNet3p', 'nnUNet','PSANet','BiSeNetv2','FPN']
    # models = ['ctformer']

    # models = ['MiniSeg','cswin','nextvit','poolformer','InfNet','UNet','volo','DeepLabv3','PSPNet','FCN','TransUNet','UNeXt','SwinUNet','SegNet','resnet','swinT']
    models =['ctformer']
    for name in models:
        args.model_name = name

    save_path = './tools/heatmap/outputs/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    main(args)
    read_img(save_path)
    read_label(save_path)

