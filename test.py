import torch
import cv2
import time
import os
import os.path as osp
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from argparse import ArgumentParser
from collections import OrderedDict
from IOUEval import iouEval
from PIL import Image
import loadData as ld

from models.net_factory import net_factory
from utils.model_del import get_model
from utils.model_del import remove_model

from config import dataset, model, img_height, img_width
from tools.flops_params_fps_count import flops_params_fps
from mmcv.cnn import get_model_complexity_info


@torch.no_grad()
def validate(args, model, image_list, label_list, crossVal, mean, std,flops,params):
    iou_eval_val = iouEval(args.classes)
    for idx in range(len(image_list)):
        image = cv2.imread(image_list[idx]) / 255
        image = image[:, :, ::-1]
        label = cv2.imread(label_list[idx], 0) / 255

        img = image.astype(np.float32)
        img = ((img - mean) / std).astype(np.float32)
        img = cv2.resize(img, (args.height, args.width))
        img = img.transpose((2, 0, 1))
        img_variable = Variable(torch.from_numpy(img).unsqueeze(0))
        if args.gpu:
            img_variable = img_variable.cuda()

        start_time = time.time()

        if args.model_name =='ctformer_t' or args.model_name =='ctformer_s' or args.model_name =='ctformer_b':
            img_out = model(img_variable)[0]+model(img_variable)[1]
        else:
            img_out = model(img_variable)

        torch.cuda.synchronize()
        diff_time = time.time() - start_time
        print('Segmentation for {}/{} takes {:.3f}s per image'.format(idx, len(image_list), diff_time))

        class_numpy = img_out[0].max(0)[1].data.cpu().numpy()
        label = cv2.resize(label, (args.height, args.width), interpolation=cv2.INTER_NEAREST)
        iou_eval_val.add_batch(class_numpy, label)
        overall_acc, per_class_acc, per_class_iu, mIOU = iou_eval_val.get_metric()
        # print(mIOU)

        out_numpy = (class_numpy * 255).astype(np.uint8)
        name = image_list[idx].split('/')[-1]
        if not osp.isdir(osp.join(args.savedir, args.data_name)):
            os.mkdir(osp.join(args.savedir, args.data_name))
        if not osp.isdir(osp.join(args.savedir, args.data_name, args.model_name)):
            os.mkdir(osp.join(args.savedir, args.data_name, args.model_name))
        if not osp.isdir(osp.join(args.savedir, args.data_name, args.model_name, 'crossVal'+str(crossVal))):
            os.mkdir(osp.join(args.savedir, args.data_name, args.model_name, 'crossVal'+str(crossVal)))
        cv2.imwrite(osp.join(args.savedir, args.data_name, args.model_name, 'crossVal'+str(crossVal), name[:-4] + '.png'), out_numpy)
        # out_label = (label * 255).astype(np.uint8)
        
        # cv2.imwrite(osp.join(args.savedir, args.data_name, 'gt', name[:-4] + '.png'), out_label)


    overall_acc, per_class_acc, per_class_iu, mIOU = iou_eval_val.get_metric()
    print('Overall Acc (Val): %.4f\t mIOU (Val): %.4f' % (overall_acc, mIOU))

    log_file = osp.join(args.pretrained + args.data_name + '/' +'all_mIOU_result.txt')
    if osp.isfile(log_file):
        logger = open(log_file, 'a')
    else:
        logger = open(log_file, 'w')
        logger.write('model '+ ' ' + 'flops' + ' ' + 'params' + ' ' + 'mIOU\n')
    logger.write(args.model_name + ' ' + f'{flops}'+ ' ' + f'{params}'+ ' ' +  "{:.4f}".format(mIOU)+'\n')

    return mIOU


def main_te(args, crossVal, pretrained, mean, std):
    # read the image list
    image_list = list()
    label_list = list()
    with open(osp.join(args.data_dir, 'COVID-19-' + args.data_name + '/dataList/'+'val'+str(crossVal)+'.txt')) as text_file:
    # with open("/root/workspace/data/ctb/COVID-19-CT/semi_ct/datasets/COVID-19-Inf/dataList/psdata.txt") as text_file:
        for line in text_file:
            line_arr = line.split()
            image_list.append(osp.join(args.data_dir, line_arr[0].strip()))
            label_list.append(osp.join(args.data_dir, line_arr[1].strip()))

    # load the model
    model = net_factory(model=args.model_name, in_chns=3, class_num=args.classes,img_size=args.width)

    flops, params = get_model_complexity_info(model, (3,224,224))


    if not osp.isfile(pretrained):
        print('Pre-trained model file does not exist...')
        exit(-1)
    state_dict = torch.load(pretrained)
    # new_keys = []
    # new_values = []
    # for key, value in zip(state_dict.keys(), state_dict.values()):
    #     if 'pred' not in key or 'pred1' in key:
    #         new_keys.append(key)
    #         new_values.append(value)
    # new_dict = OrderedDict(list(zip(new_keys, new_values)))
    # model.load_state_dict(new_dict)
    model.load_state_dict(state_dict,True)

    if args.gpu:
        model = model.cuda()

    # set to evaluation mode
    model.eval()

    if not osp.isdir(args.savedir):
        os.mkdir(args.savedir)

    mIOU = validate(args, model, image_list, label_list, crossVal, mean, std,flops,params)
    return mIOU


def main(args):
    crossVal = args.crossVal
    # maxEpoch = [80, 80, 80, 80, 80]
    mIOUList = []
    avgmIOU = 0

    path = args.pretrained + args.data_name + '/' + args.model_name + '/'
    maxEpoch = get_model(path, args.model_name,crossVal)

    # remove_model(path, args.model_name,crossVal)

    for i in range(crossVal):
        dataLoad = ld.LoadData(args.data_dir, args.classes)
        data = dataLoad.processData(i, args.data_name)
        mean = data['mean']#np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = data['std']#np.array([0.229, 0.224, 0.225], dtype=np.float32)

        print('Data statistics:')
        print(mean, std)

        pthName = 'model_' + args.model_name + '_crossVal' + str(i+1) + '_' + 'best' + '.pth'
        print(pthName)
        pretrainedModel = args.pretrained + args.data_name + '/' + args.model_name + '/' + pthName
        mIOU = "{:.4f}".format(main_te(args, i, pretrainedModel, mean, std))
        mIOU = float(mIOU)
        mIOUList.append(mIOU)
        avgmIOU = avgmIOU + mIOU/crossVal
    print(mIOUList)
    print(args.model_name, args.data_name, "{:.4f}".format(avgmIOU))


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
    parser.add_argument('--pretrained', default='./results_MiniSeg_crossVal_mod100/', help='Pretrained model')
    parser.add_argument('--classes', default=2, type=int, help='Number of classes in the dataset')
    parser.add_argument('--crossVal', type=int,  default=1, help='random seed')

    args = parser.parse_args()

    # args.data_name = 'Inf'
    # args.data_name = 'CVC'
    args.data_name = 'ISIC2018'


    args.width = 224
    args.height = 224
# ['cctnet','TransUNet','ViTAdapter','UNeXt','FCN','UTNet','InfNet','MedT','SegNet','UNet','SwinUNet','UNet++','MiniSeg','PSPNet','AttUNet','DeepLabv3','ENet','GCN','ResUNet','ResUNetpp','DANet','PraNet','EMANet','DenseASPP','CaraNet','cswin','volo','resT','banet','segbase','R2UNet','R2AttUNet','UNet3p', 'nnUNet','PSANet','BiSeNetv2','FPN']
    # models = ['CCNet','OCNet','ViTAdapter']
    # models =['cctnet','TransUNet','UNeXt','FCN','UTNet','InfNet','MedT','SegNet','UNet','SwinUNet','UNet++','MiniSeg','PSPNet','AttUNet','DeepLabv3','ENet','GCN','ResUNet','ResUNetpp','DANet','PraNet','EMANet','DenseASPP','CaraNet','cswin','volo','resT','banet','segbase','R2UNet','R2AttUNet','UNet3p', 'nnUNet','PSANet','BiSeNetv2','FPN']
    models = ['cswin']

    # models = ['MiniSeg','cswin','nextvit','poolformer','InfNet','UNet','volo','DeepLabv3','PSPNet','FCN','TransUNet','UNeXt','SwinUNet','SegNet','resnet','swinT','ctformer_t','ctformer_s','ctformer_b']
    datasets = ['Inf', 'P9', 'P20', 'P1110', 'CVC', 'ISIC2018'] 
    # models = ["danet",'davit']

    # models = ['davit']
    models = ['MiniSeg','cswin','nextvit','poolformer','InfNet','UNet','volo','DeepLabv3','PSPNet','FCN','TransUNet','UNeXt','SwinUNet','SegNet','resnet','swinT','ctformer_t','ctformer_s','ctformer_b']



    for dataset in datasets:
        args.data_name = dataset
        for model in models:
            args.model_name = model
            if model =='nnUNet' or model =='UNet3p':
                args.batch_size = 4
            if model =='UNet++' or model =='UNet3p':
                args.batch_size = 16
            print('Called with args:')
            print(args)
            main(args)



