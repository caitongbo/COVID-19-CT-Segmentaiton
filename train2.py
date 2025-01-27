import sys
sys.path.insert(0, '.')
import loadData as ld
import os
import os.path as osp
import torch
import pickle
from models.net_factory import net_factory

import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import transforms as myTransforms
import dataset as myDataLoader
import time
from argparse import ArgumentParser
from IOUEval import iouEval
import torch.optim.lr_scheduler
from collections import OrderedDict
from parallel import DataParallelModel, DataParallelCriterion
from torch.nn.parallel.scatter_gather import gather
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import CrossEntropyLoss
from util import DiceLoss
import numpy as np
import random

from config import dataset, model, img_height, img_width


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, ignore_label=255, loss_p=1):
        super(CrossEntropyLoss2d, self).__init__()

        self.loss = nn.NLLLoss(weight=weight, ignore_index=ignore_label)
        self.loss_p = loss_p

    def forward(self, *inputs):

        pred1, pred2, pred3, pred4, target = tuple(inputs)


        loss1 = self.loss(F.log_softmax(pred1, 1), target)
        loss2 = self.loss(F.log_softmax(pred2, 1), target)
        loss3 = self.loss(F.log_softmax(pred3, 1), target)
        loss4 = self.loss(F.log_softmax(pred4, 1), target)

        print(self.loss_p)

        return self.loss_p*(loss1 + loss2 + loss3 + loss4)

@torch.no_grad()
def val(args, val_loader, model, criterion):
    # switch to evaluation mode
    model.eval()
    iou_eval_val = iouEval(args.classes)
    epoch_loss = []

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(args.classes)

    total_batches = len(val_loader)
    for iter, (input, target) in enumerate(val_loader):
        start_time = time.time()

        if args.gpu:
            input = input.cuda()
            target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # run the mdoel
        output = model(input_var)

        torch.cuda.synchronize()
        time_taken = time.time() - start_time

        # loss_ce = ce_loss(output, target_var[:].long())
        # loss_dice = dice_loss(output, target_var, softmax=True)

        # loss = 0.5 * loss_ce + 0.5 * loss_dice
        # compute the loss
        if not args.gpu or torch.cuda.device_count() <= 1:
            pred1, pred2, pred3, pred4 = tuple(output)

            loss = criterion(pred1, pred2, pred3, pred4, target_var)#

        else:
            loss = criterion(output, target_var)
        epoch_loss.append(loss.data.item())

        # # compute the confusion matrix
        if args.gpu and torch.cuda.device_count() > 1:
            output = gather(output, 0, dim=0)[0]
        else:
            output = output[0]+output[1]
            
        iou_eval_val.add_batch(output.max(1)[1].data.cpu().numpy(), target_var.data.cpu().numpy())

        print('[%d/%d] loss: %.3f time: %.3f' % (iter, total_batches, loss.data.item(), time_taken))

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    overall_acc, per_class_acc, per_class_iu, mIOU = iou_eval_val.get_metric()

    return average_epoch_loss_val, overall_acc, per_class_acc, per_class_iu, mIOU


def train(args, train_loader, model, criterion, optimizer, epoch, max_batches, cur_iter=0):
    # switch to train mode
    model.train()
    iou_eval_train = iouEval(args.classes)
    epoch_loss = []

    # ce_loss = CrossEntropyLoss()
    # dice_loss = DiceLoss(args.classes)

    total_batches = len(train_loader)
    for iter, (input, target) in enumerate(train_loader):
        start_time = time.time()

        # adjust the learning rate
        lr = adjust_learning_rate(args, optimizer, epoch, iter + cur_iter, max_batches)

        if args.gpu == True:
            input = input.cuda()
            target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # run the mdoel
        output = model(input_var)

        # loss_ce = ce_loss(output, target_var[:].long())
        # loss_dice = dice_loss(output, target_var, softmax=True)

        # loss = 0.5 * loss_ce + 0.5 * loss_dice
        if not args.gpu or torch.cuda.device_count() <= 1:
            pred1, pred2, pred3, pred4 = tuple(output)
            loss = criterion(pred1, pred2, pred3, pred4, target_var)#
        else:
            loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.data.item())
        time_taken = time.time() - start_time

        # # compute the confusion matrix
        if args.gpu and torch.cuda.device_count() > 1:
            output = gather(output, 0, dim=0)[0]
        else:
            output = output[0]+output[1]

        iou_eval_train.add_batch(output.max(1)[1].data.cpu().numpy(), target_var.data.cpu().numpy())

        print('[%d/%d] lr: %.7f loss: %.3f time:%.3f' % (iter, total_batches, lr, loss.data.item(), time_taken))

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    overall_acc, per_class_acc, per_class_iu, mIOU = iou_eval_train.get_metric()

    return average_epoch_loss_train, overall_acc, per_class_acc, per_class_iu, mIOU, lr


def adjust_learning_rate(args, optimizer, epoch, iter, max_batches):
    if args.lr_mode == 'step':
        lr = args.lr * (0.5 ** (epoch // args.step_loss))
    elif args.lr_mode == 'poly':
        cur_iter = max_batches*epoch + iter
        max_iter = max_batches*args.max_epochs
        lr = args.lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def main_tr(args, crossVal):

    dataLoad = ld.LoadData(args.data_dir, args.classes)
    data = dataLoad.processData(crossVal, args.data_name)

    # load the model

    model = net_factory(model=args.model_name, in_chns=3, class_num=args.classes,img_size=args.width)

    if not osp.isdir(osp.join(args.savedir + '_mod'+ str(args.max_epochs))):
        os.mkdir(args.savedir + '_mod'+ str(args.max_epochs))
    if not osp.isdir(osp.join(args.savedir + '_mod'+ str(args.max_epochs), args.data_name)):
        os.mkdir(osp.join(args.savedir + '_mod'+ str(args.max_epochs), args.data_name))
    saveDir = args.savedir + '_mod' + str(args.max_epochs) + '/'+ args.data_name + '/' + args.model_name
    # create the directory if not exist
    if not osp.exists(saveDir):
        os.mkdir(saveDir)

    # define optimization criteria
    weight = torch.from_numpy(data['classWeights']) # convert the numpy array to torch
    if args.gpu:
        weight = weight.cuda()

    criteria = CrossEntropyLoss2d(weight, args.ignore_label, args.loss_p) #weight

    if args.gpu and torch.cuda.device_count() > 1 :
        criteria = DataParallelCriterion(criteria)
    if args.gpu:
        criteria = criteria.cuda()

    if args.gpu and torch.cuda.device_count() > 1:
        #model = torch.nn.DataParallel(model)
        model = DataParallelModel(model)
    if args.gpu:
        model = model.cuda()

    total_paramters = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters: ' + str(total_paramters))

    # compose the data with transforms
    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(args.width, args.height),
        myTransforms.RandomCropResize(int(32./1024.*args.width)),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor()
    ])
    trainDataset_scale1 = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(args.width, args.height),
        myTransforms.RandomCropResize(int(100./1024.*args.width)),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor()
    ])

    trainDataset_scale2 = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(args.width, args.height),
        myTransforms.RandomCropResize(int(100./1024.*args.width)),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor()
    ])
    trainDataset_scale3 = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(args.width, args.height),
        myTransforms.RandomCropResize(int(32./1024.*args.width)),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor()
    ])

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=data['mean'], std=data['std']),
        myTransforms.Scale(args.width, args.height),
        myTransforms.ToTensor()
    ])

    # since we training from scratch, we create data loaders at different scales
    # so that we can generate more augmented data and prevent the network from overfitting
    trainLoader = torch.utils.data.DataLoader(
        myDataLoader.Dataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_main),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    trainLoader_scale1 = torch.utils.data.DataLoader(
        myDataLoader.Dataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale1),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    trainLoader_scale2 = torch.utils.data.DataLoader(
        myDataLoader.Dataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale2),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    trainLoader_scale3 = torch.utils.data.DataLoader(
        myDataLoader.Dataset(data['trainIm'], data['trainAnnot'], transform=trainDataset_scale3),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    valLoader = torch.utils.data.DataLoader(
        myDataLoader.Dataset(data['valIm'], data['valAnnot'], transform=valDataset),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    max_batches = len(trainLoader) + len(trainLoader_scale1) + len(trainLoader_scale2) + len(trainLoader_scale3)

    if args.gpu:
        cudnn.benchmark = True

    start_epoch = 0

    if args.pretrained is not None:
        state_dict = torch.load(args.pretrained)
        new_keys = []
        new_values = []
        for idx, key in enumerate(state_dict.keys()):
            if 'pred' not in key:
                new_keys.append(key)
                new_values.append(list(state_dict.values())[idx])
        new_dict = OrderedDict(list(zip(new_keys, new_values)))
        model.load_state_dict(new_dict, strict=False)
        print('pretrained model loaded')

    if args.resume is not None:
        if osp.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            args.lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    log_file = osp.join(saveDir, 'trainValLog_'+args.model_name+'.txt')
    if osp.isfile(log_file):
        logger = open(log_file, 'a')
    else:
        logger = open(log_file, 'w')

        logger.write("%s\n" % (str(args)))

        logger.write("Parameters: %s" % (str(total_paramters)))
        logger.write("\n%s\t%s\t\t%s\t%s\t%s\t%s\tlr" % ('CrossVal', 'Epoch', 'Loss(Tr)', 'Loss(val)', 'mIOU (tr)', 'mIOU (val)'))
    logger.flush()

    optimizer = torch.optim.AdamW(model.parameters(), args.lr, (0.9, 0.999), eps=1e-8, weight_decay=0.01)
    maxmIOU = 0
    maxEpoch = 0
    print(args.model_name + '-CrossVal: '+str(crossVal+1))
    for epoch in range(start_epoch, args.max_epochs):
        # train for one epoch
        cur_iter = 0

        train(args, trainLoader_scale1, model, criteria, optimizer, epoch, max_batches, cur_iter)
        cur_iter += len(trainLoader_scale1)
        train(args, trainLoader_scale2, model, criteria, optimizer, epoch, max_batches, cur_iter)
        cur_iter += len(trainLoader_scale2)
        train(args, trainLoader_scale3, model, criteria, optimizer, epoch, max_batches, cur_iter)
        cur_iter += len(trainLoader_scale3)
        lossTr, overall_acc_tr, per_class_acc_tr, per_class_iu_tr, mIOU_tr, lr = \
                train(args, trainLoader, model,  criteria, optimizer, epoch, max_batches, cur_iter)

        # evaluate on validation set
        lossVal, overall_acc_val, per_class_acc_val, per_class_iu_val, mIOU_val = \
                val(args, valLoader, model, criteria)

        logger.write("\n%d\t\t%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.7f" % (crossVal+1, epoch + 1, lossTr, lossVal, mIOU_tr, mIOU_val, lr))
        logger.flush()
        print("\nEpoch No. %d:\tTrain Loss = %.4f\tVal Loss = %.4f\t mIOU(tr) = %.4f\t mIOU(val) = %.4f\n" \
                % (epoch + 1, lossTr, lossVal, mIOU_tr, mIOU_val))

        if mIOU_val >= maxmIOU:
            torch.save({
                'epoch': epoch + 1,
                'arch': str(model),
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lossTr': lossTr,
                'lossVal': lossVal,
                'iouTr': mIOU_tr,
                'iouVal': mIOU_val,
                'lr': lr
            }, osp.join(saveDir, 'checkpoint_' + args.model_name + '_crossVal' + str(crossVal+1) + '.pth.tar'))

            # save the model also
            model_file_name = osp.join(saveDir, 'model_' + args.model_name + '_crossVal' + str(crossVal+1) + '_' + 'best' + '.pth')
            torch.save(model.state_dict(), model_file_name)

            maxmIOU = mIOU_val
            maxEpoch = epoch + 1
        torch.cuda.empty_cache()
    logger.flush()
    logger.close()
    return maxEpoch, maxmIOU


def main(args):

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    crossVal = args.crossVal
    avgmIOU = 0
    strmIOU = ''

    saveDir = args.savedir+ '_mod' + str(args.max_epochs) + '/'+ args.data_name + '/' + args.model_name
    for i in range(crossVal):
        maxEpoch, maxmIOU = main_tr(args, i)
        avgmIOU = avgmIOU + maxmIOU/crossVal
        with open(osp.join(saveDir, 'modelBest_' + args.model_name + '.txt'), 'a+') as log:
            log.write("\n%s-CrossVal %d:\t maxEpoch: %d\t maxmIOU: %.4f" \
                    % (args.model_name, i + 1, maxEpoch, maxmIOU))
        strmIOU = strmIOU + str(maxEpoch) + ', '


    with open(osp.join(saveDir, 'modelBest_' + args.model_name + '.txt'), 'a+') as log:
        log.write("\n\navgmIOU: %.4f" % (avgmIOU))
        log.write("\n%s" % (strmIOU))

    print(args.model_name, args.data_name, avgmIOU)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default="./datasets", help='Data directory')
    parser.add_argument('--width', type=int, default=224, help='Width of RGB image')
    parser.add_argument('--height', type=int, default=224, help='Height of RGB image')
    parser.add_argument('--max_epochs', type=int, default=50, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=16, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy, step or poly')
    parser.add_argument('--savedir', default='./results_MiniSeg_crossVal', help='Directory to save the results')
    parser.add_argument('--resume', default='', help='Use this checkpoint to continue training')
    parser.add_argument('--pretrained', default=None, help='Use this pretrained model for initialization')
    parser.add_argument('--classes', type=int, default=2, help='No. of classes in the dataset')
    parser.add_argument('--ignore_label', type=int, default=255, help = "ignored label")
    parser.add_argument('--model_name', default='UNet++', help='Model name')
    parser.add_argument('--data_name', default='Inf', help='Model name')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPssU.')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--seed', type=int,  default=1337, help='random seed')
    parser.add_argument('--crossVal', type=int,  default=1, help='random seed')
    parser.add_argument('--loss_p', type=int,  default=1, help='random seed')




    args = parser.parse_args()

#     args.data_name = 'P1110'
#     args.width = 224
#     args.height = 224
# # 'cctnet','TransUNet','ViTAdapter','UNeXt','FCN','UTNet','InfNet','MedT','SegNet','UNet','SwinUNet','UNet++','MiniSeg','PSPNet','AttUNet','DeepLabv3',
#     # models = ['nnUNet','PSANet','BiSeNetv2','FPN']
#     # 'ENet','GCN','ResUNet','ResUNetpp','DANet','PraNet','EMANet','DenseASPP','CaraNet','cswin','volo','resT','banet','segbase','R2UNet','R2AttUNet','UNet3p'
#     # models = ['CCNet','OCNet']
#     # models = ['InfNet']
#     # models = ["edgenext"]
#     # models = ["parcnet"]
#     # models = ["cmt"]
#     # models = ["convmixer"]
#     # models = ["nextvit"]
#     # models = ['uniformer']
#     # models = ['edgevit']
#     # models =  ['hiformer']
#     # models =  ['poolformer']
#     # models =  ['cswin']
#     models =  ['ctformer']
#     # models =  ['MiniSeg']
#     # models =  ['resnet']
#     # models =  ['hrvit']
#     # models =  ['scaleformer']
#     # models =  ['swinT']

#     for model in models:
#         args.model_name = model
#         if model =='nnUNet' or model =='UNet3p':
#             args.batch_size = 4
#         print('Called with args:')
#         print(args)
#         main(args)


#     args.data_name = 'P20'
#     models =  ['UNet']
#     for model in models:
#         args.model_name = model
#         if model =='nnUNet' or model =='UNet3p':
#             args.batch_size = 4
#         print('Called with args:')
#         print(args)
#         main(args)

    # args.data_name = 'P9'
    # models =  ['ctformer']
    # for model in models:
    #     args.model_name = model
    #     if model =='nnUNet' or model =='UNet3p':
    #         args.batch_size = 4
    #     print('Called with args:')
    #     print(args)
    #     main(args)

    args.crossVal=1
    # args.data_name = 'P1110'
    # args.data_name = 'ISIC2018'
    # args.data_name = 'CVC'
    args.data_name = 'Inf'

    
    # models =  ['UNet']
    # models =  ['cswin']
    # models =  ['ctformer_s','ctformer_b']
    models =  ['ctformer_t']

    loss_ps = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


    for loss_p in loss_ps:
        args.loss_p = loss_p
        for model in models:
            args.model_name = model
            if model =='nnUNet' or model =='UNet3p':
                args.batch_size = 4
            print('Called with args:')
            print(args)
            main(args)