#!/usr/bin/env python

import os
import cv2
import numpy as np
import os.path as osp
import SimpleITK as sitk
from PIL import Image
from argparse import ArgumentParser
from config import dataset, model

from medpy import metric
import torch


def MAE(pred, gt):
	mae = torch.abs(pred - gt).mean()
	return mae

def compute_specificity(SEG, GT):
    TN = np.sum(np.logical_not(np.logical_or(SEG, GT)))
    FP = np.sum(SEG) - np.sum(np.logical_and(SEG, GT))
    spec = TN / (TN + FP)
    return spec


def evaluation_sample(SEG_np, GT_np):
    quality=dict()
    SEG_np = np.uint8(np.where(SEG_np, 1, 0))
    GT_np = np.uint8(np.where(GT_np, 1, 0))
    SEG = sitk.GetImageFromArray(SEG_np)
    GT = sitk.GetImageFromArray(GT_np)

    # Compute the evaluation criteria
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    # Overlap measures
    overlap_measures_filter.Execute(SEG, GT)
    #quality["jaccard"] = overlap_measures_filter.GetJaccardCoefficient()
    quality["dice"] = overlap_measures_filter.GetDiceCoefficient()
    quality["false_negative"] = overlap_measures_filter.GetFalseNegativeError()
    #quality["false_positive"] = overlap_measures_filter.GetFalsePositiveError()
    quality["sensitive"] = 1 - quality["false_negative"]
    quality["specificity"] = compute_specificity(SEG_np, GT_np)

    # Hausdorff distance
    hausdorff_distance_filter.Execute(SEG, GT)
    quality["hausdorff_distance"] = hausdorff_distance_filter.GetHausdorffDistance()

    return quality

#######################################################
#######################################################

def main(args):
    crossVal = args.crossVal
    avgmIOU = 0
    avgSEN = 0
    avgSPC = 0
    avgF = 0
    avgDSC = 0
    avgHD = 0
    avgMAE = 0
    data_root = args.data_root
    data_name = args.data_name
    model_name = args.model_name
    # UNeXt Swin_Resnet UNet TransUNet MiniSeg SwinUNet nnUNet 
    print(model_name, data_name)

    for i in range(crossVal):
        image_files = list()
        gt_files = list()
        pred_files = list()
        valFile = osp.join(data_root, 'COVID-19-' + data_name + '/dataList/'+'val'+str(i)+'.txt')
        pred_data_root = osp.join('./outputs/', data_name, model_name, 'crossVal'+str(i))
        with open(valFile) as text_file:
            for line in text_file:
                line_arr = line.split()
                image_files.append(osp.join(data_root, line_arr[0].strip()))
                gt_files.append(osp.join(data_root, line_arr[1].strip()))
                pred_files.append(osp.join(pred_data_root, line_arr[0].split('/')[-1].strip()))

        assert len(gt_files) == len(pred_files), 'The number of GT and pred must be equal'

        EPSILON = np.finfo(np.float64).eps

        recall = np.zeros((len(gt_files)))
        precision = np.zeros((len(gt_files)))
        dice = np.zeros((len(gt_files)))
        sensitive = np.zeros((len(gt_files)))
        specificity = np.zeros((len(gt_files)))
        hausdorff_distance = np.zeros((len(gt_files)))

        mae = np.zeros((len(gt_files)))

        print('crossVal sensitive specificity F1 dice HD MAE')
        for idx in range(len(gt_files)):
            gt = cv2.imread(gt_files[idx], 0)
            # pred = cv2.imread(pred_files[idx], 0) 
            pred = cv2.imread(pred_files[idx].replace('.jpg','.png'), 0) # inf-net datasets

            pred = pred.astype(np.float64) / 255
            if not gt.shape == (224, 224):
                gt = cv2.resize(gt, (224, 224), interpolation=cv2.INTER_NEAREST)
            if not pred.shape == (224, 224):
                pred = cv2.resize(pred, (224, 224), interpolation=cv2.INTER_NEAREST)
            gt = gt == 255

            zeros = 0
            zeros_pred = []

            if np.sum(pred) != 0:
                intersection = np.sum(np.logical_and(gt == pred, gt))
                recall[idx] = intersection * 1. / (np.sum(gt) + EPSILON)
                precision[idx] = intersection * 1. / (np.sum(pred) + EPSILON)
                dice[idx] = evaluation_sample(pred, gt).get("dice")
                sensitive[idx] = evaluation_sample(pred, gt).get("sensitive")
                specificity[idx] = evaluation_sample(pred, gt).get("specificity")
                hausdorff_distance[idx] = evaluation_sample(pred, gt).get("hausdorff_distance")
                mae[idx] = np.sum(np.fabs(gt - pred)) * 1. / (gt.shape[0] * gt.shape[1])


        recall = np.mean(recall, axis=0)
        precision = np.mean(precision, axis=0)
        dice = np.max(np.mean(dice, axis=0))
        sensitive = np.max(np.mean(sensitive, axis=0))
        specificity = np.max(np.mean(specificity, axis=0))
        hausdorff_distance = np.max(np.mean(hausdorff_distance, axis=0))
        mae = np.max(np.mean(mae, axis=0))
        F_beta = (1 + 0.3) * precision * recall / (0.3 * precision + recall + EPSILON)
        print(i,"{:.4f}".format(sensitive),"{:.4f}".format(specificity), "{:.4f}".format(F_beta),"{:.4f}".format(dice),"{:.4f}".format(hausdorff_distance), "{:.4f}".format(mae))

        avgSEN = avgSEN + sensitive/crossVal
        avgSPC = avgSPC + specificity/crossVal
        avgF = avgF + F_beta/crossVal
        avgDSC = avgDSC + dice/crossVal
        avgHD = avgHD + hausdorff_distance/crossVal
        avgMAE = avgMAE + mae/crossVal




    log_file = osp.join(args.pretrained + args.data_name + '/' +'all_metrix_result.txt')
    if osp.isfile(log_file):
        logger = open(log_file, 'a')
    else:
        logger = open(log_file, 'w')
        logger.write('model ' + ' sensitive '+ ' specificity ' + ' F1 ' + ' dice '+ ' HD' + ' MAE \n')
    logger.write(args.model_name + ' ' + "{:.4f}".format(avgSEN) + ' ' + "{:.4f}".format(avgSPC) + ' ' + "{:.4f}".format(avgF) + ' ' + "{:.4f}".format(avgDSC) + ' ' + "{:.2f}".format(avgHD) + ' ' + "{:.4f}".format(avgMAE) +'\n')

    print("{:.4f}".format(avgSEN))
    print("{:.4f}".format(avgSPC))
    print("{:.4f}".format(avgF))
    print("{:.4f}".format(avgDSC))
    print("{:.2f}".format(avgHD))
    print("{:.4f}".format(avgMAE))



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_root', default="./datasets", help='Data directory')
    parser.add_argument('--model_name', default='ViTAdapter', help='Model name')
    parser.add_argument('--data_name', default='Inf', help='Model name')
    parser.add_argument('--pretrained', default='./results_MiniSeg_crossVal_mod100/', help='Pretrained model')
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
    # models = ['MiniSeg','cswin',"nextvit",'poolformer','InfNet','UNet','volo','DeepLabv3','PSPNet','FCN','TransUNet','UNeXt','SwinUNet','SegNet','resnet']

    # models =['DeepLabv3','PSPNet','FCN','MiniSeg','SwinUNet','resnet','swinT','nextvit','SegNet','UNeXt','UNet','TransUNet','InfNet','volo','poolformer','cswin','ctformer_t','ctformer_s','ctformer_b']
    datasets = ['Inf', 'P9', 'P20', 'P1110', 'CVC', 'ISIC2018'] 
    # datasets = ['P9'] 

    # models = ["danet"]
    models = ["davit"]


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