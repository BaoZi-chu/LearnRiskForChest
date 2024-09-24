import os

import numpy as np
import time
import sys

import torch
torch.__version__
    
from ChexnetTrainer_new import ChexnetTrainer


#--------------------------------------------------------------------------------   

def runTrain():
    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'
    RSENET50='RESNET-50'

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    # ---- Path to the directory with images
    pathImgTrain = '/home/4t/lfy/datasets/get_distribution_hosp_new/train'
    pathImgVal = '/home/4t/lfy/datasets/get_distribution_hosp_new/val'
    pathImgTest = '/home/4t/lfy/datasets/get_distribution_hosp_new/test'

    # ---- Neural network parameters: type of the network, is it pre-trained
    # ---- on imagenet, number of classes
    nnArchitecture = RSENET50
    nnIsTrained = True
    nnClassCount = 2  # 14

    # ---- Training settings: batch size, maximum number of epochs
    trBatchSize = 8
    trMaxEpoch =40

    # ---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 256
    imgtransCrop = 224

    pathModel = 'm-' + timestampLaunch + '.pth.tar'

    print('=== Training NN architecture = ', nnArchitecture, '===')
    ChexnetTrainer.train(pathImgTrain, pathImgVal, pathImgTest, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize,
                         trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, 10, 'chest_hosp', '/home/ssd0/lfy/result_archive/chest_r50_best/max_acc_97.44.pth',None)

    # print('=== Testing the trained model ===')
    # ChexnetTrainer.test(pathImgTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize,
    #                     imgtransCrop, timestampLaunch)


# --------------------------------------------------------------------------------

def runFinetune():
    DENSENET121 = 'DENSE-NET-121'
    DENSENET169 = 'DENSE-NET-169'
    DENSENET201 = 'DENSE-NET-201'

    timestampTime = time.strftime("%H%M%S")
    timestampDate = time.strftime("%d%m%Y")
    timestampLaunch = timestampDate + '-' + timestampTime

    # ---- Path to the directory with images
    pathImgTrain = '/home/ssd0/lfy/datasets/get_distribution_hosp/val'
    pathImgVal = '/home/ssd0/lfy/datasets/get_distribution_hosp/test'
    pathImgTest = '/home/ssd0/lfy/datasets/get_distribution_hosp/test'

    # ---- Neural network parameters: type of the network, is it pre-trained
    # ---- on imagenet, number of classes
    # checkpoint = '/home/ssd0/lfy/gml_project/test/chexnet1/m-19042022-110254.pth.tar'
    nnArchitecture = DENSENET121
    nnIsTrained = True
    nnClassCount = 1  # 14

    # ---- Training settings: batch size, maximum number of epochs
    trBatchSize = 16
    trMaxEpoch = 100

    # ---- Parameters related to image transforms: size of the down-scaled image, cropped image
    imgtransResize = 256
    imgtransCrop = 224

    pathModel = 'finetune-' + timestampLaunch + '.pth.tar'

    print('=== Training NN architecture = ', nnArchitecture, '===')
    ChexnetTrainer.finetune(pathImgTrain, pathImgVal, pathImgTest, nnArchitecture, nnIsTrained, nnClassCount,
                            trBatchSize,
                            trMaxEpoch, imgtransResize, imgtransCrop, timestampLaunch, checkpoint)

    print('=== Testing the trained model ===')
    ChexnetTrainer.test(pathImgTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize,
                        imgtransCrop, timestampLaunch)


# --------------------------------------------------------------------------------

def runTest():
    pathImgTest = '/home/4t/lfy/datasets/get_distribution_hosp_new/test'
    nnArchitecture = 'DENSE-NET-121'
    nnIsTrained = True
    nnClassCount = 1  # 14
    trBatchSize = 1
    imgtransResize = 256
    imgtransCrop = 224

    pathModel = '/home/ssd0/lfy/result_archive/hosp_r50_best/max_acc_77.37.pth'

    timestampLaunch = ''

    ChexnetTrainer.test(pathImgTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, imgtransResize,
                        imgtransCrop, timestampLaunch)


# --------------------------------------------------------------------------------

if __name__ == '__main__':
 runTrain()  # 在 chest_xray 的 trainval 上预训练
 # runFinetune()  # 在 hosp_val 上微调模型
 #  runTest()  # 没什么用，改好设置后可用来算模型准确率


