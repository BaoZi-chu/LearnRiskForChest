from __future__ import print_function
import os
from glob import glob
from os.path import join

import numpy as np
import time
import sys
from  Folder import ImageFolder

from torch.autograd import Variable
# from torch.nn.functional import softmax
from tqdm import tqdm
import torch, random
# import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
# import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import torch.nn.functional as func
# from torchvision.datasets import ImageFolder

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201
from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d

from PIL import Image
import shutil
import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
# import calibration as cb
from utils import *

from risk_one_rule import risk_dataset
from risk_one_rule import risk_torch_model
import risk_one_rule.risk_torch_model as risk_model
from common import config as config_risk

from scipy.special import softmax

import csv

cfg = config_risk.Configuration(config_risk.global_data_selection, config_risk.global_deep_learning_selection)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

"""Seed and GPU setting"""
seed = 3024
# seed = (int)(sys.argv[1])
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.cuda.manual_seed(seed)

cudnn.benchmark = True
cudnn.deterministic = True


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            # true_dist = pred.data.clone()
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def output_risk_scores(file_path, id_2_scores, label_index, ground_truth_y, predict_y):
    op_file = open(file_path, 'w', 1, encoding='utf-8')
    for i in range(len(id_2_scores)):
        _id = id_2_scores[i][0]
        _risk = id_2_scores[i][1]
        _label_index = label_index.get(_id)
        _str = "{}, {}, {}, {}".format(ground_truth_y[_label_index],
                                       predict_y[_label_index],
                                       _risk,
                                       _id)
        op_file.write(_str + '\n')
    op_file.flush()
    op_file.close()
    return True

def prepare_data_4_risk_data():
    """
    first, generate , include all_info.csv, train.csv, val.csv, test.csv.
    second, use csvs to generate rules. one rule just judge one class
    :return:
    """
    train_data, validation_data, test_data = risk_dataset.load_data(cfg)
    return train_data, validation_data, test_data

def prepare_data_4_risk_model(train_data, validation_data, test_data):

    rm = risk_torch_model.RiskTorchModel()
    rm.train_data = train_data
    rm.validation_data = validation_data
    rm.test_data = test_data
    return rm

# --------------------------------------------------------------------------------

class ChexnetTrainer():

    # ---- Train the densenet network
    # ---- pathDirData - path to the directory that contains images
    # ---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    # ---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    # ---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    # ---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    # ---- class_num - number of output classes
    # ---- batch_size - batch size
    # ---- nb_epoch - number of epochs
    # ---- transResize - size of the image to scale down to (not used in current implementation)
    # ---- transCrop - size of the cropped image
    # ---- launchTimestamp - date/time, used to assign unique name for the model_path file
    # ---- model_path - if not None loads the model and continues training

    def train(pathImgTrain, pathImgVal, pathImgTest, nnArchitecture, nnIsTrained, class_num, batch_size, nb_epoch,
              transResize, transCrop, launchTimestamp, val_num, store_name, model_path=None,  start_epoch=0,resume=False):
        save_name = os.path.join('/home/4t/ltw/risk_val_pmg_result/', str(val_num), store_name.split('/')[-1],
                                 str(seed))
        if (not os.path.exists(save_name)):
            os.makedirs(save_name)

        save_p='covid_hosp_r50'
        save_p='/home/ssd0/lfy/result_archive/'+save_p


        shutil.copyfile(save_p+"/distribution_xc_test.csv", save_name + "/test_pre.csv")
        shutil.copyfile(save_p + "/distribution_x4_test.csv", save_name + "/test_dis.csv")
        shutil.copyfile(save_p + "/targets_test.csv", save_name + "/targets_test.csv")
        shutil.copyfile(save_p + "/targets_train.csv", save_name + "/targets_train.csv")
        shutil.copyfile(save_p + "/distribution_x4_train.csv", save_name + "/train_dis.csv")
        shutil.copyfile(save_p + "/predictions_test.csv", save_name + "/test_label.csv")

        # setup output
        exp_dir = save_name
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            os.stat(exp_dir)
        except:
            os.makedirs(exp_dir)

        use_cuda = torch.cuda.is_available()
        print(use_cuda)


        # -------------------- SETTINGS: NETWORK ARCHITECTURE
        # if nnArchitecture == 'DENSE-NET-121':
        #     model = DenseNet121(1, nnIsTrained).cuda()
        # elif nnArchitecture == 'DENSE-NET-169':
        #     model = DenseNet169(1, nnIsTrained).cuda()
        # elif nnArchitecture == 'DENSE-NET-201':
        #     model = DenseNet201(1, nnIsTrained).cuda()
        # elif nnArchitecture == 'RESNET-50':
        #     print('model generat')
        model_zoo = {'r18': resnet18, 'r34': resnet34, 'r50': resnet50, 'r101': resnet101, 'r152': resnet152,
                         'rx50': resnext50_32x4d}
        model = model_zoo['r50'](pretrained=True).cuda()
        model.fc = nn.Linear(model.fc.in_features, 2)

        # model = torch.nn.DataParallel(model).cuda()

        # -------------------- SETTINGS: DATA TRANSFORMS
        # normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transformList = []
        # transformList.append(transforms.RandomResizedCrop(transCrop))
        # transformList.append(transforms.RandomHorizontalFlip())
        # transformList.append(transforms.ToTensor())
        # transformList.append(normalize)
        # transformSequence = transforms.Compose(transformList)
        #
        # # -------------------- SETTINGS: DATASET BUILDERS
        # datasetTrain = ImageFolder(root=pathImgTrain, transform=transformSequence)
        # datasetVal = ImageFolder(root=pathImgVal, transform=transformSequence)
        # datasetTest = ImageFolder(root=pathImgTest, transform=transformSequence)

        # dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=batch_size, shuffle=False, num_workers=4,
        #                              pin_memory=True)
        # dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=batch_size, shuffle=False, num_workers=4,)
        #
        # dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=batch_size, shuffle=False, num_workers=4,
        #                            pin_memory=True)
        # test_shuffle_loader = torch.utils.data.DataLoader(datasetTest, batch_size=batch_size, shuffle=True, num_workers=4)

        # -------------------- SETTINGS: OPTIMIZER & SCHEDULER
        # lr_begin = (batch_size / 256) * 0.1
        # print((batch_size / 256) * 0.1)
        lr_begin = 0.0005
        optimizer = torch.optim.SGD(model.parameters(), lr=lr_begin, momentum=0.9, weight_decay=5e-4)

        # -------------------- SETTINGS: LOSS
        # loss = torch.nn.BCELoss(reduction='mean')

        # ---- Load model_path
        if model_path != None:
            ckpt = torch.load(model_path)
            model.load_state_dict(ckpt['model'], False)
            # modelmodel_path = torch.load(model_path)
            # model.load_state_dict(modelmodel_path['state_dict'])
            # optimizer.load_state_dict(ckpt['optimizer'])

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,16,2)

        # ---- TRAIN THE NETWORK
        train_data, val_data, test_data = prepare_data_4_risk_data()
        risk_data = [train_data, val_data, test_data]

        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model.to(device)
        #transforms.Normalize的的参数可以理解为是固定的,它的作用是将数据标准化,简单来说就是将数据按通道进行计算，
        # 将每一个通道的数据先计算出其方差与均值，然后再将其每一个通道内的每一个数据减去均值，再除以方差，得到归一化后的结果。
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transform_test = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.FiveCrop(224),
        #     transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        #     transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])),
        # ])

        transformList = []
        transformList.append(transforms.Resize(256))
        # transformList.append(transforms.Resize((256, 256)))
        # transformList.append(transforms.TenCrop(224))
        #transforms.FiveCrop是上下左右中心裁剪
        transformList.append(transforms.FiveCrop(224))
        transformList.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transform_test = transforms.Compose(transformList)
        #Compose()类会将transforms列表里面的transform操作进行遍历,Compose()类的参数是列表
        #这里可以理解为将裁剪完的图片放进transform_test
        # normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transformList = []
        # transformList.append(transforms.RandomResizedCrop(224))
        # transformList.append(transforms.RandomHorizontalFlip())
        # transformList.append(transforms.ToTensor())
        # transformList.append(normalize)
        # transform_test = transforms.Compose(transformList)
        testset = MyImageFolder(
            root='/home/4t/lfy/datasets/{}/test'.format(store_name), transform=transform_test
        )
        TestDataLoader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=True, num_workers=0)
        LSLLoss = LabelSmoothingLoss(class_num, 0.1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        # model = torch.nn.DataParallel(model)
        max_test_acc=0
        #参数存疑 为什么是0
        for epochID in range(0, nb_epoch):


            _,train_pre=ChexnetTrainer.test('/home/4t/lfy/datasets/{}/train'.format(store_name),model, 'RESNET-50', 1,False, 1,
                                                        256,244)
            _,val_pre=ChexnetTrainer.Valtest('/home/4t/lfy/datasets/{}/val'.format(store_name),model, 'RESNET-50', 1,False, 1,
                                                        256,244)
            _,test_pre=ChexnetTrainer.test('/home/4t/lfy/datasets/{}/test'.format(store_name),model, 'RESNET-50', 1,False, 1,
                                                        256,244)
            # lossVal, losstensor = ChexnetTrainer.epochVal(model, dataLoaderVal, optimizer, scheduler, nb_epoch,
            #                                               class_num, loss)

            my_risk_model = prepare_data_4_risk_model(risk_data[0], risk_data[1], risk_data[2])
            train_one_pre = torch.empty((0, 1), dtype=torch.float64)
            val_one_pre = torch.empty((0, 1), dtype=torch.float64)
            test_one_pre = torch.empty((0, 1), dtype=torch.float64)
            #torch.empty(0,1)创建一个1行1列的矩阵

            # scheduler.step(losstensor.data[0])
            # scheduler.step(losstensor.data)

            # if lossVal < lossMIN:
            #     lossMIN = lossVal
            #     torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN,
            #                 'optimizer': optimizer.state_dict()}, 'm-' + launchTimestamp + '.pth.tar')
            #     print('--- Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] ---\n')
            # else:
            #     print('--- Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] ---\n')


            a, _ = torch.max(train_pre, 1)
            b, _ = torch.max(val_pre, 1)
            c, _ = torch.max(test_pre, 1)

            train_one_pre = torch.cat((train_one_pre.cpu(), torch.reshape(a, (-1, 1))), dim=0).cpu().numpy()
            val_one_pre = torch.cat((val_one_pre.cpu(), torch.reshape(b, (-1, 1))), dim=0).cpu().numpy()
            test_one_pre = torch.cat((test_one_pre.cpu(), torch.reshape(c, (-1, 1))), dim=0).cpu().numpy()
            train_labels = torch.argmax(train_pre, 1).cpu().numpy()
            # np.save('train_label.npy', train_labels)
            val_labels = torch.argmax(val_pre, 1).cpu().numpy()
            # np.save('val_label', val_labels)
            #argmax函数：torch.argmax(input, dim=None, keepdim=False) 返回指定维度最大值的序号
            test_labels = torch.argmax(test_pre, 1).cpu().numpy()
            # np.save('test_label', test_labels)


            my_risk_model.train(train_one_pre, val_one_pre, test_one_pre, train_pre.cpu().numpy(),
                                     val_pre.cpu().numpy(),
                                     test_pre.cpu().numpy(), train_labels, val_labels, test_labels, epochID)
            my_risk_model.predict(test_one_pre, test_pre.cpu().numpy(), )

            test_num = my_risk_model.test_data.data_len
            test_ids = my_risk_model.test_data.data_ids
            test_pred_y = test_labels
            test_true_y = my_risk_model.test_data.true_labels
            risk_scores = my_risk_model.test_data.risk_values

            id_2_label_index = dict()
            id_2_VaR_risk = []
            for i in range(test_num):
                id_2_VaR_risk.append([test_ids[i], risk_scores[i]])
                id_2_label_index[test_ids[i]] = i
            id_2_VaR_risk = sorted(id_2_VaR_risk, key=lambda item: item[1], reverse=True)
            if epochID == 0:
                output_risk_scores(exp_dir + '/risk_score.txt', id_2_VaR_risk, id_2_label_index, test_true_y,
                                   test_pred_y)

            id_2_risk = []
            for i in range(test_num):
                test_pred = test_one_pre[i]
                m_label = test_pred_y[i]
                t_label = test_true_y[i]
                if m_label == t_label:
                    label_value = 0.0
                else:
                    label_value = 1.0
                id_2_risk.append([test_ids[i], 1 - test_pred])
            id_2_risk_desc = sorted(id_2_risk, key=lambda item: item[1], reverse=True)
            if epochID == 0:
                output_risk_scores(exp_dir + '/base_score.txt', id_2_risk_desc, id_2_label_index, test_true_y,
                                   test_pred_y)

            budgets = [10, 20, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]
            risk_correct = [0] * len(budgets)
            base_correct = [0] * len(budgets)
            for i in range(test_num):
                for budget in range(len(budgets)):
                    if i < budgets[budget]:
                        pair_id = id_2_VaR_risk[i][0]
                        _index = id_2_label_index.get(pair_id)
                        if test_true_y[_index] != test_pred_y[_index]:
                            risk_correct[budget] += 1
                        pair_id = id_2_risk_desc[i][0]
                        _index = id_2_label_index.get(pair_id)
                        if test_true_y[_index] != test_pred_y[_index]:
                            base_correct[budget] += 1


            risk_loss_criterion = risk_model.RiskLoss(my_risk_model)
            risk_loss_criterion = risk_loss_criterion.cuda()

            rule_mus = torch.tensor(my_risk_model.test_data.get_risk_mean_X_discrete(), dtype=torch.float64).cuda()
            machine_mus = torch.tensor(my_risk_model.test_data.get_risk_mean_X_continue(), dtype=torch.float64).cuda()
            rule_activate = torch.tensor(my_risk_model.test_data.get_rule_activation_matrix(),
                                         dtype=torch.float64).cuda()
            machine_activate = torch.tensor(my_risk_model.test_data.get_prob_activation_matrix(),
                                            dtype=torch.float64).cuda()
            machine_one = torch.tensor(my_risk_model.test_data.machine_label_2_one, dtype=torch.float64).cuda()
            risk_y = torch.tensor(my_risk_model.test_data.risk_labels, dtype=torch.float64).cuda()
            # risk_mul_y = torch.tensor(self.my_risk_model.test_data.risk_mul_labels).to(device[0])
            # risk_activate = torch.tensor(self.my_risk_model.test_data.risk_activate).to(device[0])
            # machine_mul_probs = torch.tensor(test_pre).to(device[0])

            test_ids = my_risk_model.test_data.data_ids
            test_ids_dict = dict()
            for ids_i in range(len(test_ids)):
                test_ids[ids_i] = os.path.basename(
                    test_ids[ids_i])
                test_ids_dict[test_ids[ids_i]] = ids_i

            del my_risk_model

            data_len = len(risk_y)
            #
            # datasetTest = ImageFolder(root='/home/4t/lfy/datasets/get_distribution_hosp_new/test', transform=transformSequence)
            # # datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
            # dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=4, num_workers=6, shuffle=False,
            #                             pin_memory=True)

            model.train()

            for batch_idx, (inputs, targets, paths) in enumerate(TestDataLoader):

                optimizer.zero_grad()

                idx = batch_idx
                if inputs.shape[0] < batch_size:
                    continue
                if use_cuda:
                    inputs, targets = inputs.to(device), targets.to(device)
                inputs, targets = Variable(inputs), Variable(targets)

                # # update learning rate
                # for nlr in range(len(optimizer.param_groups)):
                #     optimizer.param_groups[nlr]['lr'] = cosine_anneal_schedule(epoch, nb_epoch, lr[nlr])

                index = []

                # we just need class_name and image_name
                paths = list(paths)
                for path_i in range(len(paths)):
                    paths[path_i] = os.path.basename(
                        paths[path_i])
                    # print(paths[path_i])
                    index.append(test_ids_dict[paths[path_i]])
                #               print(index)

                test_pre_batch = test_pre[index]
                rule_mus_batch = rule_mus[index]
                machine_mus_batch = machine_mus[index]
                rule_activate_batch = rule_activate[index]
                machine_activate_batch = machine_activate[index]
                machine_one_batch = machine_one[index]

                # optimizer.zero_grad()
                # _, _, _, output_concat, _, _ = net(inputs)
                chex=1
                if chex == 1:
                    bs, n_crops, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)

                inputs, targets = inputs.cuda(), targets.cuda()
                try:
                    x4, xc = model(inputs)

                except:
                    xc = model(inputs)

                if chex == 1:
                    xc = xc.cuda().squeeze().view(bs, n_crops, -1).mean(1)


                # out=model(inputs).squeeze()
                # out_2=1-out
                out=xc
                out_2=1-out
                out_temp=torch.reshape(out,(-1,1))
                out_2=torch.reshape(out_2,(-1,1))
                out_2D=torch.cat((out_temp,out_2),1)
                risk_labels = risk_loss_criterion(test_pre_batch,
                                                  rule_mus_batch,
                                                  machine_mus_batch,
                                                  rule_activate_batch,
                                                  machine_activate_batch,
                                                  machine_one_batch,
                                                  out_2D, labels=None)
                # risk_labels = risk_labels.to(torch.float32)

                # out = out.to(torch.float32)

                with open('/home/ssd1/ltw/PMG/risk_lable.txt', 'a') as file:
                    file.write('%d\n'%(batch_idx))
                    out_l=np.array(out.cpu().detach().numpy())
                    risk_l=np.array(risk_labels.cpu().numpy())
                    target=np.array(targets.data.cpu().numpy())

                    targets=targets.cuda()
                    risk_labels=risk_labels.cuda()


                    file.write("risk_lab\n")
                    np.savetxt(file,risk_l,delimiter=',')
                    file.write("true_label\n")
                    np.savetxt(file,target,delimiter=',')
                    file.write("out_label\n")
                    np.savetxt(file,out_l,delimiter=',')
                    file.write('\n')

                    Loss= LSLLoss(out, risk_labels) * 1
                    Loss.backward()
                    optimizer.step()

            test_acc,_ = ChexnetTrainer.test('/home/4t/lfy/datasets/{}/test'.format(store_name), model, 'RESNET-50', 1,
                                              False, 1,
                                              256, 244)
            if test_acc > max_test_acc:
                max_test_acc = test_acc
            #     old_models = sorted(glob(join(exp_dir, 'max_*.pth')))
            #     if len(old_models) > 0: os.remove(old_models[0])
            #     torch.save({
            #                 'model': model.state_dict(),
            #                 'optimizer': optimizer.state_dict(),
            #                 'scheduler': scheduler.state_dict(),
            #                 },
            #                os.path.join('/home/ssd1/ltw/PMG/chest_covidx_duibi/', "max_acc.pth"),
            #                _use_new_zipfile_serialization=False)



        # scheduler.step()
        # torch.save({
        #             'model': model.state_dict(),
        #             'optimizer': optimizer.state_dict(),
        #             'scheduler': scheduler.state_dict(),
        #             },
        #            "/home/4t/lfy/RiskCkpt.pth",
        #            _use_new_zipfile_serialization=False)

        print(max_test_acc)





    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------

    def epochTrain(model, dataLoader, optimizer, scheduler, epochMax, classCount, loss, class_num):
        y_pred, y_true, y_score = [], [], []
        y_score_n = torch.empty([0, 2], dtype=torch.float32)
        lossTrain = 0
        lr_now = optimizer.param_groups[0]["lr"]
        model.train()

        for batchID, (input, target) in enumerate(tqdm(dataLoader, ncols=80)):
            optimizer.zero_grad()
            y_true.extend(target)
            input = input.cuda().to(torch.float32)
            target = target.cuda().to(torch.float32)


            varInput = torch.autograd.Variable(input)
            varTarget = torch.autograd.Variable(target)
            varOutput = model(varInput).squeeze()
            lossvalue = loss(varOutput, varTarget)
            varOutput_f = 1 - varOutput
            varOutput_n = torch.reshape(varOutput, (16, 1))
            varOutput_f = torch.reshape(varOutput_f, (16, 1))
            varOutput_n = torch.cat((varOutput_n, varOutput_f), 1)
            y_score_n = torch.cat((y_score_n.cuda(), varOutput_n), 0)

            y_score.extend(varOutput.data.cuda())
            y_pred.extend([0 if socre < 0.5 else 1 for socre in varOutput.data])
            lossTrain += lossvalue.item()



        lossTrain /= len(dataLoader)
        train_f1 = 100 * f1_score(y_true, y_pred)
        train_recall = 100 * recall_score(y_true, y_pred)
        train_precision = 100 * precision_score(y_true, y_pred)

        train_acc = 100 * accuracy_score(y_true, y_pred)
        train_auc = 100.0 * roc_auc_score(y_true, y_score)
        print(
            "Train | lr: {:.4f} | Loss: {:.4f} | Acc: {:.3f} | F1: {:.3f} | Auc: {:.2f}".format(
                lr_now, lossTrain, train_acc, train_f1, train_auc
            )
        )
        return train_acc,train_auc, varTarget,y_score_n

    # --------------------------------------------------------------------------------

    def epochVal(model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):

        model.eval()

        lossVal = 0
        lossValNorm = 0
        losstensorMean = 0
        y_pred, y_true, y_score = [], [], []
        y_score_n = torch.empty([0, 2], dtype=torch.float32)

        with torch.no_grad():
            for i, (input, target, _) in enumerate(tqdm(dataLoader, ncols=80)):
                y_true.extend(target)
                input = input.cuda().to(torch.float32)
                target = target.cuda().to(torch.float32)



                varInput = torch.autograd.Variable(input)
                varTarget = torch.autograd.Variable(target)
                varOutput = model(varInput).squeeze()


                varOutput_f = 1 - varOutput
                varOutput_n = torch.reshape(varOutput, (-1, 1))
                varOutput_f = torch.reshape(varOutput_f, (-1, 1))
                varOutput_n = torch.cat((varOutput_n, varOutput_f), 1)
                y_score_n = torch.cat((y_score_n.cuda(), varOutput_n), 0)

                y_score.extend(varOutput.data.cuda())
                y_pred.extend([0 if socre < 0.5 else 1 for socre in varOutput.data])
                losstensor = loss(varOutput, varTarget)
                losstensorMean += losstensor
                lossVal += losstensor.data
                lossValNorm += 1

        outLoss = lossVal / lossValNorm
        losstensorMean = losstensorMean / lossValNorm


        lossVal /= len(dataLoader)
        val_f1 = 100 * f1_score(y_true, y_pred)
        val_recall = 100 * recall_score(y_true, y_pred)
        val_precision = 100 * precision_score(y_true, y_pred)
        val_acc = 100 * accuracy_score(y_true, y_pred)
        val_auc = 100.0 * roc_auc_score(y_true, y_score)
        print(
            "val  | Loss: {:.4f} | Acc: {:.3f} | F1: {:.3f} | Auc: {:.2f}".format(
                lossVal, val_acc, val_f1, val_auc
            )
        )

        return outLoss, losstensorMean,varTarget,y_score_n

    # --------------------------------------------------------------------------------

    # ---- Computes area under ROC curve
    # ---- dataGT - ground truth data
    # ---- dataPRED - predicted data
    # ---- classCount - number of classes

    def computeAUROC(dataGT, dataPRED, classCount):

        outAUROC = []

        datanpGT = dataGT.cuda().numpy()
        datanpPRED = dataPRED.cuda().numpy()

        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))

        return outAUROC

    # --------------------------------------------------------------------------------

    # ---- Test the trained network
    # ---- pathDirData - path to the directory that contains images
    # ---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    # ---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    # ---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    # ---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    # ---- class_num - number of output classes
    # ---- batch_size - batch size
    # ---- nb_epoch - number of epochs
    # ---- transResize - size of the image to scale down to (not used in current implementation)
    # ---- transCrop - size of the cropped image
    # ---- launchTimestamp - date/time, used to assign unique name for the model_path file
    # ---- model_path - if not None loads the model and continues training

    def test(pathImgTest, pathModel, nnArchitecture, class_num, nnIsTrained, batch_size, transResize, transCrop,ckpt=False):

        # -------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        # if nnArchitecture == 'DENSE-NET-121':
        #     model = DenseNet121(class_num, nnIsTrained).cuda()
        # elif nnArchitecture == 'DENSE-NET-169':
        #     model = DenseNet169(class_num, nnIsTrained).cuda()
        # elif nnArchitecture == 'DENSE-NET-201':
        #     model = DenseNet201(class_num, nnIsTrained).cuda()
        # model_zoo = {'r18': resnet18, 'r34': resnet34, 'r50': resnet50, 'r101': resnet101, 'r152': resnet152,
        #              'rx50': resnext50_32x4d}
        # model = model_zoo['r50'](pretrained=True).cuda()
        # model.fc = nn.Linear(model.fc.in_features, 2)
        # modelmodel_path = torch.load(pathModel)
        # model.load_state_dict(modelmodel_path['state_dict'])
        model = pathModel
        # if ckpt:
        #     net_file_name = "/home/4t/lfy/RiskCkpt.pth"
        #     model.load_state_dict(torch.load(net_file_name)['model'])


        # -------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        # normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # -------------------- SETTINGS: DATASET BUILDERS
        # transformList = []
        # transformList.append(transforms.Resize(transResize))
        # transformList.append(transforms.TenCrop(transCrop))
        # transformList.append(
        # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        # transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        # transformSequence = transforms.Compose(transformList)
        #
        # datasetTest = ImageFolder(root=pathImgTest, transform=transformSequence)
        # # datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
        # dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=batch_size, num_workers=8, shuffle=False,
        #                             pin_memory=True)
        #
        # outGT = torch.FloatTensor().cuda()
        # outPRED = torch.FloatTensor().cuda()
        #
        # model.eval()
        # y_pred, y_true, y_score ,y_score_out= [], [], [] ,[]
        # paths=[]
        # y_score_n = torch.empty([0, 2], dtype=torch.float32)
        # with torch.no_grad():
        #     for i, (input, target,path) in enumerate(tqdm(dataLoaderTest, ncols=80)):
        #         y_true.extend(target)
        #         input = input.cuda().to(torch.float32)
        #         target = target.cuda().to(torch.float32)
        #         outGT = torch.cat((outGT, target), 0)
        #
        #         bs, n_crops, c, h, w = input.size()
        #
        #         varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())
        #
        #
        #         out = model(varInput).squeeze()
        #         outMean = out.view(bs, n_crops, -1).mean(1)
        #         y_score.extend(outMean.data.cpu())
        #         y_pred.extend([0 if socre < 0.5 else 1 for socre in outMean.data])
        #         outPRED = torch.cat((outPRED, outMean.data), 0)
        #
        #         varOutput_f = 1 - outMean
        #         varOutput_n = torch.reshape(outMean, (-1, 1))
        #         varOutput_f = torch.reshape(varOutput_f, (-1, 1))
        #         varOutput_n = torch.cat((varOutput_n, varOutput_f), 1)
        #         y_score_n = torch.cat((varOutput_n, y_score_n.cuda()), 0)
        #         y_score_out.extend(outMean.data.cpu().tolist())
        #         paths.extend(path)
        #
        #
        # test_f1 = 100 * f1_score(y_true, y_pred)
        # test_recall = 100 * recall_score(y_true, y_pred)
        # test_precision = 100 * precision_score(y_true, y_pred)
        # test_acc = 100 * accuracy_score(y_true, y_pred)
        # test_auc = 100.0 * roc_auc_score(y_true, y_score)
        # print("acc\tf1\tprecision\trecall\tauc")
        # print("{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}\n".format(test_acc, test_f1, test_precision, test_recall,
        #                                                           test_auc))
        # if out_print :
        #   with open('/home/ssd1/ltw/PMG/test_result.txt','a') as file:
        #     file.write("{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}\n".format(test_acc, test_f1, test_precision, test_recall,
        #                                                           test_auc))
        #
        # normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #
        # # -------------------- SETTINGS: DATASET BUILDERS
        # transformList = []
        # transformList.append(transforms.Resize(256))
        # transformList.append(transforms.TenCrop(244))
        # transformList.append(
        #     transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        # transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        # transformSequence = transforms.Compose(transformList)
        #
        # datasetTest = ImageFolder(root=pathImgTest, transform=transformSequence)
        # # datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
        # dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=4, num_workers=6, shuffle=False,
        #                             pin_memory=True)

        # if len(args.gpu) > 1: net = torch.nn.DataParallel(net)
        #   net_file_name = glob(join(exp_dir, "max_*.pth"))[0]
        #   net.load_state_dict(torch.load(join(exp_dir, net_file_name))['model'])
        model.eval()
        model.cuda()

        # testset = ImageFolder(
        #     root=os.path.join(data_dir, data_set), transform=transform_test
        # )
        # testloader = torch.utils.data.DataLoader(
        #     testset, batch_size=batch_size, shuffle=False, num_workers=6
        # )
        # testloader = dataLoaderTest
        # distribution_x4 = []
        # distribution_xc = []
        # paths = []
        # test_loss = 0
        # y_pred, y_true, y_score, varOutput_f = [], [], [], []
        y_score_n = torch.empty([0, 2], dtype=torch.float32)

        chex=1
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize(256))
        # transformList.append(transforms.Resize((256, 256)))
        # transformList.append(transforms.TenCrop(224))
        transformList.append(transforms.FiveCrop(224))
        transformList.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transform_test = transforms.Compose(transformList)

        with torch.no_grad():

            testset = ImageFolder(
                root=pathImgTest, transform=transform_test
            )
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            distribution_x4 = []
            distribution_xc = []
            paths = []
            y_pred, y_true, y_score = [], [], []

            for _, (inputs, targets, path) in enumerate(tqdm(testloader, ncols=80)):

                if chex == 1:
                    bs, n_crops, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)

                inputs, targets = inputs.cuda(), targets.cuda()
                try:
                    x4, xc = model(inputs)
                    distribution_x4.extend(x4.cuda().tolist())
                except:
                    xc = model(inputs)

                if chex == 1: xc = xc.squeeze().view(bs, n_crops, -1).mean(1)

                _, predicted = torch.max(xc.data, 1)
                y_score.extend([_[1] for _ in softmax(xc.data.cpu(), axis=1)])
                y_pred.extend(predicted.cuda().tolist())

                distribution_x4.extend(x4.cuda().tolist())
                distribution_xc.extend(xc.cuda().tolist())
                y_true.extend(targets.cuda().tolist())
                paths.extend(path)

                # lfy 2022.7.22
                y_score_t = [_[1] for _ in softmax(xc.data.cpu(), axis=1)]
                varOutput_f = ([_[1] for _ in softmax(1 - xc.data.cpu(), axis=1)])
                y_score_t = torch.tensor(y_score_t)
                varOutput_f = torch.tensor(varOutput_f)
                varOutput_n = torch.reshape(y_score_t, (-1, 1))
                varOutput_f = torch.reshape(varOutput_f, (-1, 1))
                varOutput_n = torch.cat((varOutput_n.cpu(), varOutput_f.cpu()), 1)
                y_score_n = torch.cat((varOutput_n.cpu(), y_score_n.cpu()), 0)

                # y_pred.extend(predicted.cpu().tolist())

                # distribution_x4.extend(x4.cpu().tolist())
                # distribution_xc.extend(xc.cpu().tolist())
                # y_true.extend(targets.cpu().tolist())
                # paths.extend(path)

            test_acc = 100.0 * accuracy_score(y_true, y_pred)
            test_f1 = 100.0 * f1_score(y_true, y_pred)
            test_recall = 100.0 * recall_score(y_true, y_pred)
            test_precision = 100.0 * precision_score(y_true, y_pred)
            test_auc = 100.0 * roc_auc_score(y_true, y_score)
            print("Dataset \t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}\n".format( test_acc, test_f1,
                                                                                  test_precision, test_recall,
                                                                                  test_auc))

            return test_acc,y_score_n
    def Valtest(pathImgTest, pathModel, nnArchitecture, testdataloader, nnIsTrained, batch_size, transResize, transCrop,ckpt=False):

        # -------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        # if nnArchitecture == 'DENSE-NET-121':
        #     model = DenseNet121(class_num, nnIsTrained).cuda()
        # elif nnArchitecture == 'DENSE-NET-169':
        #     model = DenseNet169(class_num, nnIsTrained).cuda()
        # elif nnArchitecture == 'DENSE-NET-201':
        #     model = DenseNet201(class_num, nnIsTrained).cuda()
        # model_zoo = {'r18': resnet18, 'r34': resnet34, 'r50': resnet50, 'r101': resnet101, 'r152': resnet152,
        #              'rx50': resnext50_32x4d}
        # model = model_zoo['r50'](pretrained=True).cuda()
        # model.fc = nn.Linear(model.fc.in_features, 2)
        # modelmodel_path = torch.load(pathModel)
        # model.load_state_dict(modelmodel_path['state_dict'])
        model = pathModel
        # if ckpt:
        #     net_file_name = "/home/4t/lfy/RiskCkpt.pth"
        #     model.load_state_dict(torch.load(net_file_name)['model'])


        # -------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        # normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # -------------------- SETTINGS: DATASET BUILDERS
        # transformList = []
        # transformList.append(transforms.Resize(transResize))
        # transformList.append(transforms.TenCrop(transCrop))
        # transformList.append(
        # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        # transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        # transformSequence = transforms.Compose(transformList)
        #
        # datasetTest = ImageFolder(root=pathImgTest, transform=transformSequence)
        # # datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
        # dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=batch_size, num_workers=8, shuffle=False,
        #                             pin_memory=True)
        #
        # outGT = torch.FloatTensor().cuda()
        # outPRED = torch.FloatTensor().cuda()
        #
        # model.eval()
        # y_pred, y_true, y_score ,y_score_out= [], [], [] ,[]
        # paths=[]
        # y_score_n = torch.empty([0, 2], dtype=torch.float32)
        # with torch.no_grad():
        #     for i, (input, target,path) in enumerate(tqdm(dataLoaderTest, ncols=80)):
        #         y_true.extend(target)
        #         input = input.cuda().to(torch.float32)
        #         target = target.cuda().to(torch.float32)
        #         outGT = torch.cat((outGT, target), 0)
        #
        #         bs, n_crops, c, h, w = input.size()
        #
        #         varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())
        #
        #
        #         out = model(varInput).squeeze()
        #         outMean = out.view(bs, n_crops, -1).mean(1)
        #         y_score.extend(outMean.data.cpu())
        #         y_pred.extend([0 if socre < 0.5 else 1 for socre in outMean.data])
        #         outPRED = torch.cat((outPRED, outMean.data), 0)
        #
        #         varOutput_f = 1 - outMean
        #         varOutput_n = torch.reshape(outMean, (-1, 1))
        #         varOutput_f = torch.reshape(varOutput_f, (-1, 1))
        #         varOutput_n = torch.cat((varOutput_n, varOutput_f), 1)
        #         y_score_n = torch.cat((varOutput_n, y_score_n.cuda()), 0)
        #         y_score_out.extend(outMean.data.cpu().tolist())
        #         paths.extend(path)
        #
        #
        # test_f1 = 100 * f1_score(y_true, y_pred)
        # test_recall = 100 * recall_score(y_true, y_pred)
        # test_precision = 100 * precision_score(y_true, y_pred)
        # test_acc = 100 * accuracy_score(y_true, y_pred)
        # test_auc = 100.0 * roc_auc_score(y_true, y_score)
        # print("acc\tf1\tprecision\trecall\tauc")
        # print("{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}\n".format(test_acc, test_f1, test_precision, test_recall,
        #                                                           test_auc))
        # if out_print :
        #   with open('/home/ssd1/ltw/PMG/test_result.txt','a') as file:
        #     file.write("{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}\n".format(test_acc, test_f1, test_precision, test_recall,
        #                                                           test_auc))
        #
        # normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #
        # # -------------------- SETTINGS: DATASET BUILDERS
        # transformList = []
        # transformList.append(transforms.Resize(256))
        # transformList.append(transforms.TenCrop(244))
        # transformList.append(
        #     transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        # transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        # transformSequence = transforms.Compose(transformList)
        #
        # datasetTest = ImageFolder(root=pathImgTest, transform=transformSequence)
        # # datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
        # dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=4, num_workers=6, shuffle=False,
        #                             pin_memory=True)

        # if len(args.gpu) > 1: net = torch.nn.DataParallel(net)
        #   net_file_name = glob(join(exp_dir, "max_*.pth"))[0]
        #   net.load_state_dict(torch.load(join(exp_dir, net_file_name))['model'])
        model.eval()
        model.cuda()

        # testset = ImageFolder(
        #     root=os.path.join(data_dir, data_set), transform=transform_test
        # )
        # testloader = torch.utils.data.DataLoader(
        #     testset, batch_size=batch_size, shuffle=False, num_workers=6
        # )
        # testloader = dataLoaderTest
        # distribution_x4 = []
        # distribution_xc = []
        # paths = []
        # test_loss = 0
        # y_pred, y_true, y_score, varOutput_f = [], [], [], []
        y_score_n = torch.empty([0, 2], dtype=torch.float32)

        chex=1
        # normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transformList = []
        # transformList.append(transforms.Resize(256))
        # # transformList.append(transforms.Resize((256, 256)))
        # # transformList.append(transforms.TenCrop(224))
        # transformList.append(transforms.FiveCrop(224))
        # transformList.append(
        #     transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        # transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.RandomResizedCrop(224))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        transform_test = transforms.Compose(transformList)
        # transform_test = transforms.Compose(transformList)
        chex=0

        with torch.no_grad():

            testset = ImageFolder(
                root=pathImgTest, transform=transform_test
            )
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False, num_workers=4
            )

            distribution_x4 = []
            distribution_xc = []
            paths = []
            y_pred, y_true, y_score = [], [], []

            for _, (inputs, targets, path) in enumerate(tqdm(testloader, ncols=80)):

                if chex == 1:
                    bs, n_crops, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)

                inputs, targets = inputs.cuda(), targets.cuda()
                try:
                    x4, xc = model(inputs)
                    distribution_x4.extend(x4.cuda().tolist())
                except:
                    xc = model(inputs)

                if chex == 1: xc = xc.squeeze().view(bs, n_crops, -1).mean(1)

                _, predicted = torch.max(xc.data, 1)
                y_score.extend([_[1] for _ in softmax(xc.data.cpu(), axis=1)])
                y_pred.extend(predicted.cuda().tolist())

                distribution_x4.extend(x4.cuda().tolist())
                distribution_xc.extend(xc.cuda().tolist())
                y_true.extend(targets.cuda().tolist())
                paths.extend(path)

                # lfy 2022.7.22
                y_score_t = [_[1] for _ in softmax(xc.data.cpu(), axis=1)]
                varOutput_f = ([_[1] for _ in softmax(1 - xc.data.cpu(), axis=1)])
                y_score_t = torch.tensor(y_score_t)
                varOutput_f = torch.tensor(varOutput_f)
                varOutput_n = torch.reshape(y_score_t, (-1, 1))
                varOutput_f = torch.reshape(varOutput_f, (-1, 1))
                varOutput_n = torch.cat((varOutput_n.cpu(), varOutput_f.cpu()), 1)
                y_score_n = torch.cat((varOutput_n.cpu(), y_score_n.cpu()), 0)

                # y_pred.extend(predicted.cpu().tolist())

                # distribution_x4.extend(x4.cpu().tolist())
                # distribution_xc.extend(xc.cpu().tolist())
                # y_true.extend(targets.cpu().tolist())
                # paths.extend(path)

            test_acc = 100.0 * accuracy_score(y_true, y_pred)
            test_f1 = 100.0 * f1_score(y_true, y_pred)
            test_recall = 100.0 * recall_score(y_true, y_pred)
            test_precision = 100.0 * precision_score(y_true, y_pred)
            test_auc = 100.0 * roc_auc_score(y_true, y_score)
            print("Dataset \t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}\n".format( test_acc, test_f1,
                                                                                  test_precision, test_recall,
                                                                                  test_auc))

            return test_acc,y_score_n

        # aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, class_num)
        # aurocMean = np.array(aurocIndividual).mean()
        # auroc = 100 * roc_auc_score(outGT.cpu(), outPRED.cpu())
        #
        # print ('AUROC: {:.2f}'.format(auroc))

        # for i in range (0, len(aurocIndividual)):
        #     print (CLASS_NAMES[i], ' ', aurocIndividual[i])
        # with open('/home/ssd1/ltw/PMG/hosp_test_b32_e64_lr0005_new/' + pathImgTest.split('/')[-1] + '_data.csv',
        #           'a') as csvfile:
        #     write = csv.writer(csvfile)
        #     write.writerows([paths,y_score_out])
        # return y_score_n
    def Ttest(pathImgTest, pathModel, nnArchitecture, testdataloader, nnIsTrained, batch_size, transResize, transCrop,ckpt=False):

        # -------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        # if nnArchitecture == 'DENSE-NET-121':
        #     model = DenseNet121(class_num, nnIsTrained).cuda()
        # elif nnArchitecture == 'DENSE-NET-169':
        #     model = DenseNet169(class_num, nnIsTrained).cuda()
        # elif nnArchitecture == 'DENSE-NET-201':
        #     model = DenseNet201(class_num, nnIsTrained).cuda()
        # model_zoo = {'r18': resnet18, 'r34': resnet34, 'r50': resnet50, 'r101': resnet101, 'r152': resnet152,
        #              'rx50': resnext50_32x4d}
        # model = model_zoo['r50'](pretrained=True).cuda()
        # model.fc = nn.Linear(model.fc.in_features, 2)
        # modelmodel_path = torch.load(pathModel)
        # model.load_state_dict(modelmodel_path['state_dict'])
        model = pathModel
        # if ckpt:
        #     net_file_name = "/home/4t/lfy/RiskCkpt.pth"
        #     model.load_state_dict(torch.load(net_file_name)['model'])


        # -------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        # normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # -------------------- SETTINGS: DATASET BUILDERS
        # transformList = []
        # transformList.append(transforms.Resize(transResize))
        # transformList.append(transforms.TenCrop(transCrop))
        # transformList.append(
        # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        # transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        # transformSequence = transforms.Compose(transformList)
        #
        # datasetTest = ImageFolder(root=pathImgTest, transform=transformSequence)
        # # datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
        # dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=batch_size, num_workers=8, shuffle=False,
        #                             pin_memory=True)
        #
        # outGT = torch.FloatTensor().cuda()
        # outPRED = torch.FloatTensor().cuda()
        #
        # model.eval()
        # y_pred, y_true, y_score ,y_score_out= [], [], [] ,[]
        # paths=[]
        # y_score_n = torch.empty([0, 2], dtype=torch.float32)
        # with torch.no_grad():
        #     for i, (input, target,path) in enumerate(tqdm(dataLoaderTest, ncols=80)):
        #         y_true.extend(target)
        #         input = input.cuda().to(torch.float32)
        #         target = target.cuda().to(torch.float32)
        #         outGT = torch.cat((outGT, target), 0)
        #
        #         bs, n_crops, c, h, w = input.size()
        #
        #         varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())
        #
        #
        #         out = model(varInput).squeeze()
        #         outMean = out.view(bs, n_crops, -1).mean(1)
        #         y_score.extend(outMean.data.cpu())
        #         y_pred.extend([0 if socre < 0.5 else 1 for socre in outMean.data])
        #         outPRED = torch.cat((outPRED, outMean.data), 0)
        #
        #         varOutput_f = 1 - outMean
        #         varOutput_n = torch.reshape(outMean, (-1, 1))
        #         varOutput_f = torch.reshape(varOutput_f, (-1, 1))
        #         varOutput_n = torch.cat((varOutput_n, varOutput_f), 1)
        #         y_score_n = torch.cat((varOutput_n, y_score_n.cuda()), 0)
        #         y_score_out.extend(outMean.data.cpu().tolist())
        #         paths.extend(path)
        #
        #
        # test_f1 = 100 * f1_score(y_true, y_pred)
        # test_recall = 100 * recall_score(y_true, y_pred)
        # test_precision = 100 * precision_score(y_true, y_pred)
        # test_acc = 100 * accuracy_score(y_true, y_pred)
        # test_auc = 100.0 * roc_auc_score(y_true, y_score)
        # print("acc\tf1\tprecision\trecall\tauc")
        # print("{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}\n".format(test_acc, test_f1, test_precision, test_recall,
        #                                                           test_auc))
        # if out_print :
        #   with open('/home/ssd1/ltw/PMG/test_result.txt','a') as file:
        #     file.write("{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}\n".format(test_acc, test_f1, test_precision, test_recall,
        #                                                           test_auc))
        #
        # normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #
        # # -------------------- SETTINGS: DATASET BUILDERS
        # transformList = []
        # transformList.append(transforms.Resize(256))
        # transformList.append(transforms.TenCrop(244))
        # transformList.append(
        #     transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        # transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        # transformSequence = transforms.Compose(transformList)
        #
        # datasetTest = ImageFolder(root=pathImgTest, transform=transformSequence)
        # # datasetTest = DatasetGenerator(pathImageDirectory=pathDirData, pathDatasetFile=pathFileTest, transform=transformSequence)
        # dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=4, num_workers=6, shuffle=False,
        #                             pin_memory=True)

        # if len(args.gpu) > 1: net = torch.nn.DataParallel(net)
        #   net_file_name = glob(join(exp_dir, "max_*.pth"))[0]
        #   net.load_state_dict(torch.load(join(exp_dir, net_file_name))['model'])
        model.eval()
        model.cuda()

        # testset = ImageFolder(
        #     root=os.path.join(data_dir, data_set), transform=transform_test
        # )
        # testloader = torch.utils.data.DataLoader(
        #     testset, batch_size=batch_size, shuffle=False, num_workers=6
        # )
        # testloader = dataLoaderTest
        # distribution_x4 = []
        # distribution_xc = []
        # paths = []
        # test_loss = 0
        # y_pred, y_true, y_score, varOutput_f = [], [], [], []
        y_score_n = torch.empty([0, 2], dtype=torch.float32)

        chex=1
        # normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # transformList = []
        # transformList.append(transforms.Resize(256))
        # # transformList.append(transforms.Resize((256, 256)))
        # # transformList.append(transforms.TenCrop(224))
        # transformList.append(transforms.FiveCrop(224))
        # transformList.append(
        #     transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        # transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.RandomResizedCrop(224))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)
        transform_test = transforms.Compose(transformList)
        # transform_test = transforms.Compose(transformList)
        chex=1

        with torch.no_grad():

            testset = ImageFolder(
                root=pathImgTest, transform=transform_test
            )
            testloader = testdataloader
            distribution_x4 = []
            distribution_xc = []
            paths = []
            y_pred, y_true, y_score = [], [], []

            for _, (inputs, targets, path) in enumerate(tqdm(testloader, ncols=80)):

                if chex == 1:
                    bs, n_crops, c, h, w = inputs.size()
                    inputs = inputs.view(-1, c, h, w)

                inputs, targets = inputs.cuda(), targets.cuda()
                try:
                    x4, xc = model(inputs)
                    distribution_x4.extend(x4.cuda().tolist())
                except:
                    xc = model(inputs)

                if chex == 1: xc = xc.squeeze().view(bs, n_crops, -1).mean(1)

                _, predicted = torch.max(xc.data, 1)
                y_score.extend([_[1] for _ in softmax(xc.data.cpu(), axis=1)])
                y_pred.extend(predicted.cuda().tolist())

                distribution_x4.extend(x4.cuda().tolist())
                distribution_xc.extend(xc.cuda().tolist())
                y_true.extend(targets.cuda().tolist())
                paths.extend(path)

                # lfy 2022.7.22
                y_score_t = [_[1] for _ in softmax(xc.data.cpu(), axis=1)]
                varOutput_f = ([_[1] for _ in softmax(1 - xc.data.cpu(), axis=1)])
                y_score_t = torch.tensor(y_score_t)
                varOutput_f = torch.tensor(varOutput_f)
                varOutput_n = torch.reshape(y_score_t, (-1, 1))
                varOutput_f = torch.reshape(varOutput_f, (-1, 1))
                varOutput_n = torch.cat((varOutput_n.cpu(), varOutput_f.cpu()), 1)
                y_score_n = torch.cat((varOutput_n.cpu(), y_score_n.cpu()), 0)

                # y_pred.extend(predicted.cpu().tolist())

                # distribution_x4.extend(x4.cpu().tolist())
                # distribution_xc.extend(xc.cpu().tolist())
                # y_true.extend(targets.cpu().tolist())
                # paths.extend(path)

            test_acc = 100.0 * accuracy_score(y_true, y_pred)
            test_f1 = 100.0 * f1_score(y_true, y_pred)
            test_recall = 100.0 * recall_score(y_true, y_pred)
            test_precision = 100.0 * precision_score(y_true, y_pred)
            test_auc = 100.0 * roc_auc_score(y_true, y_score)
            print("Dataset \t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}\n".format( test_acc, test_f1,
                                                                                  test_precision, test_recall,
                                                                                  test_auc))

            return test_acc,y_score_n

        # aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, class_num)
        # aurocMean = np.array(aurocIndividual).mean()
        # auroc = 100 * roc_auc_score(outGT.cpu(), outPRED.cpu())
        #
        # print ('AUROC: {:.2f}'.format(auroc))

        # for i in range (0, len(aurocIndividual)):
        #     print (CLASS_NAMES[i], ' ', aurocIndividual[i])
        # with open('/home/ssd1/ltw/PMG/hosp_test_b32_e64_lr0005_new/' + pathImgTest.split('/')[-1] + '_data.csv',
        #           'a') as csvfile:
        #     write = csv.writer(csvfile)
        #     write.writerows([paths,y_score_out])
        # return y_score_n
# --------------------------------------------------------------------------------





