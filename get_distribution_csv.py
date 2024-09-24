
import os, pickle, time, shutil
from os.path import join
from glob import glob

import torch
import torch.nn as nn
from Densenet import densenet121, densenet161, densenet169, densenet201
from Folder import ImageFolder
from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d
# from Efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
from torchvision.models import alexnet
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
from scipy.special import softmax
import argparse
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
# from nets import get_model_from_name
# from utils.utils import (download_weights, get_classes, get_lr_scheduler,
#                          set_optimizer_lr, show_config, weights_init)
# from transfff import RandomSized50Crop
# from nets import get_model_from_name


begin = time.time()
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('-c','--cnn',  default='r50', help='dataset dir')
parser.add_argument('-d','--dir',  default='chest_hosp', help='dataset dir')  # chest_xray, hosp
parser.add_argument('-s','--save_dir',  default='chest_hosp_r50', help='save_dir')
parser.add_argument('-m','--multiple', default=2, type=int, help='multiple of input size')
parser.add_argument('-g','--gpu',  default='1', help='set 0,1 to use multi gpu for example')
args = parser.parse_args()



# exp settings
cnn = args.cnn
datasets_dir = args.dir
exp_dir = "/home/ssd1/ltw/tmp/pycharm_project_912/result_archive/detect/{}".format(args.save_dir)
# exp_dir = "/home/ssd0/lfy/result_archive/{}".format(args.save_dir)
batch_size = 32 #8
# os.makedirs(exp_dir, exist_ok=True)
# shutil.copy('get_distribution.py', join(exp_dir, 'get_distribution.py'))


# data settings
data_dir = join("/home/4t/lfy/datasets", datasets_dir)
# data_sets = ['train', 'val', 'test']
data_sets = ['train','val','test']
# nb_class = len(os.listdir(join(data_dir, data_sets[0])))
nb_class = 2
re_size = int(128 * args.multiple)
crop_size = 112 * args.multiple
chex=1


# CUDA setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu



# transform_test = transforms.Compose(
#     [
#         transforms.Resize((re_size, re_size)),
#         # transforms.Resize(re_size),  # 91.99	93.84	90.28		97.69	95.02
#         transforms.CenterCrop(crop_size),
#         transforms.ToTensor(),
#         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#         # transforms.Normalize([0.482, 0.482, 0.482], [0.223, 0.223, 0.223]),
#     ]
# )
#chest 
normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
transformList = []
transformList.append(transforms.Resize(256))
# transformList.append(transforms.Resize((256, 256)))
# transformList.append(transforms.TenCrop(224))
transformList.append(transforms.FiveCrop(224))
# transformList.append(RandomSized50Crop(224))
transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
transform_test = transforms.Compose(transformList)

# transform
# device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = get_model_from_name['vit'](input_shape = [224,224], num_classes = 2, pretrained = True)
# if len(args.gpu) > 1: net = torch.nn.DataParallel(net)
# net_file_name = glob(join(exp_dir, "max_*.pth"))[0]
# net.load_state_dict(torch.load(join(exp_dir, net_file_name))['model'])

# Net settings
print('===== {} ====='.format(args.save_dir))
model_zoo = {'r18':resnet18, 'r34':resnet34, 'r50':resnet50, 'r101':resnet101, 'r152':resnet152, 'd169':densenet169,
             'd121':densenet121,# 'eb0':efficientnet_b0, 'eb1':efficientnet_b1, 'eb2':efficientnet_b2, 'eb3':efficientnet_b3,
             # 'eb4':efficientnet_b4, 'eb5':efficientnet_b5, 'eb6':efficientnet_b6,  'eb7':efficientnet_b7,
             'rx50':resnext50_32x4d, 'alex':alexnet}
net = model_zoo[cnn](pretrained=True)

if not args.save_dir.endswith('_pre'):
    if cnn.startswith("r"):
        net.fc = nn.Linear(net.fc.in_features, nb_class)
    elif cnn.startswith('d'):
        net.classifier = nn.Linear(net.classifier.in_features, nb_class)
    elif cnn.startswith('e'):
        net.classifier._modules['1'] = nn.Linear(net.classifier._modules['1'].in_features, nb_class)
    elif cnn.startswith("a"):
        net.classifier._modules['6'] = nn.Linear(net.classifier._modules['6'].in_features, nb_class)

# net_file_name = glob(join(exp_dir, "max_*")0)[0]
# net_file_name = '/home/ssd0/lfy/result_archive/chest_xray_e128_r50/max_acc_77.00.pth'
net_file_name = '/home/ssd0/lfy/result_archive/chest_r50_best/max_acc_97.44.pth'

# net_file_name = glob(join(exp_dir, "min_*.pth"))[0]
try:
    try: net.load_state_dict(torch.load(net_file_name)['model'])
    except: net.load_state_dict(torch.load(net_file_name))
except:
        net = torch.nn.DataParallel(net)
        try: net.load_state_dict(torch.load(net_file_name)['model'])
        except: net.load_state_dict(torch.load(net_file_name))

net.cuda()
net.eval()

scores = []
for data_set in data_sets:
    testset = ImageFolder(
        root=os.path.join(data_dir, data_set), transform=transform_test
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    distribution_x4 = []
    distribution_xc = []
    y_pred, y_true, y_score = [], [], []
    paths = []
    f_path=[]
    f_sc=[]
    test_loss = correct = total = 0

    with torch.no_grad():

        # for _, (inputs, targets, path) in enumerate(tqdm(testloader)):
        # 
        #     inputs, targets = inputs.cuda(), targets.cuda()
        #     try:
        #         x4, xc = net(inputs)
        #         distribution_x4.extend(x4.cpu().tolist())
        #     except:
        #         xc = net(inputs)
       for _, (inputs, targets, path) in enumerate(tqdm(testloader, ncols=80)):
            if chex == 1:
                bs, n_crops, c, h, w = inputs.size()
                inputs = inputs.view(-1, c, h, w)

            inputs, targets = inputs.cuda(), targets.cuda()
            try:
                x4, xc = net(inputs)
                
            except:
                xc = net(inputs)

            if chex == 1: xc = xc.squeeze().view(bs, n_crops, -1).mean(1)
            x4 = x4.squeeze().view(bs, n_crops, -1).mean(1)

            _, predicted = torch.max(xc.data, 1)
            y_score.extend([_[1] for _ in softmax(xc.data.cpu(), axis=1)])
            y_pred.extend(predicted.cpu().tolist())

            distribution_x4.extend(x4.cpu().tolist())
            distribution_xc.extend(xc.cpu().tolist())
            y_true.extend(targets.cpu().tolist())
            paths.extend(path)
            # if targets != predicted:
            #     f_path.extend(path)
            #     f_sc.extend([_[1] for _ in softmax(xc.data.cpu(), axis=1)])
    print(y_pred)
    print(y_score)

    test_acc = 100.0 * accuracy_score(y_true, y_pred)
    test_f1 = 100.0 * f1_score(y_true, y_pred)
    test_recall = 100.0 * recall_score(y_true, y_pred)
    test_precision = 100.0 * precision_score(y_true, y_pred)
    test_auc = 100.0 * roc_auc_score(y_true, y_score)

    scores.append('{:.2f}'.format(test_acc))


    scores.append('{:.4f}'.format(test_auc))

    print("Dataset\tf1\tACC\tprecision\trecall")
    print("{}\t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}".format(data_set, test_f1, test_acc, test_precision, test_recall))

    # === 保存 pkl===
    # with open(os.path.join(exp_dir, "{}_{:.2f}\n".format(data_set, test_acc)), "a+") as file: pass
    pickle.dump(y_true, open(join(exp_dir, "targets_{}.pkl".format(data_set)), 'wb+'))
    pickle.dump(y_pred, open(join(exp_dir, "predictions_{}.pkl".format(data_set)), 'wb+'))
    pickle.dump(paths, open(join(exp_dir, "paths_{}.pkl".format(data_set)), 'wb+'))
    # pickle.dump(distribution_x4, open(join(exp_dir, "padding_x4_{}.pkl".format(data_set)), 'wb+'))
    # pickle.dump(distribution_xc, open(join(exp_dir, "padding_xc_{}.pkl".format(data_set)), 'wb+'))
    pickle.dump(distribution_x4, open(join(exp_dir, "distribution_x4_{}.pkl".format(data_set)), 'wb+'))
    pickle.dump(distribution_xc, open(join(exp_dir, "distribution_xc_{}.pkl".format(data_set)), 'wb+'))
    # # # === 保存 pkl===
    #
    # # 保存 csv
    pd.DataFrame(y_true).to_csv(join(exp_dir, "targets_{}.csv".format(data_set)), index=None, header=None)
    pd.DataFrame(y_pred).to_csv(join(exp_dir, "predictions_{}.csv".format(data_set)), index=None, header=None)
    pd.DataFrame(paths).to_csv(join(exp_dir, "paths_{}.csv".format(data_set)), index=None, header=None)
    # pd.DataFrame(distribution_x4).to_csv(join(exp_dir, "padding_x4_{}.csv".format(data_set)), index=None, header=None)
    # pd.DataFrame(distribution_xc).to_csv(join(exp_dir, "padding_xc_{}.csv".format(data_set)), index=None, header=None)
    pd.DataFrame(distribution_x4).to_csv(join(exp_dir, "distribution_x4_{}.csv".format(data_set)), index=None, header=None)
    pd.DataFrame(distribution_xc).to_csv(join(exp_dir, "distribution_xc_{}.csv".format(data_set)), index=None, header=None)
    # pd.DataFrame(f_path).to_csv(join(exp_dir, "fpath_{}.csv".format(data_set)), index=None, header=None)
    # pd.DataFrame(f_sc).to_csv(join(exp_dir, "fscore_{}.csv".format(data_set)), index=None, header=None)
    # # 保存 csv

#     print("\nDataset {}\t{:.2f}\t{:.2f}\t{:.2f}\t\t{:.2f}\t{:.2f}".format(data_set, test_acc, test_f1, test_precision, test_recall, test_auc))
# for score in scores: print(score, end='\t')
print('\n')
# print('\n{:.2f}s used\n'.format(time.time() - begin))