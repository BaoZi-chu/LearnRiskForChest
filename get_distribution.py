
import os, pickle, time, shutil
from os.path import join

from Efficientnet import efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
import torch
import torch.nn as nn
from Densenet import densenet121, densenet161, densenet169, densenet201
from Folder import ImageFolder
from Resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from Vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19_bn, vgg19
from torchvision import transforms
from tqdm import tqdm
from glob import glob
import argparse


begin = time.time()
torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('-c','--cnn',  default='r50', help='dataset dir')
parser.add_argument('-d','--dir',  default='get_distribution_hosp_new', help='dataset dir')
parser.add_argument('-s','--save_path',  default='hosp_test_d169_best_e128')
parser.add_argument('-m','--multiple', default=4, type=int, help='multiple of input size')
parser.add_argument('-g','--gpu',  default='0', help='set 0,1 to use multi gpu for example')
args = parser.parse_args()


# exp settings
cnn = args.cnn
datasets_dir = args.dir
exp_dir = "/home/ssd0/lfy/result_archive/{}".format(args.save_path)
batch_size = 4
# os.makedirs(exp_dir, exist_ok=True)
# shutil.copy('get_distribution.py', join(exp_dir, 'get_distribution.py'))


# data settings
data_dir = join("/home/ssd0/lfy/datasets", datasets_dir)
data_sets = ['train', "test"]
nb_class = len(os.listdir(join(data_dir, data_sets[0])))
re_size = 128 * args.multiple
crop_size = 112 * args.multiple


# CUDA setting
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


# Transforms
transform_test = transforms.Compose(
    [
        transforms.Resize((re_size, re_size)),
        # transforms.Resize(re_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


# Net settings
print('===== {} ====='.format(cnn))
model_zoo = {'r18':resnet18, 'r34':resnet34, 'r50':resnet50, 'r101':resnet101, 'r152':resnet152, 'd169':densenet169,
             'eb0':efficientnet_b0, 'eb1':efficientnet_b1, 'eb2':efficientnet_b2, 'eb3':efficientnet_b3,
             'eb4':efficientnet_b4, 'eb5':efficientnet_b5, 'eb6':efficientnet_b6,  'eb7':efficientnet_b7}
net = model_zoo[cnn](pretrained=True)


# if '_pre' not in args.save_path:
#     if cnn.startswith("r"):
#         net.fc = nn.Linear(net.fc.in_features, nb_class)  # for resnet
#         # net.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(net.fc.in_features, nb_class))  # for resnet
#     elif cnn.startswith("d"):
#         net.classifier = nn.Linear(net.classifier.in_features, nb_class)  # for densenet
#         # net.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(net.classifier.in_features, nb_class))  # for densenet
#     elif cnn.startswith('e'):
#         net.classifier._modules['1'] = nn.Linear(net.classifier._modules['1'].in_features, nb_class)
#     # net = torch.nn.DataParallel(net)
#     if 'model' in torch.load(join(exp_dir, "min_loss.pth")):
#         net.load_state_dict(torch.load(join(exp_dir, "min_loss.pth"))['model'])
#     else: net.load_state_dict(torch.load(join(exp_dir, "min_loss.pth")))

if not args.save_dir.endswith('_pre'):
    if cnn.startswith("r"):
        net.fc = nn.Linear(net.fc.in_features, nb_class)
    elif cnn.startswith('d'):
        net.classifier = nn.Linear(net.classifier.in_features, nb_class)
    elif cnn.startswith('e'):
        net.classifier._modules['1'] = nn.Linear(net.classifier._modules['1'].in_features, nb_class)
    elif cnn.startswith("a"):
        net.classifier._modules['6'] = nn.Linear(net.classifier._modules['6'].in_features, nb_class)
    print(join(exp_dir, "max_*"))
    net_file_name = glob(join(exp_dir, "max_*"))[0]
    # net_file_name = glob(join(exp_dir, "min_loss.pth"))[0]
    try:
        try: net.load_state_dict(torch.load(net_file_name)['model'])
        except: net.load_state_dict(torch.load(net_file_name))
    except:
        net = torch.nn.DataParallel(net)
        try: net.load_state_dict(torch.load(net_file_name)['model'])
        except: net.load_state_dict(torch.load(net_file_name))

net.cuda()
net.eval()


for data_set in data_sets:
    testset = ImageFolder(
        root=os.path.join(data_dir, data_set), transform=transform_test
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=6
    )
    distribution_x4 = []
    distribution_xc = []
    predictions = []
    labels = []
    paths = []
    test_loss = correct = total = 0

    with torch.no_grad():

        for _, (inputs, targets, path) in enumerate(tqdm(testloader, ncols=80)):
            inputs, targets = inputs.cuda(), targets.cuda()
            x4, xc = net(inputs)

            _, predicted = torch.max(xc.data, 1)

            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            distribution_x4.extend(x4.cpu().tolist())
            distribution_xc.extend(xc.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())
            labels.extend(targets.cpu().tolist())
            paths.extend(path)

    test_acc = 100.0 * float(correct) / total
    print("Dataset {}\tACC:{:.2f}\n".format(data_set, test_acc))

    with open(os.path.join(exp_dir, "{}_{:.2f}".format(data_set, test_acc)), "a+") as file:
        pass

    with open(join(exp_dir, "targets_{}.pkl".format(data_set)), 'wb+') as _:
        pickle.dump(labels, _)

    with open(join(exp_dir, "predictions_{}.pkl".format(data_set)), 'wb+') as _:
        pickle.dump(predictions, _)

    with open(join(exp_dir, "paths_{}.pkl".format(data_set)), 'wb+') as _:
        pickle.dump(paths, _)

    # with open(join(exp_dir, "distribution_x3_{}.pkl".format(data_set)), 'wb+') as _:
    #     pickle.dump(distribution_x3, _)

    with open(join(exp_dir, "distribution_x4_{}.pkl".format(data_set)), 'wb+') as _:
        pickle.dump(distribution_x4, _)

    with open(join(exp_dir, "distribution_xc_{}.pkl".format(data_set)), 'wb+') as _:
        pickle.dump(distribution_xc, _)

print('{:.2f}s used\n'.format(time.time() - begin))