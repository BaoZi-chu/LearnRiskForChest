import math
import os
from os.path import join

import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as pairwise
from tqdm import tqdm, trange
from  time import time


def shorten_paths(paths):

    for i in range(len(paths)):
        paths[i] = "/".join(paths[i].split("/")[-4:])


# Calculate the center of high dimension data
def get_knn(k, list, label_train):
    list_sorted = sorted(list)
    knn = []

    for i in range(k):
        index = list.index(list_sorted[i+1])
        knn.append(label_train[index])

    return knn

def get_knn1(k, list, label_train):
    list_sorted = sorted(list)
    knn = []

    for i in range(k):
        index = list.index(list_sorted[i+1])
        knn.append(label_train[index])

    return knn

def get_sum(data_list):
    center = []
    coordinate = 0
    c = []
    count=0
    for data in data_list:

        for i in range(7):
            data[i]= data[i].strip('[]')
            data[i]=data[i].split(',')
            # for j in range(7):
            #     data[i][j]=data[i][j].astype(float)
            data[i]=np.array(data[i])
            data[i]=data[i].astype(float)
            if i==0:
                d=data[i]
            else:
                d=np.vstack((d,data[i]))
        c.append(d)
        count+=1
        if count==5:
            count=0
            c=np.array(c)
            c=np.mean(c,0).flatten()
            center.append(c)
            c = []

    # print(center)
    print(len(center))
    return center

def get_center(data_list):
    center = []
    coordinate = 0
    count=0
    for data in data_list:

        for i in range(7):
            data[i]= data[i].strip('[]')
            data[i]=data[i].split(',')
            # for j in range(7):
            #     data[i][j]=data[i][j].astype(float)
            data[i]=np.array(data[i])
            data[i]=data[i].astype(float)
            if i==0:
                d=data[i]
            else:
                d=np.vstack((d,data[i]))
        coordinate += np.var(d)
        count+=1
        if count==5:
            count=0
            coordinate=coordinate*2
            center.append(coordinate)
            coordinate=0
    # print(center)
    print(len(center))
    return center

def get_all(data_list):
    center = []
    coordinate = 0
    c = []
    count=0
    for data in data_list:

        for i in range(7):
            data[i]= data[i].strip('[]')
            data[i]=data[i].split(',')
            # for j in range(7):
            #     data[i][j]=data[i][j].astype(float)
            data[i]=np.array(data[i])
            data[i]=data[i].astype(float)
            if i==0:
                d=data[i]
            else:
                d=np.vstack((d,data[i]))
        c.append(d)
        count+=1
        if count==5:
            count=0
            c=np.array(c)
            c=c.flatten()
            center.append(c)
            c = []

    # print(center)
    print(len(center))
    return center


def take_0(elem):
    return elem[0]


def point_min(list):
    min_index = list.index(min(list))

    for i in range(len(list)):

        if i == min_index:
            list[i] = 1
        else:
            list[i] = 0

    return list


def eval_distance(data_set, layer, labels, predictions, distances):
    # Evaluate prediction
    correct = 0
    for i in range(len(labels)):
        if predictions[i] == labels[i]:
            correct += 1

    acc = correct / len(labels)

    # Evaluate distance
    d_predictions = []

    for info in distances:
        # info = list(map(float, info))
        d_predictions.append(info.index(min(info)))

    d_correct = 0

    for i in range(len(labels)):
        if d_predictions[i] == labels[i]:
            d_correct += 1

    d_acc = d_correct / len(labels)

    # Calculate different wrong
    evaluation = []

    for i in range(len(labels)):
        evaluation.append([labels[i], predictions[i], d_predictions[i]])

    p_wrong = d_wrong = 0

    for data_labels in evaluation:

        if data_labels[1] != data_labels[2]:

            if data_labels[1] == data_labels[0]:
                d_wrong += 1
            elif data_labels[2] == data_labels[0]:
                p_wrong += 1

    # print('Dataset: {}, Layer: {}'.format(data_set, layer))
    # print('Acc, D_Acc, d_right_p_wrong, p_right_d_wrong')
    print("{:.2f}, {:.2f}, {}, {}".format(acc * 100, d_acc * 100, p_wrong, d_wrong))
    # print('{:.2f}'.format(d_acc * 100))


# Calculate the distance to each class center of every data
def get_distance(
     elem_name, num_class, csv_dir, metric, data_sets=["train", "val", "test"]
):
    layer='xc'
    print("\n===== layer: {} =====".format(layer))
    print("Acc, D_Acc, d_right_p_wrong, d_wrong_p_right")
    # Calculate the class centers by train data

    k_list = [3, 5]
    for k in k_list:
        start = time()
        count_all = []
        for data_set in data_sets:
            # for data_set in ['train_more', 'val_less', 'test']:

            # Read coordinate and label of data
            labels_train = (
                pd.read_csv(
                    join(csv_dir, "targets_{}.csv".format(data_sets[0])), header=None
                )
                    .to_numpy()
                    .flatten()
            )
            coordinates = pd.read_csv(
                os.path.join(csv_dir, "cam_{}.csv".format(data_set)),
                header=None,
            ).to_numpy()

            labels = (
                pd.read_csv(
                    os.path.join(csv_dir, "targets_{}.csv".format(data_set)), header=None
                )
                    .to_numpy()
                    .flatten()
            )
            paths = (
                pd.read_csv(
                    os.path.join(csv_dir, "paths_{}.csv".format(data_set)), header=None
                )
                    .to_numpy()
                    .flatten()
            )
            shorten_paths(paths)
            predictions = (
                pd.read_csv(
                    os.path.join(csv_dir, "predictions_{}.csv".format(data_set)),
                    header=None,
                )
                    .to_numpy()
                    .flatten()
            )

            p = np.array(get_sum(coordinates))
            # Calculate the distance to each class center of every data
            # metric = euclidean, cosine, braycurtis, seuclidean, canberra
            pairwise_distances = pairwise.pairwise_distances(
                p, p, metric=metric
            ).tolist()

            knn = []

            for i in range(len(pairwise_distances)):
                if data_set == 'train':
                    knn.append(get_knn1(k, pairwise_distances[i], labels_train))
                else:
                    knn.append(get_knn(k, pairwise_distances[i], labels_train))

            # Get knn count for all
            count = []
            print(num_class)
            for i in range(len(knn)):
                line = np.bincount(knn[i], minlength=num_class).tolist()
                count.append(line)
                # Combine all knn_count
            for i in range(len(count)):
                    count[i].insert(0, paths[i])
                    count[i].insert(1, labels[i])

            count_all.extend(count)

        # Combine all knn_count
        temp_csv = []
        # fangc=[]
        # for i in range(len(fangcha)):
        #     f=[]
        #     f.append(fangcha[i])






        # Create the header of csv
        header = ["data", "label"]

        for i in range(num_class):
            header.append("{}_{}_class_{:0>3d}_xs{}".format(elem_name, layer, i, k))

        # Save the final csv
        count_all.insert(0, header)
        pd.DataFrame(count_all).to_csv(
            os.path.join(csv_dir, "{}_{}_xs{}.csv".format(elem_name, layer, k)),
            header=None,
            index=None,
        )


    # print('{}_{}_distance_to_center_all.csv saved at {}\n'.format(elem_name, layer, csv_dir))

    # # Get nearest class csv
    # n_distance_center_all = []
    #
    # for i in range(len(distance_center_all) - 1):
    #     info = distance_center_all[i + 1]
    #     raw_distance = info[2:]
    #     n_info = [info[0], info[1]]
    #     n_info.extend(point_min(raw_distance))
    #     n_distance_center_all.append(n_info)
    #
    # n_header = ['data', 'label']
    #
    # for i in range(num_class):
    #     n_header.append('{}_{}_class_{:0>3d}_is_nearest'.format(elem_name, layer, i))
    #
    # n_distance_center_all.insert(0, n_header)
    #
    # pd.DataFrame(n_distance_center_all).to_csv(os.path.join(csv_dir, '{}_{}_min_distance.csv' \
    #                                                       .format(elem_name, layer)), header=None, index=None)


# settings
metrics = [
    "cityblock",
    "cosine",
    "euclidean",
    "l1",
    "l2",
    "manhattan",
    "braycurtis",
    "canberra",
    "correlation",
    "minkowski",
    "seuclidean",
    "sqeuclidean",
]
# metrics = ['cosine', 'correlation', 'braycurtis', 'canberra', 'seuclidean']
# layer = 'x'
data_sets = ["trainval", "test"]

cnns = ["r50"]
layers = ['x4', "xc"]
elem_name_str = "{}"
csv_dir_str = '/home/ssd0/lfy/result_archive/chest_hosp_{}/'


if __name__ == "__main__":

    for layer in layers:

        for cnn in cnns:
            print("===== CNN: {} =====".format(cnn))

            elem_name = elem_name_str.format(cnn)
            csv_dir = csv_dir_str.format(cnn)
            num_class = (
                int(
                    pd.read_csv(join(csv_dir, "predictions_train.csv"), header=None)
                    .to_numpy()
                    .flatten()[-1]
                )
                + 1
            )
            num_class=2
            get_distance(elem_name, num_class, csv_dir, "cosine")

    print("Done.")
