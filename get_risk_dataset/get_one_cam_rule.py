import math
import os
from os.path import join

import numpy as np
import pandas as pd
import sklearn.metrics.pairwise as pairwise
from tqdm import tqdm, trange


def shorten_paths(paths):

    for i in range(len(paths)):
        paths[i] = "/".join(paths[i].split("/")[-4:])


# Calculate the center of high dimension data
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
    print("\n===== layer: {} =====".format(layer))
    print("Acc, D_Acc, d_right_p_wrong, d_wrong_p_right")
    # Calculate the class centers by train data

    count_all=[]
    for data_set in data_sets:
        # for data_set in ['train_more', 'val_less', 'test']:

        # Read coordinate and label of data
        coordinates = pd.read_csv(
            os.path.join(csv_dir, "cam_{}.csv".format(data_set)),
            header=None,
        ).to_numpy()
        coordinates_re = pd.read_csv(
            os.path.join(csv_dir, "recam_{}.csv".format(data_set)),
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

        # Calculate the distance to each class center of every data
        # metric = euclidean, cosine, braycurtis, seuclidean, canberra

        fangcha = get_center(coordinates)
        refangcha=get_center(coordinates_re)
        # pd.DataFrame(distance_to_center).to_csv(os.path.join(csv_dir, '{}_distance_{}.csv' \
        #                                                       .format(layer, data_set)), header=None, index=None)

        # Evaluate distance

        # Insert path and label of data to csv
        print('dis')
        print(len(fangcha))
        print('path')
        print(len(paths))
        # print(fangcha)

        # for i in range(len(fangcha)):
        #     fangcha[i].insert(0, paths[i])
        #     fangcha[i].insert(1, labels[i])

        # for i in range(len(fangcha)):
        #     fangcha[i].insert(0, paths[i])
        #     fangcha[i].insert(1, labels[i])
        temp_csv = []
        for i in range(len(paths)):
            for j in range(num_class):
                temp_csv.append(
                    [
                        "{}_{:0>3d}".format(paths[i], j),
                        "{}".format(1 if j == labels[i] else 0),
                        fangcha[i] if j == labels[i] else refangcha[i]
                    ]
                )

        exec("distance_to_center_{} = temp_csv".format(data_set))

    # Merge 3 csvs together
    distance_center_all = []

    for data_set in data_sets:
        # for data_set in ['train_more', 'val_less', 'test']:
        exec("distance_center_all.extend(distance_to_center_{})".format(data_set))

    # Create the header of csv
    header = ["data", "label","{}_{}_fangcha".format(elem_name,'xc')]


    # Save the final csv
    distance_center_all.insert(0, header)
    pd.DataFrame(distance_center_all).to_csv(
        os.path.join(csv_dir, "{}_{}_one_fangcha.csv".format(elem_name, 'xc')),
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
csv_dir_str = '/home/ssd0/lfy/result_archive/chest_hosp_seg_{}/'


if __name__ == "__main__":

    for layer in layers:

        for cnn in cnns:
            print("===== CNN: {} =====".format(cnn))

            elem_name = elem_name_str.format(cnn)
            csv_dir = csv_dir_str.format(cnn)
            num_class = 2
            get_distance(elem_name, num_class, csv_dir, "cosine")

    print("Done.")
