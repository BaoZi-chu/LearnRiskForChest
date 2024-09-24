import math
import os
from os.path import join

import numpy as np
from time import time
import pandas as pd
import sklearn.metrics.pairwise as pairwise
from sklearn.neighbors import NearestCentroid
from tqdm import tqdm, trange


def shorten_paths(paths):

    for i in range(len(paths)):
        paths[i] = "/".join(paths[i].split("/")[-4:])


# Calculate the center of high dimension data
def get_center(data_list):
    center = []

    for d in range(len(data_list[0])):  # d for dimension
        coordinate = 0

        for data in data_list:
            coordinate += data[d]

        coordinate /= len(data_list)
        center.append(coordinate)

    return center


def take_0(elem):
    return elem[0]
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

def point_min(list):
    min_index = list.index(min(list))

    for i in range(len(list)):

        if i == min_index:
            list[i] = 1
        else:
            list[i] = 0

    return list


# Calculate the distance to each class center of every data
def get_one_distance(
     elem_name, num_class, csv_dir, metric, data_sets=["train", "val", "test"]
):
    start = time()
    layer = 'xc'
    # Calculate the class centers by train data
    coordinate_train = pd.read_csv(
        os.path.join(csv_dir, "cam_{}.csv".format(data_sets[0])),
        header=None,
    ).to_numpy()
    label = pd.read_csv(
        os.path.join(csv_dir, "targets_{}.csv".format(data_sets[0])), header=None
    ).to_numpy().flatten()
    coordinate_train = np.array(get_sum(coordinate_train))
    nc = NearestCentroid(metric=metric).fit(coordinate_train, label)
    centers = nc.centroids_

    for data_set in data_sets:
        # Read coordinate and label of data
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
        coordinates = np.array(get_sum(coordinates))

        # Calculate the distance to each class center of every data
        distance_to_center = pairwise.pairwise_distances(
            coordinates, centers, metric=metric
        ).tolist()

        # Insert path and label of data to csv
        temp_csv = []
        for i in range(len(paths)):

            for j in range(num_class):
                temp_csv.append(
                    [
                        "{}_{:0>3d}".format(paths[i], j),
                        "{}".format(1 if j == labels[i] else 0),
                        distance_to_center[i][j],
                    ]
                )

        exec("distance_to_center_{} = temp_csv".format(data_set))


    # Merge 3 csvs together
    distance_center_all = []

    for data_set in data_sets:
        # for data_set in ['train_more', 'val_less', 'test']:
        exec("distance_center_all.extend(distance_to_center_{})".format(data_set))

    # Create the header of csv
    header = ["data", "label", "{}_{}_xsdistance".format(elem_name, layer)]

    # Save the final csv
    distance_center_all.insert(0, header)
    pd.DataFrame(distance_center_all).to_csv(
        os.path.join(csv_dir, "{}_{}_one_xsdis.csv".format(elem_name, layer)),
        header=None, index=None
    )

    print("--- distance {:.2f} s ---".format(time() - start))


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
csv_dir_str = '/home/ssd0/lfy/result_archive/chest_covid100_{}/'


if __name__ == "__main__":

    for layer in layers:

        for cnn in cnns:
            print("===== CNN: {} =====".format(cnn))

            elem_name = elem_name_str.format(cnn)
            csv_dir = csv_dir_str.format(cnn)
            num_class = 2
            get_one_distance(elem_name, num_class, csv_dir, "cosine")

    print("Done.")
