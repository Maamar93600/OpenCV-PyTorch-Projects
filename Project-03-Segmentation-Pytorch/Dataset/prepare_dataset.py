import os
import shutil
import pandas as pd
import torch
from torch.utils.data import random_split


# Preparation of train, valid and test directories

class CREATE_DIRECTORY:

    def __init__(self, data_root, file_train_csv, file_test_csv):

        self.file_train_csv = file_train_csv
        self.file_test_csv = file_test_csv
        self.data_root = data_root

        self.dataframe_train = pd.read_csv(
            os.path.join(self.data_root, self.file_train_csv)
        )

        self.dataframe_test = pd.read_csv(
            os.path.join(self.data_root, self.file_test_csv)
        )

        self.id_file_train = [
            i.item() for i in self.dataframe_train.values
        ]

        self.id_file_test = [
            i.item() for i in self.dataframe_test.values
        ]

        generator = torch.manual_seed(42)

        train_test_split = [0.8, 0.2]

        train_set, valid_set = random_split(
            self.id_file_train,
            train_test_split,
            generator=generator
        )

        train_indice = train_set.indices
        valid_indice = valid_set.indices

        self.src_id_train_imgs = [
            self.id_file_train[i] for i in train_indice
        ]

        self.src_id_valid_imgs = [
            self.id_file_train[i] for i in valid_indice
        ]

        self.src_id_test_imgs = self.id_file_test


dataset = CREATE_DIRECTORY(
    "data",
    "train.csv",
    "test.csv"
)

id_train = dataset.src_id_train_imgs
id_valid = dataset.src_id_valid_imgs
id_test = dataset.src_id_test_imgs


# Create directories

os.makedirs(
    os.path.join("data_project", "train", "images"),
    exist_ok=True
)

os.makedirs(
    os.path.join("data_project", "train", "masks"),
    exist_ok=True
)

os.makedirs(
    os.path.join("data_project", "valid", "images"),
    exist_ok=True
)

os.makedirs(
    os.path.join("data_project", "valid", "masks"),
    exist_ok=True
)

os.makedirs(
    os.path.join("data_project", "test", "images"),
    exist_ok=True
)


# Copy files
from librairie import *


def move_file(directory, src_id):

    print(f"Copy files for {directory} in progress...")

    if directory != "test":

        for i in src_id:

            src_path_imgs = os.path.join(
                "data", "imgs", f"{int(i)}.jpg"
            )

            src_path_masks = os.path.join(
                "data", "masks", f"{int(i)}.png"
            )

            dst_path_imgs = os.path.join(
                "data_project", directory, "images"
            )

            dst_path_masks = os.path.join(
                "data_project", directory, "masks"
            )

            shutil.copy(src_path_imgs, dst_path_imgs)
            shutil.copy(src_path_masks, dst_path_masks)

    else:

        for i in src_id:

            src_path_imgs = os.path.join(
                "data", "imgs", f"{int(i)}.jpg"
            )

            dst_path_imgs = os.path.join(
                "data_project", directory, "images"
            )

            shutil.copy(src_path_imgs, dst_path_imgs)

    print("Finish...")


move_file("train", id_train)
move_file("valid", id_valid)
move_file("test", id_test)

