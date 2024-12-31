from __future__ import print_function, division

import matplotlib
import pandas as pd
import numpy as np
from numpy import *
import matplotlib.pyplot as plt

import torchvision.models.segmentation
import torchvision.transforms as tf

import torch
import torch.nn as nn
import torch.nn.functional as F

# import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image

# from PIL import Image
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import multivariate_normal

from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split, SubsetRandomSampler, ConcatDataset, \
    Subset

from torch.distributions import multivariate_normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import normal

import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import math
import seaborn as sns

import sys
from torchmetrics.classification import AUROC
from sklearn import metrics
from itertools import cycle

from datetime import datetime
import fnmatch
from fnmatch import fnmatchcase
from PIL import Image
import transformers
import accelerate
import peft

from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, ViTForImageClassification
from peft import LoraConfig, LoraModel

from utils.common_utils import set_seed, print_training_progress, print_trainable_parameters
from utils.train_utils import train_model, plot_loss, test_stage

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # #  Important Parameters # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
seed = 0
rank_LoRa = 4  # can set to 8, 16 for other experiments

work_id = 240211
seed_list = [seed]

k = 5  # <<<<<< KFOLD <<<<<<
batch_size = 32  # <<<<<< BATCH SIZE <<<<<<
num_epochs = 20  # 150 # <<<<<< EPOCH <<<<<<

validation_split = 0.25
shuffle_train_set = True
train_ratio = 0.8

learning_rate = 0.001
lr_decay_step_size = 20
lr_decay_prop = 0.5

cudnn.benchmark = True
plt.ion()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Define the seeds

set_seed(seed)
# 创建保存路径
simu_warmup_fold = "./Simu/"
copula_fold = "./Copula/"

os.makedirs(simu_warmup_fold, exist_ok=True)  # 存在路径则不创建也不报错
os.makedirs(copula_fold, exist_ok=True)  # 存在路径则不创建也不报错
# # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # #  Handling Input  # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

if __name__ == "__main__":
    df = pd.read_csv('label-5246.csv')
    filename = df['filename']

    width = height = 224
    width_raw = height_raw = 512
    # width=height=128

    transformImg = tf.Compose([tf.ToPILImage(),
                               tf.Resize((height, width)),
                               tf.ToTensor(),
                               ])  # Set image transformation

    Z = df[['gender', 'age']]
    Y = df[['al', 'class']]
    # Y = df[['al','se']]
    al = df[['al']]
    se = df[['se']]
    high = df[['class']]
    age = df[['age']]
    gender = df[['gender']]
    ##########################################################
    X_length = 2623
    N_2623 = 5246

    X_left_raw = torch.empty(X_length, 3, height_raw, width_raw)
    X_right_raw = torch.empty(X_length, 3, height_raw, width_raw)
    X_left = torch.empty(X_length, 3, height, width)
    X_right = torch.empty(X_length, 3, height, width)

    Age = torch.empty(X_length, 1)
    Gender = torch.empty(X_length, 1)

    Y_left_myopia = torch.empty(X_length, 1)
    Y_right_myopia = torch.empty(X_length, 1)
    Y_left_al = torch.empty(X_length, 1)
    Y_left_se = torch.empty(X_length, 1)
    Y_right_al = torch.empty(X_length, 1)
    Y_right_se = torch.empty(X_length, 1)

    ##########################################################

    YL = torch.empty((X_length, 2))

    j = 0
    for i in range(0, X_length):
        if fnmatchcase(filename[j], '*-L*'):
            X_left_raw[i] = transforms.ToTensor()(Image.open("D:/Code/Copula/5246/" + filename[j]))
            X_left[i] = transformImg(X_left_raw[i])
            Y_left_al[i] = torch.from_numpy(al.values[j])
            Y_left_se[i] = torch.from_numpy(se.values[j])
            Y_left_myopia[i] = torch.from_numpy(high.values[j])

            X_right_raw[i] = transforms.ToTensor()(Image.open("D:/Code/Copula/5246/" + filename[j + 1]))
            X_right[i] = transformImg(X_right_raw[i])
            Y_right_al[i] = torch.from_numpy(al.values[j + 1])
            Y_right_se[i] = torch.from_numpy(se.values[j + 1])
            Y_right_myopia[i] = torch.from_numpy(high.values[j + 1])

            Age[i] = torch.from_numpy(age.values[j])
            Gender[i] = torch.from_numpy(gender.values[j])

            j = j + 2
        elif fnmatchcase(filename[j], '*-R*'):
            X_right_raw[i] = transforms.ToTensor()(Image.open("D:/Code/Copula/5246/" + filename[j]))
            X_right[i] = transformImg(X_right_raw[i])
            Y_right_al[i] = torch.from_numpy(al.values[j])
            Y_right_se[i] = torch.from_numpy(se.values[j])
            Y_right_myopia[i] = torch.from_numpy(high.values[j])

            X_left_raw[i] = transforms.ToTensor()(Image.open("D:/Code/Copula/5246/" + filename[j + 1]))
            X_left[i] = transformImg(X_left_raw[i])
            Y_left_al[i] = torch.from_numpy(al.values[j + 1])
            Y_left_se[i] = torch.from_numpy(se.values[j + 1])
            Y_left_myopia[i] = torch.from_numpy(high.values[j + 1])

            Age[i] = torch.from_numpy(age.values[j])
            Gender[i] = torch.from_numpy(gender.values[j])

            j = j + 2
        else:
            print(f"i: {i}")
            print(f"j: {j}")
            print("ERROR when creating X and Y!!!!!!!!!!!!")

    ds_XL = TensorDataset(X_left.float(),
                          X_right.float(),
                          Age.float(),
                          Gender.float(),
                          torch.tensor(Y_left_myopia).float(),
                          torch.tensor(Y_right_myopia).float(),
                          torch.tensor(Y_left_al).float(),
                          torch.tensor(Y_left_se).float(),
                          torch.tensor(Y_right_al).float(),
                          torch.tensor(Y_right_se).float())

    ###########################################################
    # 输出生成的ds_XL供检查
    data = []
    for i in range(len(ds_XL)):
        sample = ds_XL[i]
        data.append([sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], sample[6],
                     sample[7], sample[8], sample[9]])

    df = pd.DataFrame(data,
                      columns=["X_left", "X_right", "Age", "Gender",
                               "Y_left_myopia", "Y_right_myopia",
                               "Y_left_al", "Y_left_se", "Y_right_al", "Y_right_se"])
    # df.to_csv('TESTPREV.csv', index=False)

    ###########################################################
    # 检查并调试 左右眼 al se myopia 之间的相关性
    pd.set_option('display.max_columns', None)  # Set maximum number of columns to display
    pd.set_option('display.max_colwidth', None)  # Set column width to display
    df_respones = df[["Y_left_al", "Y_left_se", "Y_left_myopia", "Y_right_al", "Y_right_se", "Y_right_myopia"]]
    # print(df_respones)
    df_numeric = df_respones.applymap(lambda x:
                                      x.detach().flatten().numpy()[0] if isinstance(x,
                                                                                    torch.Tensor) else x)  # convert each tensor item to a 1D array (numpy array)
    correlation_matrix = np.corrcoef(df_numeric, rowvar=False)
    correlation_df = pd.DataFrame(correlation_matrix, columns=df_respones.columns, index=df_respones.columns)
    print('CorrMat of : ');
    print(correlation_matrix)

    # 查看各列的均值
    print("Mean of: ")
    mean_values = df[df_respones.columns].apply(
        lambda col: np.mean([x.item() if isinstance(x, torch.Tensor) else x for x in col]))
    print(mean_values)

    # Plot the corr heatmap
    sns.set()
    plt.figure(figsize=(8, 8))
    sns.heatmap(correlation_df, annot=True, cmap="coolwarm", square=True)
    plt.title("Correlation Matrix Heatmap")
    plt.savefig(simu_warmup_fold + f"WorkID{work_id}_Simu_heatmap.png")  # Save the plot to an image file

    # 存储数据集
    PATH_ds = f'data_processed.pt'
    torch.save(ds_XL, PATH_ds)
