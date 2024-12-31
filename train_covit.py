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
from datetime import datetime

# from PIL import Image
from sklearn.model_selection import KFold, train_test_split

from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split, SubsetRandomSampler, ConcatDataset, \
    Subset

import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import sys
from tqdm import tqdm

from transformers import AutoModelForImageClassification, TrainingArguments, Trainer, ViTForImageClassification
from peft import LoraConfig, LoraModel

from utils.common_utils import set_seed, print_training_progress, print_trainable_parameters
from utils.train_utils_normal import train_model, plot_loss, test_stage
from modeling.module1 import CoViT, CoViT_Normal
from utils.train_stage23_utils_normal import train_model_stage23, plot_loss_stage23, test_stage_stage23
from utils.loss_normal import calculate_sigma_hat
from utils.loss_normal import parametricLoss

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # #  Important Parameters # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

rank_LoRa = 8  # can set to 8, 16 for other experiments

work_id = 122801
seed_list = [0, 42, 84]
model_name = 'CoViT'

k = 5  # <<<<<< KFOLD <<<<<<
batch_size = 32  # <<<<<< BATCH SIZE <<<<<<
num_epochs = 12  # 150 # <<<<<< EPOCH <<<<<<

validation_split = 0.25
shuffle_train_set = True
train_ratio = 0.8

X_length = 2623
learning_rate = 0.001
lr_decay_step_size = 2
lr_decay_prop = 0.9

cudnn.benchmark = True
plt.ion()

device = torch.device('cuda:0')

# 创建保存路径
simu_warmup_fold = "./Simu/"
copula_fold = "./Copula/"

os.makedirs(simu_warmup_fold, exist_ok=True)  # 存在路径则不创建也不报错
os.makedirs(copula_fold, exist_ok=True)  # 存在路径则不创建也不报错


if __name__ == "__main__":
    for seed in seed_list:
        set_seed(seed)
        ds_XL = torch.load('data_processed.pt')
        # K fold
        since = time.time()  # 运行起始时间

        kfold = KFold(n_splits=k, shuffle=True, random_state=seed)

        history = {'avg_loss_left_high': [],
                   'avg_loss_right_high': [],
                   'avg_loss_left_AL': [],
                   'avg_loss_right_AL': [],
                   'avg_loss_left': [],
                   'avg_loss_right': [],
                   'avg_loss_high': [],
                   'avg_loss_AL': [],
                   'avg_loss': [],
                   'classification_accuracy_left': [],
                   'classification_accuracy_right': [],
                   'AUC_left': [],
                   'AUC_right': []}

        # Print at the top banner
        print('=' * 48)
        batch_num_per_epoch = X_length * train_ratio * (1 - validation_split) // batch_size
        print(
            "    ooo " + " " + "u   u" + " " + " ccc " + " " + " ooo " + " " + "pppp " + " " + "u   u" + " " + "l  " + " " + "   a")
        print(
            "  o   o" + " " + "u   u" + " " + "c    " + " " + "o   o" + " " + "p   p" + " " + "u   u" + " " + "l   " + " " + "  a a")
        print(
            " o   o" + " " + "u   u" + " " + "c    " + " " + "o   o" + " " + "pppp " + " " + "u   u" + " " + "l   " + " " + " aaaaa")
        print(
            " ooo " + " " + " uuu " + " " + " ccc " + " " + " ooo " + " " + "p    " + " " + " uuu " + " " + "llll" + " " + "a     a")
        print('=' * 48)
        print(f"WorkID: {work_id}")
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        print("This experiment started at:", formatted_time)
        print('=' * 48)
        print("Important parameters: ")
        print("\tPython version:", sys.version)
        print("\tPytorch version: ", torch.__version__)
        print("\tBatch num per epoch: ", int(batch_num_per_epoch))
        print("\tlearning_rate: ", learning_rate)
        print("\tLearning rate decay every {} epoches.".format(lr_decay_step_size))
        print("Other parameters: ")
        print("\tRandom state KFold: ", seed)
        print("\tSeed: ", seed)
        print("\tKFold: ", k)
        print("\tBatch size: ", batch_size)
        print("\tNumber of epochs: ", num_epochs)
        print("\tSample size: ", X_length)
        # print("\tResNet输出的向量的维度: ", resnet_output_size)
        print("\tshuffle_train_set: ", shuffle_train_set)
        print("\ttrain_ratio (prop of train+val): ", train_ratio)
        print("\tvalidation_split: ", validation_split)
        print('=' * 48 + '\n')

        for fold, (train_idx, test_idx) in enumerate(kfold.split(np.arange(X_length))):
            # For loss plot
            train_loss_batchwise_allfold = []
            val_loss_batchwise_allfold = []
            test_loss_batchwise_allfold = []

            print('=' * 60)
            print('Fold {}'.format(fold + 1))
            print('=' * 60)

            # Check train_idx, test_idx
            print("train_idx: ", train_idx[:20])
            print("test_idx: ", test_idx[:20])

            split = int(np.floor(validation_split * len(train_idx)))
            if shuffle_train_set:
                np.random.shuffle(train_idx)
            train_indices, val_indices = train_idx[split:], train_idx[:split]

            # 上述代码会出现loss锯齿下降，使用下述代码不会出现，而是正常的锯齿下降
            train_dataset = Subset(ds_XL, train_indices)
            val_dataset = Subset(ds_XL, val_indices)
            test_dataset = Subset(ds_XL, test_idx)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                      num_workers=2,
                                      pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

            # 存储划分的数据集以便copula直接调用，从源头防止数据泄露！ @2023-09-27
            # warmup和copula我的batch size设置的也必须是一样的
            """PATH_dl_train = f'train_data_save/WorkID{work_id}_RC_{model_name}_Train_Data_seed{seed}_fold{fold}_X_length{X_length}_batchsize{batch_size}_num_epoch{num_epochs}.pt'
            torch.save(train_loader, PATH_dl_train)

            PATH_dl_val = f'train_data_save/WorkID{work_id}_RC_{model_name}_Val_Data_seed{seed}_fold{fold}_X_length{X_length}_batchsize{batch_size}_num_epoch{num_epochs}.pt'
            torch.save(val_loader, PATH_dl_val)

            PATH_dl_test = f'train_data_save/WorkID{work_id}_RC_{model_name}_Test_Data_seed{seed}_fold{fold}_X_length{X_length}_batchsize{batch_size}_num_epoch{num_epochs}.pt'
            torch.save(test_loader, PATH_dl_test)"""

            covit = CoViT_Normal(device=device, rank_LoRa=rank_LoRa).to(device)
            print("parameter count for CoViT: ")
            print_trainable_parameters(covit)
            gpu_ids = ['cuda:0']
            covit = nn.DataParallel(covit, device_ids=gpu_ids, output_device=gpu_ids[0])

            gamma12, gamma34, gamma3412, sigma1, sigma2, gamma = calculate_sigma_hat(train_loader, covit, device)

            # 输出 Gamma and std_ei 供检查
            print("CopulaPara: ")
            print("gamma12: ", gamma12)
            print("gamma34: ", gamma34)
            print("gamma3412:", gamma3412)
            print("sigma1:", sigma1)
            print("sigma2:", sigma2)
            print()

            # Criterion
            # criterion_classification = nn.BCEWithLogitsLoss()
            criterion_classification = nn.BCELoss()
            criterion_regression = nn.MSELoss()
            criterion_copula = parametricLoss()

            # Observe that all parameters are being optimized
            optimizer = optim.AdamW(covit.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                    weight_decay=0.01, amsgrad=False)

            # Decay LR by a factor of ~ every ~ epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_prop)

            copula_test, train_loss_batchwise, val_loss_batchwise, test_loss_batchwise = train_model_stage23(covit,
                                                                                                             criterion_copula,
                                                                                                             criterion_regression,
                                                                                                             criterion_classification,
                                                                                                             optimizer,
                                                                                                             exp_lr_scheduler,
                                                                                                             train_loader,
                                                                                                             val_loader,
                                                                                                             test_loader,
                                                                                                             num_epochs,
                                                                                                             gamma12,
                                                                                                             gamma34,
                                                                                                             gamma3412,
                                                                                                             sigma1,
                                                                                                             sigma2,
                                                                                                             gamma,
                                                                                                             device,
                                                                                                             since,
                                                                                                             k, fold,
                                                                                                             work_id,
                                                                                                             seed,
                                                                                                             X_length,
                                                                                                             batch_size,
                                                                                                             exp_lr_scheduler,
                                                                                                             learning_rate)






            # network参数
            covit = CoViT_Normal(device=device, rank_LoRa=rank_LoRa).to(device)
            gpu_ids = ['cuda:0']
            covit = nn.DataParallel(covit, device_ids=gpu_ids, output_device=gpu_ids[0])

            print(f"parameter count for {model_name}: ")
            print_trainable_parameters(covit)

            # Criterion
            criterion_classification = nn.BCELoss().to(device)
            criterion_regression = nn.MSELoss().to(device)

            # Observe that all parameters are being optimized
            optimizer = optim.AdamW(covit.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                    weight_decay=0.01, amsgrad=False)

            # Decay LR by a factor of ~ every ~ epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_prop)

            # ====== 训练模型 ======
            covit_test, train_loss_batchwise, val_loss_batchwise, test_loss_batchwise = train_model(covit,
                                                                                                    criterion_regression,
                                                                                                    criterion_classification,
                                                                                                    optimizer,
                                                                                                    exp_lr_scheduler,
                                                                                                    train_loader,
                                                                                                    val_loader,
                                                                                                    test_loader,
                                                                                                    num_epochs,
                                                                                                    device,
                                                                                                    since,
                                                                                                    k,
                                                                                                    fold)
            # Now oucopula is the best parameter setting acquired in this fold
            # Record batchwise for all folds to plot the loss plot for whole training process
            train_loss_batchwise_allfold = [*train_loss_batchwise_allfold, *train_loss_batchwise]
            val_loss_batchwise_allfold = [*val_loss_batchwise_allfold, *val_loss_batchwise]
            test_loss_batchwise_allfold = [*test_loss_batchwise_allfold, *test_loss_batchwise]

            # Plot train val test loss
            print('-' * 100)
            print("Train, validation, & test loss per 20 epoch")
            plot_loss(batch_num_per_epoch, train_loss_batchwise_allfold, val_loss_batchwise_allfold,
                      test_loss_batchwise_allfold, work_id, seed, fold, X_length, batch_size, num_epochs)

            # ====== 记录模型 ======
            # For 本地
            warm_up_PATH = f'model_save/WorkID{work_id}_R{rank_LoRa}_{model_name}_seed{seed}_fold{fold}_X_length{X_length}_batchsize{batch_size}_num_epoch{num_epochs}.pth'

            # Save the best parameter setting in warm_up_PATH
            torch.save(covit_test.module.state_dict(), warm_up_PATH)  # 模型提取 state_dict状态字典 并保存

            # ====== 测试模型 ======
            net_test = CoViT(device).to(device)  # 载入模型之前先初始化
            net_test.load_state_dict(torch.load(warm_up_PATH))  # 载入模型：先从PATH载入状态字典，再load_state_dict

            # Test the best model in this fold
            loss_summary = test_stage(net_test, criterion_regression, criterion_classification, test_loader, device)

            # Record the number into the dict
            for key, val in loss_summary.items():
                if torch.is_tensor(val):
                    val = val.item()
                history[key].append(val)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        history['mse_left_AL'] = history["avg_loss_left_AL"]
        history['mse_right_AL'] = history["avg_loss_right_AL"]

        # 输出到csv上的结果，我们只基于test dataset
        pd.DataFrame(history).to_csv(
            f'./Simu/WorkID{work_id}_Warmup_R{rank_LoRa}_{model_name}_seed{seed}_X_length{X_length}_batchsize{batch_size}_num_epoch{num_epochs}.csv',
            index=True)


        """第二阶段和第三阶段"""

        since = time.time()  # 运行起始时间
        learning_rate = 0.0001

        kfold = KFold(n_splits=k, shuffle=True, random_state=seed)

        history = {'avg_COPULA_loss': [],
                   'avg_loss_left_high': [],
                   'avg_loss_right_high': [],
                   'avg_loss_left_AL': [],
                   'avg_loss_right_AL': [],
                   'avg_loss_left': [],
                   'avg_loss_right': [],
                   'avg_loss_high': [],
                   'avg_loss_AL': [],
                   'avg_loss': [],
                   'classification_accuracy_left': [],
                   'classification_accuracy_right': [],
                   'AUC_left': [],
                   'AUC_right': []}

        # Print at the top banner
        print('=' * 48)
        batch_num_per_epoch = X_length * train_ratio * (1 - validation_split) // batch_size
        print(
            "    ooo " + " " + "u   u" + " " + " ccc " + " " + " ooo " + " " + "pppp " + " " + "u   u" + " " + "l  " + " " + "   a")
        print(
            "  o   o" + " " + "u   u" + " " + "c    " + " " + "o   o" + " " + "p   p" + " " + "u   u" + " " + "l   " + " " + "  a a")
        print(
            " o   o" + " " + "u   u" + " " + "c    " + " " + "o   o" + " " + "pppp " + " " + "u   u" + " " + "l   " + " " + " aaaaa")
        print(
            " ooo " + " " + " uuu " + " " + " ccc " + " " + " ooo " + " " + "p    " + " " + " uuu " + " " + "llll" + " " + "a     a")
        print('=' * 48)
        print(f"WorkID: {work_id}")
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
        print("This experiment started at:", formatted_time)
        print('=' * 48)
        print("Important parameters: ")
        print("\tPython version:", sys.version)
        print("\tPytorch version: ", torch.__version__)
        print("\tBatch num per epoch: ", int(batch_num_per_epoch))
        print("\tlearning_rate: ", learning_rate)
        print("\tLearning rate decay every {} epoches.".format(lr_decay_step_size))
        print("Other parameters: ")
        print("\tRandom state KFold: ", seed)
        print("\tSeed: ", seed)
        print("\tKFold: ", k)
        print("\tBatch size: ", batch_size)
        print("\tNumber of epochs: ", num_epochs)
        print("\tSample size: ", X_length)
        # print("\tResNet输出的向量的维度: ", resnet_output_size)
        print("\tshuffle_train_set: ", shuffle_train_set)
        print("\ttrain_ratio (prop of train+val): ", train_ratio)
        print("\tvalidation_split: ", validation_split)
        print('=' * 48 + '\n')

        for fold in range(k):

            train_loss_batchwise_allfold = []
            val_loss_batchwise_allfold = []
            test_loss_batchwise_allfold = []

            print('=' * 60)
            print('Fold {}'.format(fold + 1))
            print('=' * 60)

            PATH_dl_train = f'train_data_save/WorkID{work_id}_RC_{model_name}_Train_Data_seed{seed}_fold{fold}_X_length{X_length}_batchsize{batch_size}_num_epoch{20}.pt'
            train_loader = torch.load(PATH_dl_train)

            # Load dl_val for this fold
            PATH_dl_val = f'train_data_save/WorkID{work_id}_RC_{model_name}_Val_Data_seed{seed}_fold{fold}_X_length{X_length}_batchsize{batch_size}_num_epoch{20}.pt'
            val_loader = torch.load(PATH_dl_val)

            # Load dl_test for this fold
            PATH_dl_test = f'train_data_save/WorkID{work_id}_RC_{model_name}_Test_Data_seed{seed}_fold{fold}_X_length{X_length}_batchsize{batch_size}_num_epoch{20}.pt'
            test_loader = torch.load(PATH_dl_test)

            # network参数
            covit = CoViT_Normal(device=device, rank_LoRa=rank_LoRa).to(device)
            print("parameter count for CoViT: ")
            print_trainable_parameters(covit)
            warmup_PATH = f'model_save/WorkID{work_id}_R{rank_LoRa}_{model_name}_seed{seed}_fold{fold}_X_length{X_length}_batchsize{batch_size}_num_epoch{num_epochs}.pth'
            covit.load_state_dict(torch.load(warmup_PATH))
            gpu_ids = ['cuda:0', 'cuda:6', 'cuda:7']
            covit = nn.DataParallel(covit, device_ids=gpu_ids, output_device=gpu_ids[0])


            gamma12, gamma34, gamma3412, sigma1, sigma2, gamma = calculate_sigma_hat(train_loader, covit, device)

            # 输出 Gamma and std_ei 供检查
            print("CopulaPara: ")
            print("gamma12: ", gamma12)
            print("gamma34: ", gamma34)
            print("gamma3412:", gamma3412)
            print("sigma1:", sigma1)
            print("sigma2:", sigma2)
            print()

            # Criterion
            # criterion_classification = nn.BCEWithLogitsLoss()
            criterion_classification = nn.BCELoss()
            criterion_regression = nn.MSELoss()
            criterion_copula = parametricLoss()

            # Observe that all parameters are being optimized
            optimizer = optim.AdamW(covit.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                                    weight_decay=0.01, amsgrad=False)

            # Decay LR by a factor of ~ every ~ epochs
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_step_size, gamma=lr_decay_prop)

            copula_test, train_loss_batchwise, val_loss_batchwise, test_loss_batchwise = train_model_stage23(covit,
                                                                                                             criterion_copula,
                                                                                                             criterion_regression,
                                                                                                             criterion_classification,
                                                                                                             optimizer,
                                                                                                             exp_lr_scheduler,
                                                                                                             train_loader,
                                                                                                             val_loader,
                                                                                                             test_loader,
                                                                                                             num_epochs,
                                                                                                             gamma12,
                                                                                                             gamma34,
                                                                                                             gamma3412,
                                                                                                             sigma1,
                                                                                                             sigma2,
                                                                                                             gamma,
                                                                                                             device,
                                                                                                             since,
                                                                                                             k, fold,
                                                                                                             work_id,
                                                                                                             seed,
                                                                                                             X_length,
                                                                                                             batch_size,
                                                                                                             exp_lr_scheduler,
                                                                                                             learning_rate)

            # Now oucopula is the best parameter setting acquired in this fold
            # Record batchwise for all folds to plot the loss plot for whole training process
            train_loss_batchwise_allfold = [*train_loss_batchwise_allfold, *train_loss_batchwise]
            val_loss_batchwise_allfold = [*val_loss_batchwise_allfold, *val_loss_batchwise]
            test_loss_batchwise_allfold = [*test_loss_batchwise_allfold, *test_loss_batchwise]

            # Plot train val test loss
            print('-' * 100)
            print("Train, validation, & test loss per 20 epoch")
            plot_loss_stage23(batch_num_per_epoch, train_loss_batchwise_allfold, val_loss_batchwise_allfold,
                              test_loss_batchwise_allfold, work_id, seed, fold, X_length, batch_size, num_epochs)

            # ====== 记录模型 ======
            # For 本地
            copula_PATH = f'model_save/WorkID{work_id}_COPULA_R{rank_LoRa}_seed{seed}_fold{fold}_X_length{X_length}_batchsize{batch_size}_num_epoch{num_epochs}_lr{learning_rate}.pth'

            # Save the best parameter setting in warm_up_PATH
            torch.save(copula_test.module.state_dict(), copula_PATH)  # 模型提取 state_dict状态字典 并保存

            # ====== 测试模型 ======
            oucopula_test = CoViT(device).to(device)  # 载入模型之前先初始化
            oucopula_test.load_state_dict(torch.load(copula_PATH))  # 载入模型：先从PATH载入状态字典，再load_state_dict

            # Test the best model in this fold
            loss_summary = test_stage_stage23(oucopula_test, criterion_copula, criterion_regression,
                                              criterion_classification,
                                              test_loader, gamma12, gamma34, gamma3412, sigma1, sigma2, gamma, device,
                                              work_id,
                                              seed, X_length, batch_size, num_epochs, learning_rate, )

            # Record the number into the dict
            for key, val in loss_summary.items():
                if torch.is_tensor(val):
                    val = val.item()
                history[key].append(val)

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

        history['mse_left_AL'] = history["avg_loss_left_AL"]
        history['mse_right_AL'] = history["avg_loss_right_AL"]

        # 输出到csv上的结果，我们只基于test dataset
        pd.DataFrame(history).to_csv(
            f'./Simu/WorkID{work_id}_R{rank_LoRa}_COPULA_seed{seed}_X_length{X_length}_batchsize{batch_size}_num_epoch{num_epochs}_lr{learning_rate}.csv',
            index=True)

