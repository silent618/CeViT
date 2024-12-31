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
from utils.train_simulation_stage1_utils import train_model, plot_loss, test_stage
from dataloader.dataloader import get_datasets
from modeling.module1 import MultiTaskMLP

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # #  Important Parameters # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
seed = 0
rank_LoRa = 0  # can set to 8, 16 for other experiments

work_id = 122501
seed_list = [seed]
model_name = 'CoViT'

k = 10  # <<<<<< KFOLD <<<<<<
batch_size = 32  # <<<<<< BATCH SIZE <<<<<<
num_epochs = 12  # 150 # <<<<<< EPOCH <<<<<<

validation_split = 0.111
shuffle_train_set = True
train_ratio = 0.8

X_length = 15000
learning_rate = 0.001
lr_decay_step_size = 2
lr_decay_prop = 0.9

cudnn.benchmark = True
plt.ion()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 创建保存路径
simu_warmup_fold = "./Simu/"
copula_fold = "./Copula/"

os.makedirs(simu_warmup_fold, exist_ok=True)  # 存在路径则不创建也不报错
os.makedirs(copula_fold, exist_ok=True)  # 存在路径则不创建也不报错

set_seed(seed)


class ViTOutput(nn.Module):
    def __init__(self, in_features, out_features, dropout_prob) -> None:
        super().__init__()
        self.dense = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class CoViT(nn.Module):
    def __init__(self, device='cuda:0'):
        super(CoViT, self).__init__()
        self.model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
        )

        self.model.classifier = nn.Identity()

        config = LoraConfig(
            r=rank_LoRa,
            lora_alpha=8,
            target_modules=["query", "value"],
            lora_dropout=0.1,
            bias="none",
            modules_to_save=["classifier"],
        )
        self.model = LoraModel(self.model, config, "LoRa")
        print("parameter count for model: ")
        print_trainable_parameters(self.model)

        self.fc_classification = nn.Linear(768, 1)
        self.fc_regression = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_left = x[0]
        x_right = x[1]

        hidden_states_left = self.model.model.vit.embeddings(x_left)
        hidden_states_right = self.model.model.vit.embeddings(x_right)

        for j, block in enumerate(self.model.model.vit.encoder.layer):
            """处理左眼图像"""
            hidden_states_ln_left = block.layernorm_before(hidden_states_left)
            self_attention_outputs_left = block.attention(hidden_states_ln_left, head_mask=None,
                                                          output_attentions=False)

            attention_output_left = self_attention_outputs_left[0]
            outputs_left = self_attention_outputs_left[1:]

            # Residual connection and LayerNorm after the self-attention block
            hidden_states_left = attention_output_left + hidden_states_left

            # in ViT, layernorm is also applied after self-attention
            layer_output_left = block.layernorm_after(hidden_states_left)

            layer_output_left = block.intermediate(layer_output_left)

            # second residual connection is done here
            layer_output_left = block.output(layer_output_left, hidden_states_left)

            # Residual connection
            hidden_states_left = (layer_output_left,) + outputs_left

            hidden_states_left = hidden_states_left[0]
            """处理右眼图像"""
            hidden_states_ln_right = block.layernorm_before(hidden_states_right)
            self_attention_outputs_right = block.attention(hidden_states_ln_right, head_mask=None,
                                                           output_attentions=False)

            attention_output_right = self_attention_outputs_right[0]
            outputs_right = self_attention_outputs_right[1:]

            # Residual connection and LayerNorm after the self-attention block
            hidden_states_right = attention_output_right + hidden_states_right

            # in ViT, layernorm is also applied after self-attention
            layer_output_right = block.layernorm_after(hidden_states_right)

            layer_output_right = block.intermediate(layer_output_right)

            # second residual connection is done here
            layer_output_right = block.output(layer_output_right, hidden_states_right)

            # Residual connection
            hidden_states_right = (layer_output_right,) + outputs_right

            hidden_states_right = hidden_states_right[0]

        # Final LayerNorm
        hidden_states_left = self.model.model.vit.layernorm(hidden_states_left)
        hidden_states_right = self.model.model.vit.layernorm(hidden_states_right)

        logits_left = self.model.model.classifier(hidden_states_left[:, 0, :])
        # Calculate binary classification output
        out_left = self.fc_classification(logits_left)
        y_classification_left = self.sigmoid(out_left)
        # Calculate regression output
        y_regression_left = self.fc_regression(logits_left)

        logits_right = self.model.model.classifier(hidden_states_right[:, 0, :])
        # Calculate binary classification output
        out_right = self.fc_classification(logits_right)
        y_classification_right = self.sigmoid(out_right)
        # Calculate regression output
        y_regression_right = self.fc_regression(logits_right)

        y_vec = torch.cat((y_classification_left, y_regression_left, y_classification_right, y_regression_right), -1)

        return y_vec[:, 0:2], y_vec[:, 2:]


if __name__ == "__main__":
    ds_XL = get_datasets('simulation_1.csv', 'D:/Code/OUCopula')
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

    for fold, (train_idx, test_idx) in enumerate(kfold.split(np.arange(len(ds_XL)))):
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

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2,
                                  pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        # 存储划分的数据集以便copula直接调用，从源头防止数据泄露！ @2023-09-27
        # warmup和copula我的batch size设置的也必须是一样的
        data = []
        for i in range(len(train_dataset)):
            sample = train_dataset[i]
            data.append([sample['left_image_path'], sample['right_image_path'], sample['Y_left_myopia'],
                         sample["Y_right_myopia"], sample["Y_left_al"], sample["Y_right_al"]])

        df = pd.DataFrame(data,
                          columns=['left_image_path', 'right_image_path',
                                   'Y_left_myopia', "Y_right_myopia",
                                   "Y_left_al", "Y_right_al"
                                   ])
        df.to_csv(f'train_data_save/WorkID{work_id}_RC_{model_name}_Train_Data_seed{seed}_fold{fold}_X_length{X_length}_batchsize{batch_size}_num_epoch{num_epochs}.csv', index=False)
        
        data = []
        for i in range(len(val_dataset)):
            sample = val_dataset[i]
            data.append([sample['left_image_path'], sample['right_image_path'], sample['Y_left_myopia'],
                         sample["Y_right_myopia"], sample["Y_left_al"], sample["Y_right_al"]])

        df = pd.DataFrame(data,
                          columns=['left_image_path', 'right_image_path',
                                   'Y_left_myopia', "Y_right_myopia",
                                   "Y_left_al", "Y_right_al"
                                   ])
        df.to_csv(
            f'train_data_save/WorkID{work_id}_RC_{model_name}_Val_Data_seed{seed}_fold{fold}_X_length{X_length}_batchsize{batch_size}_num_epoch{num_epochs}.csv',
            index=False)

        data = []
        for i in range(len(test_dataset)):
            sample = test_dataset[i]
            data.append([sample['left_image_path'], sample['right_image_path'], sample['Y_left_myopia'],
                         sample["Y_right_myopia"], sample["Y_left_al"], sample["Y_right_al"]])

        df = pd.DataFrame(data,
                          columns=['left_image_path', 'right_image_path',
                                   'Y_left_myopia', "Y_right_myopia",
                                   "Y_left_al", "Y_right_al"
                                   ])
        df.to_csv(
            f'train_data_save/WorkID{work_id}_RC_{model_name}_Test_Data_seed{seed}_fold{fold}_X_length{X_length}_batchsize{batch_size}_num_epoch{num_epochs}.csv',
            index=False)

        # network参数
        covit = CoViT(device=device).to(device)
        # covit = MultiTaskMLP().to(device)
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
        net_adapter, train_loss_batchwise, val_loss_batchwise, test_loss_batchwise = train_model(covit,
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
        torch.save(net_adapter.state_dict(), warm_up_PATH)  # 模型提取 state_dict状态字典 并保存

        # ====== 测试模型 ======
        net_test = CoViT(device).to(device)  # 载入模型之前先初始化
        net_test.load_state_dict(torch.load(warm_up_PATH))  # 载入模型：先从PATH载入状态字典，再load_state_dict

        # Test the best model in this fold
        loss_summary = test_stage(net_test, criterion_regression, criterion_classification, test_loader, device)

        # history['Classification_accuracy'].append(Classification_accuracy)
        # history['Regression_loss'].append(Regression_loss)

        # Record the number into the dict
        for key, val in loss_summary.items():
            if torch.is_tensor(val):
                val = val.item()
            history[key].append(val)

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    # 输出到csv上的结果，我们只基于test dataset
    pd.DataFrame(history).to_csv(
        f'./Simu/WorkID{work_id}_Warmup_R{rank_LoRa}_{model_name}_seed{seed}_X_length{X_length}_batchsize{batch_size}_num_epoch{num_epochs}.csv',
        index=True)

    """x1 = torch.randn(16, 3, 224, 224).to(device)
    x2 = torch.randn(16, 3, 224, 224).to(device)
    y_label = torch.rand(16).to(device)
    x = [x1, x2]
    model = CoViT(device).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,
                            weight_decay=0.01, amsgrad=False)
    output_left, output_right = model(x)
    y = output_left[:, 0]
    loss = F.binary_cross_entropy(y, y_label)
    loss.backward()
    b = 1

    ds_XL = torch.load('simulation.pt')
    train_loader = DataLoader(ds_XL, batch_size=batch_size, shuffle=True, drop_last=True)
    for i in train_loader:
        # X_left, X_right, *labels = i
        inputs = [i[0], i[1]]

        # get the data to GPU (if available)
        inputs = [_data.to(device) for _data in inputs]
        labels = [_label.to(device) for _label in i[2:]]

        # Separate labels
        labels_left_AL = labels[2].squeeze().to(device)
        labels_right_AL = labels[3].squeeze().to(device)
        labels_left_high = labels[0].squeeze().to(device)
        labels_right_high = labels[1].squeeze().to(device)
        optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(True):
            # 前向传播
            outputs_left, outputs_right = model(inputs)
            outputs_left_high = outputs_left[:, 0]
            outputs_right_high = outputs_right[:, 0]
            outputs_left_AL = outputs_left[:, 1]
            outputs_right_AL = outputs_right[:, 1]

            # Cal loss elementwise
            loss_left_high = F.binary_cross_entropy(outputs_left_high,
                                                      labels_left_high)  # 不用reduce="none"，直接计算mean
            loss_right_high = F.binary_cross_entropy(outputs_right_high, labels_right_high)
            # print(labels_right_AL)
            loss_left_AL = F.mse_loss(outputs_left_AL, labels_left_AL)
            loss_right_AL = F.mse_loss(outputs_right_AL, labels_right_AL)

            # Agg element loss by row, col and total
            loss = loss_left_high + loss_right_high + loss_left_AL + loss_right_AL

            # 反向传播
            loss_left_high.backward()
            loss_right_high.backward()
            loss_left_AL.backward()
            loss_right_AL.backward()
            # loss.backward()
            # 更新
            optimizer.step()"""
