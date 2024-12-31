from __future__ import print_function, division

import numpy as np
from numpy import *

import torch

import matplotlib.pyplot as plt
import time

import copy

from torchmetrics.classification import AUROC
from sklearn import metrics

from utils.common_utils import print_training_progress
from tqdm import tqdm


def get_test_loss_stage23(net, criterion_copula, criterion_regression, criterion_classification, data_loader, gamma12, gamma34,
                          gamma3412, sigma1, sigma2, gamma, device):
    """Object: Monitor during training procedure, use traditional loss
       Input: net,
              criterion_classification,
              criterion_regression,
              data_loader: 注意输入数据的结构顺序不能变
       Output: loss_summary: a dictionary:
                    "avg_loss_left_classification"
                    "avg_loss_right_classification"
                    "avg_loss_left_regression"
                    "avg_loss_right_regression"
                    "avg_loss_left"
                    "avg_loss_right"
                    "avg_loss_classification"
                    "avg_loss_regression"
                    "avg_loss"
    """
    testing_loss_left_high = []  # by bce
    testing_loss_right_high = []
    testing_loss_left_AL = []  # by mse
    testing_loss_right_AL = []

    COPULA_testing_loss = []

    net.eval()

    with torch.no_grad():
        for data in data_loader:
            # Load data from loader (below 6 lines)
            X_left, X_right, age, gender, *labels = data
            inputs = X_left, X_right, age, gender

            # get the data to GPU (if available)
            inputs = [_data.to(device) for _data in inputs]
            labels = [_label.to(device) for _label in labels]

            # Separate labels
            labels_left_AL = labels[2].squeeze().to(device).unsqueeze(0)
            labels_right_AL = labels[4].squeeze().to(device).unsqueeze(0)
            labels_left_high = labels[0].squeeze().to(device).unsqueeze(0)
            labels_right_high = labels[1].squeeze().to(device).unsqueeze(0)

            # Get and separate outputs
            outputs_left, outputs_right = net(inputs)
            outputs_left_high = outputs_left[:, 0].unsqueeze(0)
            outputs_right_high = outputs_right[:, 0].unsqueeze(0)
            outputs_left_AL = outputs_left[:, 1].unsqueeze(0)
            outputs_right_AL = outputs_right[:, 1].unsqueeze(0)

            # Cal loss
            loss_left_high = criterion_classification(outputs_left_high, labels_left_high)  # 不用reduce="none"，直接计算mean
            loss_right_high = criterion_classification(outputs_right_high, labels_right_high)
            loss_left_AL = criterion_regression(outputs_left_AL, labels_left_AL)
            loss_right_AL = criterion_regression(outputs_right_AL, labels_right_AL)

            y_hat = torch.cat((outputs_left_high, outputs_left_AL, outputs_right_high, outputs_right_AL), dim=0)

            y = torch.cat((labels_left_high, labels_left_AL, labels_right_high, labels_right_AL), dim=0)

            loss = criterion_copula(y_hat, y, gamma12, gamma34, gamma3412, sigma1, sigma2)
            COPULA_testing_loss.append(loss.item())

            # Record loss elements
            testing_loss_left_high.append(loss_left_high)
            testing_loss_right_high.append(loss_right_high)
            testing_loss_left_AL.append(loss_left_AL)
            testing_loss_right_AL.append(loss_right_AL)

    # Cal the avg loss for each loss element
    avg_COPULA_loss = sum(COPULA_testing_loss) / len(COPULA_testing_loss)

    avg_loss_left_high = torch.mean(torch.stack(testing_loss_left_high))  # 这个没错，batch内部已经是平均值了，现在只要list的均值
    avg_loss_right_high = torch.mean(torch.stack(testing_loss_right_high))  # 同上
    avg_loss_left_AL = torch.mean(torch.stack(testing_loss_left_AL))  # 这个没错，batch内部已经是平均值了，现在只要list的均值
    avg_loss_right_AL = torch.mean(torch.stack(testing_loss_right_AL))  # 同上

    # Agg loss by row, col and total
    avg_loss_left = avg_loss_left_high + avg_loss_left_AL
    avg_loss_right = avg_loss_right_high + avg_loss_right_AL
    avg_loss_high = avg_loss_left_high + avg_loss_right_high
    avg_loss_AL = avg_loss_left_AL + avg_loss_right_AL
    avg_loss = avg_loss_high + avg_loss_AL  # avg loss per item

    # Arrange the output
    loss_summary = {"avg_COPULA_loss": avg_COPULA_loss,
                    "avg_loss_left_high": avg_loss_left_high,
                    "avg_loss_right_high": avg_loss_right_high,
                    "avg_loss_left_AL": avg_loss_left_AL,
                    "avg_loss_right_AL": avg_loss_right_AL,
                    "avg_loss_left": avg_loss_left,
                    "avg_loss_right": avg_loss_right,
                    "avg_loss_high": avg_loss_high,
                    "avg_loss_AL": avg_loss_AL,
                    "avg_loss": avg_loss}
    return loss_summary


def AUC_plot_satge23(y_test, y_score):
    """
    compute ROC curve and ROC area for each class in each fold
    """
    N_classes = 2
    fpr = dict()
    tpr = dict()
    local_roc_auc = dict()
    for i in range(N_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(np.array(y_test[:, i]), np.array(y_score[:, i]))
        local_roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())
    local_roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(N_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(N_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= N_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr

    return local_roc_auc


# test_stage
def test_stage_stage23(net, criterion_copula, criterion_regression, criterion_classification, test_loader, gamma12, gamma34,
                       gamma3412, sigma1, sigma2, gamma, device, work_id, seed, X_length, batch_size, num_epochs, learning_rate,
                       ):
    # For AUC
    correct_classification_left = 0
    correct_classification_right = 0
    total_classification_times = 0

    # For MSE and BCE
    testing_loss_left_high = []  # by bce
    testing_loss_right_high = []
    testing_loss_left_AL = []  # by mse
    testing_loss_right_AL = []

    # For ROC
    y_test_left_high = []
    y_test_right_high = []
    y_score_left_high = []
    y_score_right_high = []

    COPULA_testing_loss = []

    net.eval()

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in test_loader:
            # Load data from loader (below 4 lines)
            X_left, X_right, age, gender, *labels = data
            inputs = X_left, X_right, age, gender

            # get the data to GPU (if available)
            inputs = [_data.to(device) for _data in inputs]
            labels = [_label.to(device) for _label in labels]

            # Separate labels
            labels_left_AL = labels[2].squeeze().to(device).unsqueeze(0)
            labels_right_AL = labels[4].squeeze().to(device).unsqueeze(0)
            labels_left_high = labels[0].squeeze().to(device).unsqueeze(0)
            labels_right_high = labels[1].squeeze().to(device).unsqueeze(0)

            # Get and separate outputs
            outputs_left, outputs_right = net(inputs)
            outputs_left_high = outputs_left[:, 0].unsqueeze(0)
            outputs_right_high = outputs_right[:, 0].unsqueeze(0)
            outputs_left_AL = outputs_left[:, 1].unsqueeze(0)
            outputs_right_AL = outputs_right[:, 1].unsqueeze(0)

            y_score_left_high.append(outputs_left_high.cpu().detach().numpy())
            y_score_right_high.append(outputs_right_high.cpu().detach().numpy())
            y_test_left_high.append(labels_left_high.cpu().detach().numpy())
            y_test_right_high.append(labels_right_high.cpu().detach().numpy())

            # Cal loss
            y_hat = torch.cat((outputs_left_high, outputs_left_AL, outputs_right_high, outputs_right_AL), dim=0)

            y = torch.cat((labels_left_high, labels_left_AL, labels_right_high, labels_right_AL), dim=0)

            loss_copula = criterion_copula(y_hat, y, gamma12, gamma34, gamma3412, sigma1, sigma2)
            # add the loss of this batch to the list
            COPULA_testing_loss.append(loss_copula.item())
            loss_left_high = criterion_classification(outputs_left_high, labels_left_high)  # 不用reduce="none"，直接计算mean
            loss_right_high = criterion_classification(outputs_right_high, labels_right_high)
            loss_left_AL = criterion_regression(outputs_left_AL, labels_left_AL)
            loss_right_AL = criterion_regression(outputs_right_AL, labels_right_AL)

            # Record loss elements
            testing_loss_left_high.append(loss_left_high)
            testing_loss_right_high.append(loss_right_high)
            testing_loss_left_AL.append(loss_left_AL)
            testing_loss_right_AL.append(loss_right_AL)

            # Cal TP FP and so on
            predicted_left_classification = outputs_left_high > 0.5
            predicted_right_classification = outputs_right_high > 0.5
            total_classification_times += torch.numel(outputs_left_high)  # torch中item的个数
            # 注意，这里累加的是每次分类输出的tensor的元素的个数，也就是每个batch里分类的的次数，最后就能得到总的分类次数

            # 计算正确分类的次数
            # ==判断两个tensor各个位置是否相等，sum把true加起来，item取出其中的tensor
            correct_classification_left += (predicted_left_classification == labels_left_high).sum().item()
            correct_classification_right += (predicted_right_classification == labels_right_high).sum().item()

    """
    compute ROC curve and ROC area for each class in each fold
    """
    y_test_left_high = np.array([t.ravel() for t in y_test_left_high])
    y_test_right_high = np.array([t.ravel() for t in y_test_right_high])
    y_score_left_high = np.array([t.ravel() for t in y_score_left_high])
    y_score_right_high = np.array([t.ravel() for t in y_score_right_high])

    # Cal AUC and ROC
    print('-' * 100)
    print("ROC plot & AUC under test set")
    AUC_results_left = AUC_plot_satge23(y_test_left_high, y_score_left_high)
    AUC_results_right = AUC_plot_satge23(y_test_right_high, y_score_right_high)

    ###############################
    y_test_left_high2 = torch.flatten(torch.tensor(y_test_left_high)).to(torch.int64)
    y_score_left_high2 = torch.flatten(torch.tensor(y_score_left_high))
    y_test_right_high2 = torch.flatten(torch.tensor(y_test_right_high)).to(torch.int64)
    y_score_right_high2 = torch.flatten(torch.tensor(y_score_right_high))

    auroc = AUROC(task="binary")
    AUC_left_doubleCheck = auroc(y_score_left_high2, y_test_left_high2)
    print("AUC doubleCheck left: ", AUC_left_doubleCheck.detach().cpu().numpy())

    auroc = AUROC(task="binary")
    AUC_right_doubleCheck = auroc(y_score_right_high2, y_test_right_high2)
    print("AUC doubleCheck right: ", AUC_right_doubleCheck.detach().cpu().numpy())
    ###############################

    # Cal classification accuracy
    classification_accuracy_left = correct_classification_left / total_classification_times
    classification_accuracy_right = correct_classification_right / total_classification_times

    # Cal the avg loss for each loss element
    avg_COPULA_loss = sum(COPULA_testing_loss) / len(COPULA_testing_loss)

    avg_loss_left_high = torch.mean(torch.stack(testing_loss_left_high))  # 这个没错，batch内部已经是平均值了，现在只要list的均值
    avg_loss_right_high = torch.mean(torch.stack(testing_loss_right_high))  # 同上
    avg_loss_left_AL = torch.mean(torch.stack(testing_loss_left_AL))  # 这个没错，batch内部已经是平均值了，现在只要list的均值
    avg_loss_right_AL = torch.mean(torch.stack(testing_loss_right_AL))  # 同上

    # Agg loss by row, col and total
    avg_loss_left = avg_loss_left_high + avg_loss_left_AL
    avg_loss_right = avg_loss_right_high + avg_loss_right_AL
    avg_loss_high = avg_loss_left_high + avg_loss_right_high
    avg_loss_AL = avg_loss_left_AL + avg_loss_right_AL
    avg_loss = avg_loss_high + avg_loss_AL  # avg loss per item

    # Arrange the output
    loss_summary = {"avg_COPULA_loss": avg_COPULA_loss,
                    "avg_loss_left_high": avg_loss_left_high,
                    "avg_loss_right_high": avg_loss_right_high,
                    "avg_loss_left_AL": avg_loss_left_AL,
                    "avg_loss_right_AL": avg_loss_right_AL,
                    "avg_loss_left": avg_loss_left,
                    "avg_loss_right": avg_loss_right,
                    "avg_loss_high": avg_loss_high,
                    "avg_loss_AL": avg_loss_AL,
                    "avg_loss": avg_loss,
                    "classification_accuracy_left": classification_accuracy_left,
                    "classification_accuracy_right": classification_accuracy_right,
                    "AUC_left": AUC_left_doubleCheck,
                    "AUC_right": AUC_right_doubleCheck}

    print('-' * 100)
    print("Performance of the BEST parameter setting on TEST set:")
    for key, val in loss_summary.items():
        if torch.is_tensor(val):
            val = val.item()
        print("\t {}: {}".format(key, val))
    with open(
            f'WorkID{work_id}_ds_XL_Data_seed{seed}_X_length{X_length}_lr{learning_rate}_batchsize{batch_size}_num_epoch{num_epochs}.txt',
            'a') as f:
        print('-' * 100, file=f)
        print("Performance of the BEST parameter setting on TEST set:", file=f)
        for key, val in loss_summary.items():
            if torch.is_tensor(val):
                val = val.item()
            print("\t {}: {}".format(key, val), file=f)

    return loss_summary


# Train model
def train_model_stage23(model, criterion_copula, criterion_regression, criterion_classification, optimizer, scheduler, dl_train,
                        dl_val, dl_test,
                        num_epochs, gamma12, gamma34, gamma3412, sigma1, sigma2, gamma, device, since, k, fold,
                        work_id, seed, X_length, batch_size, exp_lr_scheduler, learning_rate):
    """Object:
       Input: model: 就是net，你构造的需要训练的网络
              criterion_classification:
              criterion_regression:
              optimizer: 实例化的一个优化器
              scheduler:
              dl_train:
              dl_val:
              num_epochs:
    """

    best_model_wts = copy.deepcopy(model.state_dict())  # 存储最佳参数
    least_loss = 10000000.0  # 存储最小损失
    best_epoch_num = 0  # 存储? # TODO
    val_loss = []  # return of get_test_loss per 20 batch
    n_train = int(len(dl_train))  # loader的长度，注意不是图片的张数，要翻倍

    # For plot
    train_loss_batchwise = []
    val_loss_batchwise = []
    test_loss_batchwise = []

    # Loop epoch
    for epoch in range(num_epochs):  # [0, num_epochs-1]
        # Report learning rate change if happened
        if epoch % exp_lr_scheduler.state_dict()['step_size'] == 0:
            print(">" * 30 + " Learning rate is to {} now.".format(
                exp_lr_scheduler.state_dict()['_last_lr'][0]) + "<" * 30)
        print('-' * 100)
        print(f'Epoch {epoch + 1}/{num_epochs}')
        with open(
                f'WorkID{work_id}_ds_XL_Data_seed{seed}_X_length{X_length}_lr{learning_rate}_batchsize{batch_size}_num_epoch{num_epochs}.txt',
                'a') as f:
            # Report learning rate change if happened
            if epoch % exp_lr_scheduler.state_dict()['step_size'] == 0:
                print(">" * 30 + " Learning rate is to {} now.".format(
                    exp_lr_scheduler.state_dict()['_last_lr'][0]) + "<" * 30, file=f)
            print('-' * 100, file=f)
            print(f'Epoch {epoch + 1}/{num_epochs}', file=f)
        model.train()

        running_loss = 0.0
        running_loss_temp = 0.0
        i = 0
        # Loop train data in one epoch
        for data in tqdm(dl_train):  # i 从0开始
            # Load data from loader (below 4 lines)
            X_left, X_right, age, gender, *labels = data
            inputs = X_left, X_right, age, gender

            # get the data to GPU (if available)
            inputs = [_data.to(device) for _data in inputs]
            labels = [_label.to(device) for _label in labels]

            # Separate labels
            labels_left_AL = labels[2].squeeze().to(device).unsqueeze(0)
            labels_right_AL = labels[4].squeeze().to(device).unsqueeze(0)
            labels_left_high = labels[0].squeeze().to(device).unsqueeze(0)
            labels_right_high = labels[1].squeeze().to(device).unsqueeze(0)

            # 训练过程
            # 反向传播之前初始化梯度就行，这里我们放在最前面
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                # 前向传播
                outputs_left, outputs_right = model(inputs)
                outputs_left_high = outputs_left[:, 0].unsqueeze(0)
                outputs_right_high = outputs_right[:, 0].unsqueeze(0)
                outputs_left_AL = outputs_left[:, 1].unsqueeze(0)
                outputs_right_AL = outputs_right[:, 1].unsqueeze(0)

                # # Cal loss elementwise

                y_hat = torch.cat((outputs_left_high, outputs_left_AL, outputs_right_high, outputs_right_AL), dim=0)

                y = torch.cat((labels_left_high, labels_left_AL, labels_right_high, labels_right_AL), dim=0)

                loss = criterion_copula(y_hat, y, gamma12, gamma34, gamma3412, sigma1, sigma2)

                # 反向传播
                loss.backward()
                # 更新
                optimizer.step()

            running_loss += loss.item()  # 记录每个epoch的平均loss
            running_loss_temp += loss.item()  # 记录每20个batch的累计loss

            if i % 100 == 99:
                # record train/val/test loss per 20 batch for plot
                # train_loss_summary = get_test_loss(model, criterion_classification, criterion_regression, dl_train)
                val_loss_summary = get_test_loss_stage23(model, criterion_copula, criterion_regression,
                                                         criterion_classification, dl_val, gamma12, gamma34, gamma3412, sigma1,
                                                         sigma2, gamma, device)
                # test_loss_summary = get_test_loss(model, criterion_classification, criterion_regression, dl_test)

                # 下方使用item，把cuda里的tensor的数值部分提取到cpu里，这种方法比什么tensor.to("cpu")来得直接
                avg_val_loss_l_r_cla_reg_sum = [val_loss_summary["avg_COPULA_loss"],
                                                val_loss_summary["avg_loss_left"],
                                                val_loss_summary["avg_loss_right"],
                                                val_loss_summary["avg_loss_high"],
                                                val_loss_summary["avg_loss_AL"],
                                                val_loss_summary["avg_loss"]]

                avg_loss_str = ' '.join(f'{value:.3f}' for value in avg_val_loss_l_r_cla_reg_sum)
                # Avg train loss past 20 batch: the mean of train losses of model during past 20 batch
                # Val LRClaRegTot loss this batch: the Left, Right, Classification, Regression, and Total loss of model with parameter attained in this batch on val set
                print(
                    f'[Num of batch: {i + 1:2d}] Avg train loss past 100 batch: {running_loss_temp / 100 :.3f} | Val LRClaRegTot loss this batch: {avg_loss_str} ')  # Epoch: {epoch + 1},
                with open(
                        f'WorkID{work_id}_ds_XL_Data_seed{seed}_X_length{X_length}_lr{learning_rate}_batchsize{batch_size}_num_epoch{num_epochs}.txt',
                        'a') as f:
                    print(
                        f'[Num of batch: {i + 1:2d}] Avg train loss past 20 batch: {running_loss_temp / 20 :.3f} | Val LRClaRegTot loss this batch: {avg_loss_str} ',
                        file=f)  # Epoch: {epoch + 1},
                # print(f'\t Train loss this batch: {train_loss_batchwise :.3f}') # <<<<<< 和 Val tot loss this batch 相对于的
                running_loss_temp = 0.0
            i += 1

        # 每个epoch结束时计算三个loss，输出，并在之后用来画loss plot
        # 下方使用item，把cuda里的tensor的数值部分提取到cpu里，这种方法比什么tensor.to("cpu")来得直接
        train_loss_summary = get_test_loss_stage23(model, criterion_copula, criterion_regression, criterion_classification,
                                                   dl_train, gamma12, gamma34, gamma3412, sigma1, sigma2, gamma, device)
        val_loss_summary = get_test_loss_stage23(model, criterion_copula, criterion_regression, criterion_classification,
                                                 dl_val, gamma12, gamma34, gamma3412, sigma1, sigma2, gamma, device)
        test_loss_summary = get_test_loss_stage23(model, criterion_copula, criterion_regression, criterion_classification,
                                                  dl_test, gamma12, gamma34, gamma3412, sigma1, sigma2, gamma, device)

        train_loss_batchwise.append(train_loss_summary["avg_loss"].item())
        # avg_val_loss_cla_reg_sum = get_test_loss(model, criterion_classification, criterion_regression, dl_val)
        val_loss_batchwise.append(val_loss_summary["avg_loss"].item())
        test_loss_batchwise.append(test_loss_summary["avg_loss"].item())

        print("-" * 20)
        print("Train copula loss this batch: ", train_loss_summary["avg_COPULA_loss"].item())
        print("Val copula loss this batch:   ", val_loss_summary["avg_COPULA_loss"].item())
        print("Test copula loss this batch:  ", test_loss_summary["avg_COPULA_loss"].item())
        print("Train loss this batch: ", train_loss_summary["avg_loss"].item())
        print("Val loss this batch:   ", val_loss_summary["avg_loss"].item())
        print("Test loss this batch:  ", test_loss_summary["avg_loss"].item())
        print("-" * 20)

        with open(
                f'WorkID{work_id}_ds_XL_Data_seed{seed}_X_length{X_length}_lr{learning_rate}_batchsize{batch_size}_num_epoch{num_epochs}.txt',
                'a') as f:
            print("-" * 20, file=f)
            print("Train copula loss this batch: ", train_loss_summary["avg_COPULA_loss"].item(), file=f)
            print("Val copula loss this batch:   ", val_loss_summary["avg_COPULA_loss"].item(), file=f)
            print("Test copula loss this batch:  ", test_loss_summary["avg_COPULA_loss"].item(), file=f)

            print("Train MSE loss this batch: ", train_loss_summary["avg_loss"].item(), file=f)
            print("Val MSE loss this batch:   ", val_loss_summary["avg_loss"].item(), file=f)
            print("Test MSE loss this batch:  ", test_loss_summary["avg_loss"].item(), file=f)

            print("-" * 20, file=f)

        scheduler.step()  # 放在epoch循环里

        # Cal and record best model over val set over epoches
        # 这边只能使用val_dataset，不能使用test set，test是用来检验此处挑选出的模型的
        avg_val_loss_this_epoch = \
        get_test_loss_stage23(model, criterion_copula, criterion_regression, criterion_classification, dl_val, gamma12, gamma34,
                              gamma3412, sigma1, sigma2, gamma, device)["avg_loss"]

        if avg_val_loss_this_epoch < least_loss:  # 使用val loss选择最优模型
            least_loss = avg_val_loss_this_epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch_num = epoch

        # place to print time summary
        print_training_progress(epoch, num_epochs, time.time() - since, k, fold)
        print()

        # ====== This is the end of code within a epoch ======#

    # Load the BEST model when the training process for all epoches is ended
    model.load_state_dict(best_model_wts)

    avg_train_loss_best_model = \
    get_test_loss_stage23(model, criterion_copula, criterion_regression, criterion_classification, dl_train, gamma12, gamma34,
                          gamma3412, sigma1, sigma2, gamma, device)["avg_loss"]
    avg_val_loss_best_model = \
    get_test_loss_stage23(model, criterion_copula, criterion_regression, criterion_classification, dl_val, gamma12, gamma34,
                          gamma3412, sigma1, sigma2, gamma, device)["avg_loss"]
    avg_test_loss_best_model = \
    get_test_loss_stage23(model, criterion_copula, criterion_regression, criterion_classification, dl_test, gamma12, gamma34,
                          gamma3412, sigma1, sigma2, gamma, device)["avg_loss"]

    # Evaluate the BEST model
    print('-' * 100)
    print("Evaluate the best parameter setting acquired in this fold: ")
    print("\t Best setting acquired in Epoch: {}/{}".format(best_epoch_num + 1, num_epochs))
    print("\t Avg loss on train set: {:.3f}".format(avg_train_loss_best_model))
    print("\t Avg loss on validation set: {:.3f}".format(avg_val_loss_best_model))
    print("\t Avg loss on test set: {:.3f}".format(avg_test_loss_best_model))
    with open(
            f'WorkID{work_id}_ds_XL_Data_seed{seed}_X_length{X_length}_lr{learning_rate}_batchsize{batch_size}_num_epoch{num_epochs}.txt',
            'a') as f:
        print('-' * 100, file=f)
        print("Evaluate the best parameter setting acquired in this fold: ", file=f)
        print("\t Best setting acquired in Epoch: {}/{}".format(best_epoch_num + 1, num_epochs), file=f)
        print("\t Avg loss on train set: {:.3f}".format(avg_train_loss_best_model), file=f)
        print("\t Avg loss on validation set: {:.3f}".format(avg_val_loss_best_model), file=f)
        print("\t Avg loss on test set: {:.3f}".format(avg_test_loss_best_model), file=f)

    return model, train_loss_batchwise, val_loss_batchwise, test_loss_batchwise


def plot_loss_stage23(batch_num_per_epoch, train_loss, val_loss, test_loss, work_id, seed, fold, X_length, batch_size, num_epochs):
    plt.figure(figsize=(50, 30))
    x = range(1, len(train_loss) + 1)
    xticks = range(1, len(train_loss) + 1, 2)

    plt.plot(x, train_loss, label='Train Loss')
    plt.plot(x, val_loss, label='Validation Loss')
    plt.plot(x, test_loss, label='Test Loss')

    plt.xticks(xticks, fontsize=12)
    plt.xlabel('Per 20 Epochs')
    plt.ylabel('Loss')
    plt.title('Itemwise Loss Per 20 Batch')
    plt.legend()
    plt.savefig(
        f'./Simu/WorkID{work_id}_RC_copula_LoRaViT_LossPlot_seed{seed}_fold{fold}_X_length{X_length}_batchsize{batch_size}_num_epoch{num_epochs}.png')
    plt.show()