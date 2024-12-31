import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import multivariate_normal

from torch.utils.data import TensorDataset, Dataset, DataLoader, random_split, SubsetRandomSampler, ConcatDataset, \
    Subset
import torchvision

from torch.distributions import multivariate_normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions import normal
import torchvision.transforms as tf

import os
from torch import nn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm


class my_normal(nn.Module):
    def __init__(self, mu, sigma):
        super(my_normal, self).__init__()
        self.mu = mu
        self.sigma = sigma
        self.normal_distribution = normal.Normal(self.mu, self.sigma)

    def forward(self, x):
        return self.normal_distribution.cdf(x)


class ToThreeChannels:
    def __call__(self, img):
        return img.repeat(3, 1, 1)


def save_image(tensor, output_dir, filename):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torchvision.utils.save_image(tensor, os.path.join(output_dir, filename))


width = height = 224
batch_size = 16  # <<<<<< BATCH SIZE <<<<<<
num_epochs = 120  # <<<<<< EPOCH <<<<<<

validation_split = 0.25
shuffle_train_set = True
train_ratio = 0.8

learning_rate = 0.01
lr_decay_step_size = 25
lr_decay_prop = 0.5

work_id = 100

seed = 0

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

X_length = 15000

transformImg = tf.Compose([
    tf.Resize((height, width)),
])  # Set image transformation
transform = tf.Compose([ToThreeChannels()])

# #################### Data generation #####################################
# #################### Data generation #####################################
X_left = [0.0] * X_length
X_right = [0.0] * X_length
X_left_raw = torch.empty(X_length, 1, 72, 72)
X_right_raw = torch.empty(X_length, 1, 72, 72)
Y_left_myopia = torch.empty(X_length, 1)
Y_right_myopia = torch.empty(X_length, 1)
Y_left_al = torch.empty(X_length, 1)
Y_right_al = torch.empty(X_length, 1)
Y_left = torch.empty(X_length, 2)
Y_right = torch.empty(X_length, 2)

mean = torch.tensor([0, 0], dtype=torch.float32)
cov_matrix = torch.tensor([[0.25, 0.125], [0.125, 0.25]], dtype=torch.float32)

mean_33 = torch.tensor([0.5, 0.5], dtype=torch.float32)  # A_33 和 B_33 的均值向量
cov_33_matrix = torch.tensor([[0.25, 0.125], [0.125, 0.25]], dtype=torch.float32)

mvn = torch.distributions.MultivariateNormal(mean, cov_matrix)
mvn_33 = torch.distributions.MultivariateNormal(mean_33, cov_33_matrix)


def compute_sum_blocks(A, B, use_tanh=True):
    nnTanh = torch.tanh
    S1_x = nnTanh(A[0:3, 0:3]).sum() if use_tanh else A[0:3, 0:3].sum()
    S2_x = A[3:6, 3:6].sum()
    S3_x = nnTanh(A[6:9, 6:9].sum()) if use_tanh else A[6:9, 6:9].sum()
    S1_z = nnTanh(B[0:3, 0:3]).sum() if use_tanh else B[0:3, 0:3].sum()
    S2_z = B[3:6, 3:6].sum()
    S3_z = nnTanh(B[6:9, 6:9]).sum() if use_tanh else B[6:9, 6:9].sum()
    return S1_x, S2_x, S3_x, S1_z, S2_z, S3_z


for i in range(X_length):
    A = torch.empty(72, 72)
    B = torch.empty(72, 72)

    for j in range(3):
        for k in range(3):
            if (j, k) == (2, 2):  # 特殊分布 A_33 和 B_33
                samples = mvn_33.sample(sample_shape=(576,))
            else:
                samples = mvn.sample(sample_shape=(576,))

            A[j * 24:(j + 1) * 24, k * 24:(k + 1) * 24] = samples[..., 0].reshape(24, 24)
            B[j * 24:(j + 1) * 24, k * 24:(k + 1) * 24] = samples[..., 1].reshape(24, 24)

    # 计算 y1, y2
    S1_x, S2_x, S3_x, S1_z, S2_z, S3_z = compute_sum_blocks(A, B)
    e = torch.distributions.MultivariateNormal(loc=torch.tensor([0., 0., 0., 0.]),
                                               covariance_matrix=torch.tensor([[1., 1 / 2, 1/2, 1/8], [1 / 2, 1., 1/8, 1/2], [1/2, 1/8, 1., 1 / 2], [1/8, 1/2, 1 / 2, 1.]]))
    sample_e = e.sample()
    e1 = sample_e[0]
    e2 = sample_e[1]
    e3 = sample_e[2]
    e4 = sample_e[3]

    y1 = S1_x + S2_x + S3_x + e1
    y2 = S1_z + S2_z + S3_z + e2

    # 计算 y3, y4
    _, S2_sx, _, _, S2_sz, _ = compute_sum_blocks(A, B, use_tanh=False)
    nnSigmoid = torch.sigmoid
    y3 = torch.bernoulli(nnSigmoid(S2_sx + e3))
    y4 = torch.bernoulli(nnSigmoid(S2_sz + e4))
    # save_image(A, './images/', 'test.png')
    # 填充数据
    X_left_raw[i] = A.unsqueeze(0)
    X_left[i] = transformImg(X_left_raw[i])
    X_right_raw[i] = B.unsqueeze(0)
    X_right[i] = transformImg(X_right_raw[i])
    Y_left_al[i] = torch.tensor([y1.squeeze()])
    Y_right_al[i] = torch.tensor([y2.squeeze()])
    Y_left_myopia[i] = torch.tensor([y3.squeeze()])
    Y_right_myopia[i] = torch.tensor([y4.squeeze()])
    Y_left[i] = torch.cat((Y_left_myopia[i], Y_left_al[i]), dim=0)
    Y_right[i] = torch.cat((Y_right_myopia[i], Y_right_al[i]), dim=0)

X_left = torch.stack(X_left)
X_right = torch.stack(X_right)

# 定义输出目录和CSV文件路径
dir = '/images_20/'
output_dir = '.' + dir
os.makedirs(output_dir, exist_ok=True)
csv_file = 'simulation_20.csv'

# 创建一个空的DataFrame来存储路径和标签
data = []

# 遍历所有数据
for i in tqdm(range(len(X_left))):
    # 保存左图
    left_filename = f'left_{i:05d}.png'
    save_image(X_left[i], output_dir, left_filename)

    # 保存右图
    right_filename = f'right_{i:05d}.png'
    save_image(X_right[i], output_dir, right_filename)

    # 记录路径和标签
    data.append({
        'left_image_path': os.path.join(f'{dir}', left_filename),
        'right_image_path': os.path.join(f'{dir}', right_filename),
        'Y_left_myopia': Y_left_myopia[i].item(),
        'Y_right_myopia': Y_right_myopia[i].item(),
        'Y_left_al': Y_left_al[i].item(),
        'Y_right_al': Y_right_al[i].item()
    })

# 将数据转换为DataFrame并保存为CSV文件
df = pd.DataFrame(data)
df.to_csv(csv_file, index=False)

"""ds_XL = TensorDataset(X_left.clone().detach().requires_grad_(True).float(),
                      X_right.clone().detach().requires_grad_(True).float(),
                      torch.tensor(Y_left_myopia).float(),
                      torch.tensor(Y_right_myopia).float(),
                      torch.tensor(Y_left_al).float(),
                      torch.tensor(Y_right_al).float(),

                      )

data = []
for i in range(len(ds_XL)):
    sample = ds_XL[i]
    data.append([sample[0], sample[1], sample[2], sample[3], sample[4], sample[5]])

df = pd.DataFrame(data,
                  columns=["X_left", "X_right",
                           "Y_left_myopia", "Y_right_myopia",
                           "Y_left_al", "Y_right_al"
                           ])
df.to_csv('TESTPREV1.csv', index=False)

PATH_ds = f'WorkID{work_id}_ds_XL_Data_seed{seed}_X_length{X_length}_batchsize{batch_size}_num_epoch{num_epochs}.pt'
# dir_path = os.path.dirname(PATH_ds)
# os.makedirs(dir_path, exist_ok=True)
torch.save(ds_XL, PATH_ds)"""
