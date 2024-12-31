import torch
import torch.nn as nn
from torch.distributions import normal
import numpy as np
from torch.distributions import MultivariateNormal
from mvnorm import multivariate_normal_cdf


def compute_discrete_probability(y3, y4, a1, a2, mu_tilde, V_tilde):
    device = mu_tilde.device
    mu_tilde1, mu_tilde2 = mu_tilde[0], mu_tilde[1]
    V_tilde11, V_tilde22 = V_tilde[0, 0], V_tilde[1, 1]

    Gauss_dist = normal.Normal(torch.zeros(1).cuda(), 1)
    # bivariate_normal = MultivariateNormal(loc=mu_tilde, covariance_matrix=V_tilde)
    bivariate_normal_cdf = multivariate_normal_cdf(value=torch.cat((a1, a2), dim=-1), loc=mu_tilde, covariance_matrix=V_tilde)
    P_y3_y4 = (
            (1 - 2 * y3) * (1 - 2 * y4) * bivariate_normal_cdf +
            y3 * (1 - 2 * y4) * Gauss_dist.cdf((a2 - mu_tilde2) / torch.sqrt(V_tilde22)) +
            y4 * (1 - 2 * y3) * Gauss_dist.cdf((a1 - mu_tilde1) / torch.sqrt(V_tilde11)) +
            y3 * y4
    )
    return P_y3_y4


class parametricLoss(nn.Module):
    def __init__(self):
        super(parametricLoss, self).__init__()

    def forward(self, y_hat, y, gamma12, gamma34, gamma3412, sigma1, sigma2):
        device = y_hat.device
        # gamma12左上角矩阵，gamma34右下角，gamma3412斜对角
        # sigma1连续变量标准差，
        gamma1234 = gamma3412.T

        gamma12_inv = torch.inverse(gamma12).to(device)
        V_tilde = gamma34 - torch.matmul(gamma3412, torch.matmul(gamma12_inv, gamma1234))

        Gauss_dist = normal.Normal(torch.zeros(1).cuda(), 1)
        # X1,X2,X3,X4对应位置为0,2,1,3
        label_left_myopia = y[0, :].to(device)
        label_right_myopia = y[2, :].to(device)
        outputs_left_myopia = y_hat[0, :].to(device)
        outputs_right_myopia = y_hat[2, :].to(device)

        ei = (y - y_hat).to(device)
        e1 = ei[1, :]
        e2 = ei[3, :]

        st_e1 = e1 / sigma1
        st_e2 = e2 / sigma2

        phi_inv_u1 = st_e1
        phi_inv_u2 = st_e2

        log_xxx_vec = torch.empty(0).to(device)

        for i, (q1i, q2i, p3i, p4i) in enumerate(zip(phi_inv_u1, phi_inv_u2, outputs_left_myopia, outputs_right_myopia),
                                                 0):
            q1i = q1i.unsqueeze(0)
            q2i = q2i.unsqueeze(0)
            q12i = torch.cat((q1i, q2i), dim=0).unsqueeze(0).T.to(device)
            mu_tilde = torch.matmul(gamma3412, torch.matmul(gamma12_inv, q12i)).T.squeeze()
            label3i = label_left_myopia[i]
            label4i = label_right_myopia[i]
            a1 = -Gauss_dist.icdf(p3i)
            a2 = -Gauss_dist.icdf(p4i)
            C = compute_discrete_probability(label3i, label4i, a1, a2, mu_tilde, V_tilde)
            logC = torch.log(C).to(device)
            log_xxx_i = - torch.matmul(q12i.T, torch.matmul(gamma12_inv, q12i)) / 2 + logC
            log_xxx_vec = torch.cat((log_xxx_vec, log_xxx_i), dim=0)

        loss = - torch.sum(log_xxx_vec)

        return loss


def calculate_sigma_hat(dl_train, net, device):
    net.eval()

    e1_vec = torch.empty(0).to(device)
    e2_vec = torch.empty(0).to(device)
    outputs_left_myopia_vec = torch.empty(0).to(device)
    outputs_right_myopia_vec = torch.empty(0).to(device)
    label_left_myopia_vec = torch.empty(0).to(device)
    label_right_myopia_vec = torch.empty(0).to(device)
    AL_left_vec = torch.empty(0).to(device)
    AL_right_vec = torch.empty(0).to(device)

    # calculate residual for every training sample
    with torch.no_grad():
        for i, data in enumerate(dl_train, 0):
            X_left, X_right, age, gender, *labels = data
            inputs = X_left, X_right, age, gender

            # get the data to GPU (if available)
            inputs = [_data.to(device) for _data in inputs]
            labels = [_label.to(device) for _label in labels]

            # Separate labels
            labels_left_AL = labels[2].squeeze().to(device)
            labels_right_AL = labels[4].squeeze().to(device)
            labels_left_high = labels[0].squeeze().to(device)
            labels_right_high = labels[1].squeeze().to(device)
            label_left_myopia_vec = torch.cat((label_left_myopia_vec, labels_left_high), dim=0)
            label_right_myopia_vec = torch.cat((label_right_myopia_vec, labels_right_high), dim=0)

            outputs_left, outputs_right = net(inputs)
            outputs_left_myopia = outputs_left[:, 0]
            outputs_right_myopia = outputs_right[:, 0]
            outputs_left_AL = outputs_left[:, 1]
            outputs_right_AL = outputs_right[:, 1]

            resid_left_AL = labels_left_AL - outputs_left_AL
            resid_right_AL = labels_right_AL - outputs_right_AL

            e1_vec = torch.cat((e1_vec, resid_left_AL), dim=0)
            e2_vec = torch.cat((e2_vec, resid_right_AL), dim=0)
            AL_left_vec = torch.cat((AL_left_vec, labels_left_AL), dim=0)
            AL_right_vec = torch.cat((AL_right_vec, labels_right_AL), dim=0)

            outputs_left_myopia_vec = torch.cat((outputs_left_myopia_vec, outputs_left_myopia), dim=0)
            outputs_right_myopia_vec = torch.cat((outputs_right_myopia_vec, outputs_right_myopia), dim=0)

    sigma_hat1 = torch.std(e1_vec).to(device)
    sigma_hat2 = torch.std(e2_vec).to(device)

    st_e1_vec = e1_vec / sigma_hat1
    st_e2_vec = e2_vec / sigma_hat2
    Gauss_dist = normal.Normal(torch.zeros(1).cuda(), 1)

    phi_inv_CDF1 = st_e1_vec
    phi_inv_CDF2 = st_e2_vec

    u3 = outputs_left_myopia_vec
    u4 = outputs_right_myopia_vec

    phi_inv_u3 = Gauss_dist.icdf(u3)
    phi_inv_u4 = Gauss_dist.icdf(u4)

    score_vec = torch.stack((phi_inv_CDF1, phi_inv_CDF2, phi_inv_u3, phi_inv_u4), dim=0).to(device)
    gamma = torch.corrcoef(score_vec)

    gamma12 = gamma[:2, :2]
    gamma34 = gamma[2:, 2:]
    gamma3412 = gamma[2:, :2]

    return gamma12, gamma34, gamma3412, sigma_hat1, sigma_hat2, gamma


