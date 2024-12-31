from __future__ import print_function, division

import torch
import torch.nn as nn

from torch.distributions import normal
import multivariate_normal_cdf as Phi


def BerCDF(x, p):
    cdf = torch.where(x < 0, 0.0, torch.where(x < 1, 1 - p, 1))
    return cdf


def Copula_(u, miu_star, gamma_star):
    Gauss_dist = normal.Normal(torch.zeros(1).cuda(), 1)
    return Gauss_dist.cdf((Gauss_dist.icdf(u) - miu_star) / torch.sqrt(1 - gamma_star))


def Copula(u3, u4, miu_star, gamma_star):
    Gauss_dist = normal.Normal(torch.zeros(1).cuda(), 1)
    if u3 == 0 or u4 == 0:
        C = torch.tensor([0])
    elif u3 == 1 and u4 == 1:
        C = torch.tensor([1])
    elif u3 == 1:
        miu2_star = miu_star[1]
        gamma22_star = gamma_star[1][1].unsqueeze(0).unsqueeze(0)
        phi_inv_u4 = Gauss_dist.icdf(u4)
        C = Phi(phi_inv_u4, miu2_star, gamma22_star)
    elif u4 == 1:
        miu1_star = miu_star[0]
        gamma11_star = gamma_star[0][0].unsqueeze(0).unsqueeze(0)
        phi_inv_u3 = Gauss_dist.icdf(u3)
        C = Phi(phi_inv_u3, miu1_star, gamma11_star)
    else:
        phi_inv_u3 = Gauss_dist.icdf(u3)
        phi_inv_u4 = Gauss_dist.icdf(u4)
        upper = torch.cat((phi_inv_u3, phi_inv_u4), dim=0)
        C = Phi(upper, miu_star, gamma_star)
    return C


class parametricLoss(nn.Module):
    def __init__(self):
        super(parametricLoss, self).__init__()

    def forward(self, y_hat, y, gamma12, gamma34, gamma3412, sigma1, sigma2):
        device = y_hat.device
        gamma12_inv = torch.inverse(gamma12).to(device)
        gamma1234 = gamma3412.T

        gamma_star = gamma34 - torch.matmul(gamma3412, torch.matmul(gamma12_inv, gamma1234))

        dist1 = normal.Normal(torch.zeros(1).cuda(), sigma1)
        dist2 = normal.Normal(torch.zeros(1).cuda(), sigma2)

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
            miu_star = torch.matmul(gamma3412, torch.matmul(gamma12_inv, q12i)).T.squeeze()

            label3i = label_left_myopia[i]
            label4i = label_right_myopia[i]

            CDF3i = BerCDF(label3i, p3i)
            CDF4i = BerCDF(label4i, p4i)

            CDF3i_leftlimit = BerCDF(label3i - 1, p3i)
            CDF4i_leftlimit = BerCDF(label4i - 1, p4i)

            C1i = Copula(CDF3i_leftlimit, CDF4i_leftlimit, miu_star, gamma_star).to(device)
            C2i = Copula(CDF3i_leftlimit, CDF4i, miu_star, gamma_star).to(device)
            C3i = Copula(CDF3i, CDF4i_leftlimit, miu_star, gamma_star).to(device)
            C4i = Copula(CDF3i, CDF4i, miu_star, gamma_star).to(device)

            Ci = C1i - C2i - C3i + C4i
            # print("Ci:",Ci)
            logCi = torch.log(Ci).to(device)

            log_xxx_i = - torch.matmul(q12i.T, torch.matmul(gamma12_inv, q12i)) / 2 + logCi
            log_xxx_vec = torch.cat((log_xxx_vec, log_xxx_i), dim=0)

        loss = - torch.sum(log_xxx_vec)

        return loss
