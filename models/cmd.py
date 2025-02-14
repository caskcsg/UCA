import torch.nn as nn
import torch

# class CMD(nn.Module):
#     """
#     Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
#     """

#     def __init__(self, trade_off=0.1):
#         super(CMD, self).__init__()
#         self.trade_off = trade_off

#     def forward(self, f_s, f_t):
#         U_s, _, _ = torch.svd(f_s.t())
#         U_t, _, _ = torch.svd(f_t.t())
#         P_s, cosine, P_t = torch.svd(torch.mm(U_s.t(), U_t))
#         sine = torch.sqrt(1 - torch.pow(cosine, 2))
#         rsd = torch.norm(sine, 1)  # Representation Subspace Distance
#         bmp = torch.norm(torch.abs(P_s) - torch.abs(P_t), 2)  # Base Mismatch Penalization
#         return rsd + self.trade_off * bmp
class CMD(nn.Module):
    """
    Adapted from https://github.com/wzell/cmd/blob/master/models/domain_regularizer.py
    """

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments=5):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)