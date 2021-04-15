import torch
import torch.nn as nn
from torch.nn import Parameter
import numpy as np
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.use_cuda = args.cuda
        self.P = args.window
        self.m = data.m
        self.hidR = args.hidRNN
        #self.RNN1 = nn.LSTM(self.m, self.hidR)
        #self.RNN2 = nn.LSTM(self.m, self.hidR)
		self.RNN1 = nn.GRU(self.m, self.hidR)
        self.RNN2 = nn.GRU(self.m, self.hidR)
        self.mask_mat = nn.Parameter(torch.Tensor(self.m, self.m))
        self.mask_mat_1 = nn.Parameter(torch.Tensor(1))
        self.mask_mat_2 = nn.Parameter(torch.Tensor(1))
        self.adj = data.adj
        self.dropout = nn.Dropout(p=args.dropout)
        self.linear1 = nn.Linear(self.hidR, self.m)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

    def forward(self, x):

        masked_adj = self.adj * self.mask_mat
        x = x.matmul(masked_adj)
        r = x.permute(1, 0, 2).contiguous()
        r2 = r[-4:,:,:]
        a, r = self.RNN1(r)
        b, r2 = self.RNN2(r2)
        r = self.dropout(torch.squeeze(r, 0)) # [1]     
        r2 = self.dropout(torch.squeeze(r2, 0)) # [1]
        r = self.mask_mat_1 * r + self.mask_mat_2 * r2
        res = self.linear1(r)

        if self.output is not None:
            res = self.output(res).float()

        return res
