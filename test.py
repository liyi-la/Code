import math
import numpy as np
import torch
import torch.nn as nn
device = torch.device("cuda:0",torch.cuda.is_available())

class pe(nn.Module):
    def __init__(self, dmodel=512, max_len=10000, droupout=0.1):
        super(pe, self).__init__()
        self.dmodel = dmodel
        self.max_len=max_len
        self.droupout = droupout
        pe = torch.zeros(dmodel, max_len)
        for pos in range(max_len):
            for i in range(0, dmodel, 2):
                pe[pos, i] = math.sin(pos/(10000**(i/dmodel)))
                pe[pos, i+1] = math.cos(pos/(10000**(i/dmodel)))
        self.pe = torch.FloatTensor(pe)
        self.register_buffer('pe',pe)
    def forward(self, x):
        x = x + self.pe[:,0:x.size(1)]
        return self.dropout(x.to(device))

