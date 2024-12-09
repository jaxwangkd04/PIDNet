#!/usr/bin/python2 -utt
#-*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.PCLayer = nn.Sequential(nn.Conv2d(1, 1, 3, 1, 1, bias=True, groups=1, padding_mode='zeros'),
                                      nn.BatchNorm2d(1), nn.ReLU(),
                                      nn.Conv2d(1, 1, 3, 1, 1, bias=True, groups=1, padding_mode='zeros'),
                                      nn.BatchNorm2d(1), nn.ReLU())
        self.PCLayers = tuple( self.PCLayer  for i in range(52) )
        self.convnet1 = nn.Sequential(nn.Conv2d(52, 52, 5, 1, 2, bias=True,groups=52, padding_mode='zeros'),
                                      nn.BatchNorm2d(52), nn.ReLU(),
                                      nn.Conv2d(52, 52, 3, 1, 1, bias=True,groups=52, padding_mode='zeros'),
                                      nn.BatchNorm2d(52), nn.ReLU(),
                                      nn.MaxPool2d(2, stride=2)      )
        self.convnet2 = nn.Sequential(nn.Conv2d(52, 64, 3, 1, 1, bias=True, padding_mode='zeros'),
                                      nn.BatchNorm2d(64), nn.ReLU(),
                                      nn.Conv2d(64, 64, 3, 1, 1, bias=True, padding_mode='zeros'),
                                      nn.BatchNorm2d(64), nn.ReLU(),
                                      nn.MaxPool2d(2, stride=2)      )
        self.convnet3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=True, padding_mode='zeros'),
                                      nn.BatchNorm2d(128), nn.ReLU(),
                                      nn.Conv2d(128, 128, 3, 1, 1, bias=True, padding_mode='zeros'),
                                      nn.BatchNorm2d(128), nn.ReLU(),
                                      nn.MaxPool2d(2, stride=2)      )
        self.fc = nn.Sequential(nn.Linear(128*8*8, 128),
                                nn.ReLU(),
                                nn.Linear(128, 128)
                                           )
    def forward(self, x):
        xMulti = []
        for indx, aPclayer in enumerate(self.PCLayers):
            out1 = aPclayer(x[:,indx,:,:].unsqueeze(1))
            xMulti.append(out1)
        xMulti = torch.stack(xMulti,1)
        xMulti = xMulti.squeeze(2)
        output = self.convnet1(xMulti)
        output = self.convnet2(output)
        output = self.convnet3(output)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        output = F.normalize(output)
        return output
    def get_embedding(self, x):
        return self.forward(x)

class EmbeddingNetL2(EmbeddingNet):
    def __init__(self):
        super(EmbeddingNetL2, self).__init__()
    def forward(self, x):
        output = super(EmbeddingNetL2, self).forward(x)
        output /= output.pow(2).sum(1, keepdim=True).sqrt()
        return output
    def get_embedding(self, x):
        return self.forward(x)

class ClassificationNet(nn.Module):
    def __init__(self, embedding_net, n_classes):
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.PReLU()
        self.fc1 = nn.Linear(2, n_classes)
    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores
    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))

class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2
    def single_forward(self,x1):
        output1 = self.embedding_net(x1)
        return output1
    def get_embedding(self, x):
        return self.embedding_net(x)

class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net
    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3
    def get_embedding(self, x):
        return self.embedding_net(x)
