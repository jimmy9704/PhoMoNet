import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

class SPAD_Layer(nn.Module):
    def __init__(self,spad_dim):
        super(SPAD_Layer, self).__init__()
        self.spad_layer1 = nn.Linear(spad_dim[0], spad_dim[1])
        self.spad_layer2 = nn.Linear(spad_dim[1], spad_dim[2])
        self.spad_layer3 = nn.Linear(spad_dim[2], spad_dim[3])

        self.spad_relu = nn.LeakyReLU()

    def forward(self, x):
        # features = self.spad_encoder(x).reshape(x.shape[0],-1,1,1)
        b = x.shape[0]

        x_1 = self.spad_layer1(x)
        x_1_ = self.spad_relu(x_1)
        x_2 = self.spad_layer2(x_1_)
        x_2_ = self.spad_relu(x_2)
        x_3 = self.spad_layer3(x_2_)

        return [x_1.reshape(b,-1,1,1), x_2.reshape(b,-1,1,1), x_3.reshape(b,-1,1,1)]

class Photon_Module(nn.Module):
    def __init__(self,input_features):
        super(Photon_Module, self).__init__()
        self.distance = nn.CosineSimilarity(dim=1, eps=1e-6)
        self.relu_1 = nn.Sequential(\
            nn.BatchNorm2d(input_features),\
            nn.LeakyReLU(inplace=True))
        self.relu_2 = nn.Sequential(\
            nn.BatchNorm2d(input_features),\
            nn.LeakyReLU(inplace=True))
        self.spad_layer = nn.Conv2d(input_features, input_features, kernel_size=1, stride=1)
        self.conv1 = nn.Conv2d(input_features, input_features, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(input_features, input_features, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(input_features, input_features, kernel_size=3, stride=1, padding=1)

    def forward(self, x, spad):
        spad_exp = spad.expand_as(x)
        w = self.distance(x,spad_exp).unsqueeze(1)
        w = w.abs()

        x_attention = x*spad
        x_attention = self.conv1(x_attention)
        x = self.conv2(x)
        x = self.relu_1(x_attention*(1-w) + x*(w))
        features = self.relu_2(self.conv3(x))

        return features
