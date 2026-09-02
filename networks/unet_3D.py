# -*- coding: utf-8 -*-
"""
An implementation of the 3D U-Net paper:
     Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger:
     3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation.
     MICCAI (2) 2016: 424-432
Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
The implementation is borrowed from: https://github.com/ozan-oktay/Attention-Gated-Networks
"""
import math
import numpy as np
import torch.nn as nn
import os
import sys
import torch
import torch.nn.functional as F

from networks.networks_other import init_weights
from networks.utils import UnetConv3, UnetUp3, UnetUp3_CT
# from .GaborConv import Gabor3DConv
# import monai
# from monai.networks.layers.filtering.py import *
from functools import partial
nonlinearity = partial(F.relu, inplace=True)
from networks.cbam import *
#from dcn.modules.deform_conv import *

class unet_3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(unet_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # self.gaborConv = Gabor3DConv(filters[0],filters[0],kernel_size=(
        #     3, 3, 3),sigma=1.0,theta=np.pi/4,lambd=5.0,gamma=1.0,psi=0.0,padding_size=(1, 1, 1))

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)



        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        center = self.dropout1(center)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        # up1 = self.gaborConv(up1)
        up1 = self.dropout2(up1)

        final = self.final(up1)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class unet_3D_dt(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(unet_3D_dt, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))



        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # self.filter = monai.networks.layers.filtering.TrainableBilateralFilter()
        # self.gaborConv = Gabor3DConv(filters[0], filters[0], kernel_size=(
        #     3, 3, 3), sigma=1.0, theta=np.pi / 4, lambd=5.0, gamma=1.0, psi=0.0, padding_size=(1, 1, 1))

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)
        # self.tanh = nn.Tanh()

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        ###dropout
        center = self.dropout1(center)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        #dropout
        up1 = self.dropout2(up1)

        # up1 = self.filter.apply(up1)

        final = self.final(up1)

        # dis = self.tanh(final)

        # print(final.shape,dis.shape)

        return center,final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p

# ---------------------------------------------------- #
# （2）自注意力机制
class self_attention(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, in_channal = 512, out_channal =1024, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1)):
        # 继承父类初始化方法
        super(self_attention, self).__init__()

        # # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        # padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,in_c,h,w]==>[b,1,h,w]
        self.conv_k = nn.Conv3d(in_channal, 1, kernel_size, init_stride, padding_size)
        self.conv_q = nn.Conv3d(in_channal, 1, kernel_size, init_stride, padding_size)
        self.conv_v = nn.Conv3d(in_channal, 1, kernel_size, init_stride, padding_size)
        self.softmax = nn.Softmax(dim=1)

        self.conv = nn.Conv3d(1, out_channal, kernel_size, init_stride, padding_size)
        # self.conv = nn.Sequential(
        #                 nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
        #                       padding=padding, bias=False),
        #                 nn.BatchNorm2d(1),
        #                 nn.Sigmoid(),
        #                   )
        # sigmoid函数
        # self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        k = self.conv_k(inputs)
        q = self.conv_q(inputs)
        v = self.conv_v(inputs)
        k_q = self.softmax(k * q)
        outputs = self.conv(k_q * v)

        return outputs



class Dblock(nn.Module):
    def __init__(self, in_channal = 512, out_channal = 1024, kernel_size=(3,3,3)):
        super(Dblock, self).__init__()
        init_stride = (1, 1, 1)
        dilation_size1 = (1, 1, 1)
        dilation_size2 = (2, 2, 2)
        dilation_size3 = (4, 4, 4)
        dilation_size4 = (8, 8, 8)
        padding_size1 = (1, 1, 1)
        padding_size2 = (2, 2, 2)
        padding_size3 = (4, 4, 4)
        padding_size4 = (8, 8, 8)
        self.dilate1 = nn.Conv3d(in_channal, out_channal, kernel_size, init_stride, padding_size1, dilation_size1)
        self.dilate2 = nn.Conv3d(out_channal, out_channal, kernel_size, init_stride, padding_size2, dilation_size2)
        self.dilate3 = nn.Conv3d(out_channal, out_channal, kernel_size, init_stride, padding_size3, dilation_size3)
        self.dilate4 = nn.Conv3d(out_channal, out_channal, kernel_size, init_stride, padding_size4, dilation_size4)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)

        # self.cbam1 = cbam(in_channel=channel, ratio=4)
        # self.cbam2 = cbam(in_channel=channel, ratio=8)
        # self.cbam3 = cbam(in_channel=channel, ratio=16)
        # self.cbam4 = cbam(in_channel=channel, ratio=32)

        # self.Conv = nn.Conv2d(channel*5, channel, kernel_size=3, dilation=1, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # print(x.shape) # [4, 256, 6, 6, 6]
        dilate1_out = nonlinearity(self.dilate1(x))
        # dilate1_out = self.cbam1(dilate1_out)
        # print(dilate1_out.shape)  #[4, 256, 6, 6, 6]
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        # print(dilate2_out.shape)
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        # print(dilate3_out.shape)
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        # print(dilate4_out.shape)
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out
        # out = x + self.cbam1(dilate1_out) + self.cbam2(dilate2_out) + self.cbam3(dilate3_out) + self.cbam4(dilate4_out)  # + dilate5_out
        # out = torch.cat((x,self.cbam1(dilate1_out),self.cbam2(dilate2_out), self.cbam3(dilate3_out),self.cbam4(dilate4_out)),dim=1)
        # out = self.Conv(out)

        return out


class unet_3D_dt_c(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(unet_3D_dt_c, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        #print(filters)

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # self.dblock = Dblock(filters[3], filters[4], kernel_size=(3, 3, 3))

        # self.atten = self_attention(in_channal = filters[3], out_channal = filters[4])

        # self.atten1 = self_attention(in_channal=self.in_channels, out_channal=filters[0])
        # self.atten2 = self_attention(in_channal=filters[0], out_channal=filters[1])
        # self.atten3 = self_attention(in_channal=filters[1], out_channal=filters[2])
        # self.atten4 = self_attention(in_channal=filters[2], out_channal=filters[3])
        # self.cbam1 = cbam(in_channel=filters[0],ratio=16)
        # self.cbam2 = cbam(in_channel=filters[1],ratio=16)
        # self.cbam3 = cbam(in_channel=filters[2],ratio=16)
        # self.cbam4 = cbam(in_channel=filters[3],ratio=16)

        self.cbam = cbam(in_channel=filters[4],ratio=16)


        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # self.filter = monai.networks.layers.filtering.TrainableBilateralFilter()
        # self.gaborConv = Gabor3DConv(filters[0], filters[0], kernel_size=(
        #     3, 3, 3), sigma=1.0, theta=np.pi / 4, lambd=5.0, gamma=1.0, psi=0.0, padding_size=(1, 1, 1))

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)
        self.tanh = nn.Tanh()

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        # conv1 = self.cbam1(conv1)
        maxpool1 = self.maxpool1(conv1)
        #print(maxpool1.shape)

        conv2 = self.conv2(maxpool1)
        # conv2 = self.cbam2(conv2)
        maxpool2 = self.maxpool2(conv2)
        #print(maxpool2.shape)

        conv3 = self.conv3(maxpool2)
        # conv3 = self.cbam3(conv3)
        maxpool3 = self.maxpool3(conv3)
        #print(maxpool3.shape)

        conv4 = self.conv4(maxpool3)
        # conv4 = self.cbam4(conv4)
        maxpool4 = self.maxpool4(conv4)
        #print(maxpool4.shape)


        ###dropout
        maxpool4 = self.dropout1(maxpool4)

        center = self.center(maxpool4) #+ self.atten(maxpool4)
        center = self.cbam(center)
        # center = self.dropout1(center)

        # center = self.dropout1(maxpool4)
        # center = self.center(center) + self.atten(center)
        # center = self.cbam(center)

        #print(conv4.shape,center.shape)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        #dropout
        up1 = self.dropout2(up1)

        # up1 = self.filter.apply(up1)

        final = self.final(up1)

        dis = self.tanh(final)

        # print(final.shape,dis.shape)

        return dis,final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class ResBlock_3d(nn.Module):
    def __init__(self, nf):
        super(ResBlock_3d, self).__init__()
        self.dcn0 = DeformConvPack_d(nf, nf, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.dcn1 = DeformConvPack_d(nf, nf, kernel_size=3, stride=1, padding=1, dimension='HW')
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        return self.dcn1(self.lrelu(self.dcn0(x))) + x



class unet_3D_Deform(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(unet_3D_Deform, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]
        #print(filters)


        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.residual_layer1 = self.make_layer(partial(ResBlock_3d, filters[0]), 5)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))


        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.residual_layer2 = self.make_layer(partial(ResBlock_3d, filters[1]), 5)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.residual_layer3 = self.make_layer(partial(ResBlock_3d, filters[2]), 5)
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.residual_layer4 = self.make_layer(partial(ResBlock_3d, filters[3]), 5)
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))


        # self.dblock = Dblock(filters[3], filters[4], kernel_size=(3, 3, 3))

        # self.atten = self_attention(in_channal = filters[3], out_channal = filters[4])

        # self.atten1 = self_attention(in_channal=self.in_channels, out_channal=filters[0])
        # self.atten2 = self_attention(in_channal=filters[0], out_channal=filters[1])
        # self.atten3 = self_attention(in_channal=filters[1], out_channal=filters[2])
        # self.atten4 = self_attention(in_channal=filters[2], out_channal=filters[3])
        # self.cbam1 = cbam(in_channel=filters[0],ratio=16)
        # self.cbam2 = cbam(in_channel=filters[1],ratio=16)
        # self.cbam3 = cbam(in_channel=filters[2],ratio=16)
        # self.cbam4 = cbam(in_channel=filters[3],ratio=16)

        self.cbam = cbam(in_channel=filters[4],ratio=16)


        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # self.filter = monai.networks.layers.filtering.TrainableBilateralFilter()
        # self.gaborConv = Gabor3DConv(filters[0], filters[0], kernel_size=(
        #     3, 3, 3), sigma=1.0, theta=np.pi / 4, lambd=5.0, gamma=1.0, psi=0.0, padding_size=(1, 1, 1))

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)
        self.tanh = nn.Tanh()

        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        # conv1 = self.cbam1(conv1)
        conv1 = self.residual_layer1(conv1)
        maxpool1 = self.maxpool1(conv1)
        #print(conv1.shape)

        conv2 = self.conv2(maxpool1)
        # conv2 = self.cbam2(conv2)
        conv2 = self.residual_layer2(conv2)
        maxpool2 = self.maxpool2(conv2)
        #print(conv2.shape)

        conv3 = self.conv3(maxpool2)
        # conv3 = self.cbam3(conv3)
        conv3 = self.residual_layer3(conv3)
        maxpool3 = self.maxpool3(conv3)
        #print(conv3.shape)

        conv4 = self.conv4(maxpool3)
        # conv4 = self.cbam4(conv4)
        conv4 = self.residual_layer4(conv4)
        maxpool4 = self.maxpool4(conv4)
        #print(conv4.shape)


        ###dropout
        maxpool4 = self.dropout1(maxpool4)

        center = self.center(maxpool4) #+ self.atten(maxpool4)
        center = self.cbam(center)
        # center = self.dropout1(center)

        # center = self.dropout1(maxpool4)
        # center = self.center(center) + self.atten(center)
        # center = self.cbam(center)

        #print(conv4.shape,center.shape)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        #dropout
        up1 = self.dropout2(up1)

        # up1 = self.filter.apply(up1)

        final = self.final(up1)

        dis = self.tanh(final)

        # print(final.shape,dis.shape)

        return dis,final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


if __name__=='__main__':
     input=torch.rand(8, 1, 112,112,80).cuda()
     net=unet_3D_Deform(n_classes=2, in_channels=1).cuda()
     center,out=net(input)
     #print(out.shape)