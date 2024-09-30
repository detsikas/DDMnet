import torch
import sys


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, filters, activation, kernel_size=3, dilation_rate=1):
        super(ConvBlock, self).__init__()
        self.activation = activation
        self.out_channels = filters
        self.conv_layer = torch.nn.Conv2d(
            in_channels=in_channels, out_channels=filters, kernel_size=kernel_size, dilation=dilation_rate, padding='same')
        self.bn_layer = torch.nn.BatchNorm2d(num_features=filters)
        if activation == 'relu':
            self.activation_layer = torch.nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation_layer = torch.nn.LeakyReLU()
        elif activation is None:
            self.activation_layer = None
        else:
            print('Bad activation')
            sys.exit(0)

    def forward(self, inputs):
        x = self.conv_layer(inputs)
        x = self.bn_layer(x)
        if self.activation is not None:
            x = self.activation_layer(x)
        return x


class MultiResBlock(torch.nn.Module):
    def __init__(self, in_channels, filters, activation, dilation_rate=1, alpha=1.67):
        super(MultiResBlock, self).__init__()
        W = alpha * filters
        self.output_filters = int(
            W * 0.167) + int(W * 0.333) + int(W * 0.5)
        self.activation = activation
        self.sc_conv_block = ConvBlock(in_channels=in_channels, filters=int(
            W * 0.167) + int(W * 0.333) + int(W * 0.5), kernel_size=1, activation=activation, dilation_rate=dilation_rate)
        self.conv_block_3 = ConvBlock(in_channels=in_channels, filters=int(
            W * 0.167), activation=activation, dilation_rate=dilation_rate)
        self.conv_block_5 = ConvBlock(in_channels=self.conv_block_3.out_channels, filters=int(
            W * 0.333), activation=activation, dilation_rate=dilation_rate)
        self.conv_block_7 = ConvBlock(in_channels=self.conv_block_5.out_channels, filters=int(
            W * 0.5), activation=activation, dilation_rate=dilation_rate)

        self.bn_layer = torch.nn.BatchNorm2d(num_features=self.output_filters)
        self.bn_output_layer = torch.nn.BatchNorm2d(
            num_features=self.output_filters)
        if activation == 'leaky_relu':
            self.activation_layer = torch.nn.LeakyReLU()
        elif activation == 'relu':
            self.activation_layer = torch.nn.ReLU()
        elif activation is None:
            self.activation_layer = None
        else:
            print('Bad extivation')
            sys.exit(0)

    def forward(self, inputs):
        shortcut = self.sc_conv_block(inputs)
        conv3x3 = self.conv_block_3(inputs)
        conv5x5 = self.conv_block_5(conv3x3)
        conv7x7 = self.conv_block_7(conv5x5)

        if self.activation is None:
            mresx = torch.cat((conv3x3, conv5x5, conv7x7), dim=1)
            mresx = mresx + shortcut
            return self.bn_output_layer(mresx)

        mresx = torch.cat((conv3x3, conv5x5, conv7x7), dim=1)
        mresx = self.bn_layer(mresx)
        mresx = mresx + shortcut
        mresx = self.bn_output_layer(mresx)
        mresx = self.activation_layer(mresx)

        return mresx
