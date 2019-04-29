from torch import nn
import torch
from functools import partial


class ImageConcater(nn.Module):

    def __init__(self, image_size, n_channels):
        super(ImageConcater, self).__init__()
        self.image = nn.Parameter(torch.randn(n_channels, image_size, image_size))

    def forward(self, image):
        batchsize = self.image.shape[0]
        torch.cat((self.image.repeat((batchsize, 1, 1, 1)), image), dim=0)


class BlenderNet(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, image_size=256):
        """
        Construct a BlenderNet based generator

        Parameters
        ----------
        input_nc: int
            the number of channels in input images
        output_nc: int
            the number of channels in output images
        ngf: int
            the number of filters in the last conv layer
        norm_layer:
            normalization layer
        use_dropout: bool
            if use dropout layers
        n_blocks: int
            the number of ResNet blocks
        image_size: int
        """

        assert(n_blocks >= 0)

        super(BlenderNet, self).__init__()

        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=3, bias=True), norm_layer(ngf), nn.ReLU()]

        n_downsampling = 2

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=True),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU()]

        mult = 2 ** n_downsampling

        model += [ImageConcater(64, ngf * mult / 4),
                  nn.Conv2d(ngf * mult, ngf * mult, kernel_size=3, stride=1, padding=1, bias=True),
                  nn.ReLU()]

        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=True),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU()]

        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=3)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out
