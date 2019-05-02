import torch
from torch import nn
import numpy as np
from tqdm import tqdm

from .resnet import ResnetBlock, ResnetGenerator


def compute_kernel(n_input, n_output, kernel, stride, n_features):
    weights = torch.randn([n_output, n_input, kernel[0], kernel[0]]) * 0.001
    center = kernel[0] // 2

    assert stride[0] == stride[1]
    assert kernel[0] == kernel[1]
    s = stride[0]
    for feature in range(n_features):
        for i, j in np.ndindex(stride):
            weights[feature * s * s + i * s + j, feature, center + i, center + j] = 1

    return weights


def compute_kernel_transpose(n_input, n_output, kernel, stride, n_features):
    weights = torch.zeros([n_input, n_output, kernel[0], kernel[0]]) * 0.001
    center = kernel[0] // 2

    assert stride[0] == stride[1]
    assert kernel[0] == kernel[1]
    assert stride[0] < kernel[0]

    s = stride[0]
    for feature in range(n_features // s // s):
        for i, j in np.ndindex(stride):
            weights[feature * s * s + i * s + j, feature, center + i, center + j] = 1

    return weights


def zero_resblock(layer):
    layers = list(layer.conv_block.children())

    nn.init.normal(layers[0].weight, 0.00, 0.001)
    if layers[0].bias is not None: nn.init.constant(layers[0].bias, 0)
    nn.init.constant(layers[1].weight, 1)
    if layers[1].bias is not None: nn.init.constant(layers[1].bias, 0)
    nn.init.normal(layers[3].weight, 0.00, 0.001)
    if layers[3].bias is not None: nn.init.constant(layers[3].bias, 0)
    nn.init.constant(layers[4].weight, 1)
    nn.init.constant(layers[4].running_var, 1)
    nn.init.constant(layers[4].running_mean, 0)
    if layers[4].bias is not None: nn.init.constant(layers[4].bias, 0)


def ResnetGeneratorIdentity(input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                            padding_type='zero'):

    model = ResnetGenerator(input_nc, output_nc, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                            n_blocks=n_blocks, padding_type=padding_type)

    n_features_info = 3

    progress_bar = tqdm(model.model.children())
    for layer in progress_bar:
        if layer.__class__ == nn.BatchNorm2d:
            nn.init.zeros_(layer.bias)
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.running_mean)
            nn.init.ones_(layer.running_var)
        elif layer.__class__ == ResnetBlock:
            zero_resblock(layer)
        elif layer.__class__ == nn.Conv2d:
            kernel = compute_kernel(layer.in_channels, layer.out_channels,
                                    layer.kernel_size, layer.stride, n_features_info)
            layer.weight.data = kernel
            if layer.bias is not None: nn.init.zeros_(layer.bias)
            n_features_info *= layer.stride[0] * layer.stride[1]
        elif layer.__class__ == nn.ConvTranspose2d:
            kernel = compute_kernel_transpose(layer.in_channels, layer.out_channels,
                                              layer.kernel_size, layer.stride, n_features_info)
            layer.weight.data = kernel
            if layer.bias is not None: nn.init.zeros_(layer.bias)
            n_features_info = n_features_info // (layer.stride[0] * layer.stride[1])

        progress_bar.set_description(layer.__class__.__name__)

    return model