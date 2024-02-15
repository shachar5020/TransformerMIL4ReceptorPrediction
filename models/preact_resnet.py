"""preactresnet in pytorch
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks
    https://arxiv.org/abs/1603.05027
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import os

THIS_FILE = os.path.basename(os.path.realpath(__file__)).split('.')[0] + '.'


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(PreActResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.linear = nn.Linear(128*block.expansion, num_classes)
        self.model_name = ''

        self.is_HeatMap = False

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def change_num_classes(self, num_classes):
        self.linear = nn.Linear(self.linear.in_features, num_classes)

    def forward(self, x):
        if len(x.shape) == 5:
            num_of_bags, tiles_amount, _, tiles_size, _ = x.shape
            x = torch.reshape(x, (num_of_bags * tiles_amount, 3, tiles_size, tiles_size))

        if self.is_HeatMap:
            if self.training is True:
                raise Exception('Pay Attention that the model in not in eval mode')
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)


            initial_image_size = out.shape[2]
            vectored_image = torch.transpose(torch.reshape(out.squeeze(0), (out.size(1), out.size(2) * out.size(3))), 1, 0)
            vectored_heat_image_2_channels = self.linear(vectored_image)
            vectored_heat_image = vectored_heat_image_2_channels[:, 1] - vectored_heat_image_2_channels[:, 0]
            small_heat_map = vectored_heat_image.view(1, 1, initial_image_size, initial_image_size)
            return small_heat_map


        else:
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

            out = F.avg_pool2d(out, out.shape[3])
            out = out.view(out.size(0), -1)
            features = out
            #out = self.linear(self.dropout(out))
            out = self.linear(out)

            if self.training and torch.any(torch.isnan(out)):
                import pandas as pd
                print('Model: Found NaN in Score')
                scores_dict = {'Scores': list(torch.reshape(out, (18,)).detach().cpu().numpy())}
                scores_DF = pd.DataFrame(scores_dict).transpose()
                scores_DF.to_excel('debug_data_scores_from_model.xlsx')

                weights = list(torch.reshape(self.linear.weight, (512,)).detach().cpu().numpy())
                bias = list(self.linear.bias.detach().cpu().numpy()) + [-1] * 511

                linear_layer_dict = {'Weights': weights,
                                     'Bias': bias}
                linear_layer_DF = pd.DataFrame(linear_layer_dict).transpose()
                linear_layer_DF.to_excel('debug_data_models_linear_layer.xlsx')

                features_DF = pd.DataFrame(features.detach().cpu().numpy())
                features_DF.to_excel('debug_data_features_from_models.xlsx')

                print('Finished saving 3 debug files from model code')

            return out, features




def PreActResNet50(train_classifier_only=False, num_classes=2):
    model = PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.model_name = THIS_FILE + 'PreActResNet50()'

    if train_classifier_only:
        model.model_name = THIS_FILE + 'PreActResNet50(train_classifier_only=True)'
        if num_classes != 2:
            model.model_name = model.model_name[:-1] + ', num_classes=' + str(num_classes) + ')'
        for param in model.parameters():
            param.requires_grad = False
        for param in model.linear.parameters():
            param.requires_grad = True

    elif num_classes != 2:
        model.model_name = model.model_name[:-1] + 'num_classes=' + str(num_classes) + ')'

    return model



