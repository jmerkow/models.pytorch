import torch
import torch.nn as nn
import torch.nn.functional as F

from models_pytorch.utils import Flatten, get_activation


def sum_pooling(x):
    return torch.sum(x.view(x.size(0), x.size(1), -1), dim=2)


class AttentionBlock(nn.Module):
    """
    F_att Block
    """

    def __init__(self, inplanes=1024, channels=512, nclasses=6):
        super(AttentionBlock, self).__init__()
        self.nclasses = nclasses
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(channels, self.nclasses, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv3(x)

        # Spatial softmax: Normalize attention maps
        x = self.spatial_softmax(x)

        return x

    def spatial_softmax(self, x):
        """Spatial softmax for normalizing attention maps"""
        return torch.softmax(x.view(1, self.nclasses, -1), 2).view_as(x)


class ConfidenceBlock(nn.Module):
    def __init__(self, inplanes=1024, channels=1024, nclasses=6):
        """
        Conv1: 1 x 1 x 1024
        """
        super(ConfidenceBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, nclasses, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)
        return x


class SpatialRegularizationBlock(nn.Module):
    """
    F_sr Block
    """

    def __init__(self, outplanes=512, nclasses=6):
        super(SpatialRegularizationBlock, self).__init__()
        self.kernel_size = 14
        self.conv1 = nn.Conv2d(nclasses, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(outplanes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(outplanes, outplanes * 4, kernel_size=self.kernel_size, stride=1, padding=0, groups=4,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes * 4)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(outplanes * 4, nclasses)
        self.flatten = Flatten()

    def forward(self, x):
        x = F.interpolate(x, size=(self.kernel_size, self.kernel_size))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class SpatialRegularizationNet(nn.Module):
    def __init__(self, inplanes=1024, fatt_channels=512,
                 conf_channels=1024, fsr_channels=512, nclasses=6):
        super(SpatialRegularizationNet, self).__init__()

        self.flatten = Flatten()
        # F_att
        self.fatt = AttentionBlock(inplanes=inplanes, channels=fatt_channels,
                                   nclasses=nclasses)
        # Conf (conv1)
        self.conf = ConfidenceBlock(inplanes=inplanes, channels=conf_channels,
                                    nclasses=nclasses)

        # F_sr
        self.fsr = SpatialRegularizationBlock(outplanes=fsr_channels, nclasses=nclasses)

    def forward(self, x):
        A = self.fatt(x)
        S = self.conf(x)
        S_sigmoid = torch.sigmoid(S)
        U = torch.mul(A, S_sigmoid)
        fsr_logits = self.fsr(U)
        sum_pool = torch.mul(A, S)
        sum_pool = sum_pooling(sum_pool)
        sum_pool = self.flatten(sum_pool)
        return fsr_logits, sum_pool


class SpatialRegularizationClassifier(nn.Module):

    def __init__(self, encoder_channels, activation='sigmoid', nclasses=6, tasks=('final',),
                 fatt_channels=512,
                 conf_channels=1024,
                 fsr_channels=512,
                 **kwargs):
        self.activation_type = str(activation)
        super(SpatialRegularizationClassifier, self).__init__()
        self.activation = get_activation(activation)

        self.tasks = tasks
        self.is_multi_task = len(self.tasks) > 1
        self.nclasses = nclasses

        self.srn = SpatialRegularizationNet(inplanes=encoder_channels[1], nclasses=nclasses,
                                            fatt_channels=fatt_channels,
                                            conf_channels=conf_channels,
                                            fsr_channels=fsr_channels, **kwargs)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), Flatten())
        self.fc = nn.Linear(encoder_channels[0], nclasses)

    def forward(self, features, **kwargs):

        outputs = {}
        x = self.pool(features[0])
        outputs['cls'] = self.fc(x)
        outputs['sr'], outputs['att'] = self.srn(features[1])
        outputs['final'] = outputs['cls'] + outputs['sr']

        if not self.is_multi_task:
            return outputs[self.tasks[0]]
        else:
            return {t: outputs[t] for t in self.tasks}

    def predict(self, features, **kwargs):
        if self.training:
            self.eval()
        with torch.no_grad():
            if self.is_multi_task:
                return {t: self.activation(o) for t, o in self.forward(features, **kwargs).items()}
            else:
                return self.activation(self.forward(features, **kwargs))

    def output_info(self):
        return {t: {'nclasses': self.nclasses, 'activation': self.activation_type} for t in self.tasks}
