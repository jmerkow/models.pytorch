import torch.nn.functional as F

from pretrainedmodels.models.xception import Xception, pretrained_settings


class XceptionEncoder(Xception):

    def __init__(self, *args, **kwargs):
        last_only = kwargs.pop('last_only', True)
        super().__init__(*args, **kwargs)
        self.last_only = last_only
        del self.fc

    def forward(self, input):
        x0 = self.conv1(input)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)  # 1/2, 32 out
        x0 = self.conv2(x0)
        x0 = self.bn2(x0)
        x0 = self.relu(x0)  # 1/2, 64 out

        x1 = self.block1(x0)  # 1/4, 128 out
        x2 = self.block2(x1)  # 1/8, 256 out
        x3 = self.block3(x2)  # 1/16, 728 out

        x3 = self.block4(x3)
        x3 = self.block5(x3)
        x3 = self.block6(x3)
        x3 = self.block7(x3)
        x3 = self.block8(x3)
        x3 = self.block9(x3)
        x3 = self.block10(x3)
        x3 = self.block11(x3)  # 1/16 728

        x4 = self.block12(x3)  # 1/32, 1024 out

        x4 = self.conv3(x4)
        x4 = self.bn3(x4)
        x4 = self.relu(x4)

        x4 = self.conv4(x4)
        x4 = self.bn4(x4)

        if not self.last_only:
            ## The scale isn't exactly 2 it doesn't work normally... So instead we are
            return [
                F.relu(x4, inplace=True),  # 1/32, 2048
                F.relu(x3, inplace=True),  # 1/16, 728
                F.relu(x2, inplace=True),  # 1/8, 256
                F.relu(x1, inplace=True),  # 1/4, 128
                x0,  # 1/2, 64
            ]
        return [F.relu(x4, inplace=True), None, None, None, None, ]

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('fc.bias')
        state_dict.pop('fc.weight')
        super().load_state_dict(state_dict, **kwargs)


xception_encoders = {
    'xception_last_only': {
        'encoder': XceptionEncoder,
        'out_shapes': (2048, 0, 0, 0, 0),
        'pretrained_settings': pretrained_settings['xception'],
        'params': {'last_only': True},
    },
    'xception': {
        'encoder': XceptionEncoder,
        'out_shapes': (2048, 768, 256, 128, 64),
        'pretrained_settings': pretrained_settings['xception'],
        'params': {'last_only': False},
    }
}
