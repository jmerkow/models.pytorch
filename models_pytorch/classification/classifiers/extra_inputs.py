import torch
from torch import nn

from models_pytorch.utils import Flatten, get_activation


class ExtraScalarInputsClassifier(nn.Module):

    def __init__(self, encoder_channels, num_hidden=None, nclasses=1, activation='sigmoid', **input_blocks):
        input_blocks = dict(input_blocks)
        self.activation_type = str(activation)
        self.nclasses = nclasses

        super().__init__()

        self.required_inputs = list(input_blocks.keys())
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()

        self.module_list = nn.ModuleDict()
        total_input = encoder_channels[0]
        for key, nh in input_blocks.items():
            self.module_list[key] = nn.Linear(1, nh)
            total_input += nh

        if num_hidden is not None:
            self.classifier = nn.Sequential(
                Flatten(),
                nn.Linear(total_input, num_hidden),
                nn.ReLU(),
                nn.Linear(num_hidden, nclasses)
            )
        else:
            self.classifier = nn.Sequential(
                Flatten(),
                nn.Linear(total_input, nclasses))
        self.activation = get_activation(activation)
        self.is_multi_task = False

    def forward(self, encoder_input, **inputs):
        inputs = {k: v for k, v in inputs.items() if v is not None}
        assert all(k in inputs.keys() for k in self.module_list.keys()), 'incorrect keys input into network'
        outputs = [self.pool(encoder_input[0])]
        for k, fc in self.module_list.items():
            h = self.flatten(fc(inputs[k]))
            h = torch.unsqueeze(torch.unsqueeze(h, -1), -1)
            outputs.append(h)
        return self.classifier(torch.cat(outputs, dim=1))

    def predict(self, encoder_input, **inputs):
        return self.activation(self.forward(encoder_input, **inputs))

    @property
    def tasks(self):
        return ['final']

    def output_info(self):
        return {'final': {'nclasses': self.nclasses, 'activation': self.activation_type}}
