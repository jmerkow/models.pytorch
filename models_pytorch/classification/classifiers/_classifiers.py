import torch
import torch.nn as nn

from models_pytorch.utils import Flatten, get_activation

pooling_types = {
    'avg': nn.AdaptiveAvgPool2d,
    'max': nn.AdaptiveMaxPool2d,
}


class BasicClassifier(nn.Module):
    pooling_types = pooling_types

    def __init__(self, encoder_channels, nclasses, hidden_layers=(), pool_type='avg', channel_index=0,
                 activation='sigmoid', extra_inputs=None):

        self.nclasses = nclasses
        self.activation_type = str(activation)

        hidden_layers = hidden_layers or []

        self.channel_index = channel_index
        input_shape = encoder_channels[self.channel_index]

        super().__init__()

        self.required_inputs = []
        self.extra_input_fcs = None
        if extra_inputs is not None:
            self.required_inputs += list(extra_inputs.keys())
            self.extra_input_fcs = nn.ModuleDict()
            for key, nh in extra_inputs.items():
                self.extra_input_fcs[key] = nn.Linear(1, nh)
                input_shape += nh

        input_shapes = [input_shape]
        input_shapes.extend(hidden_layers)

        output_shapes = list(hidden_layers)
        output_shapes.append(nclasses)

        self.pool = self.pooling_types[pool_type]((1, 1))
        self.flatten = Flatten()
        modules = [nn.Sequential(), nn.Sequential()]  # this is for backwards compatz
        for ih, oh in zip(input_shapes, output_shapes):
            modules.extend([nn.Linear(ih, oh),
                            nn.ReLU()])

        modules.pop(-1)
        self.classifier = nn.Sequential(*modules)
        self.activation = get_activation(activation)

    def forward(self, features, **inputs):
        encoder_features = self.pool(features[self.channel_index])
        outputs = [encoder_features]
        if self.extra_input_fcs is not None:
            inputs = {k: v for k, v in inputs.items() if v is not None}
            assert all(k in inputs.keys() for k in self.extra_input_fcs.keys()), 'incorrect keys input into network'
            for k, fc in self.extra_input_fcs.items():
                h = self.flatten(fc(inputs[k]))
                h = torch.unsqueeze(torch.unsqueeze(h, -1), -1)
                outputs.append(h)
        return self.classifier(self.flatten((torch.cat(outputs, dim=1))))

    def predict(self, features, **inputs):
        if self.training:
            self.eval()
        with torch.no_grad():
            return self.activation(self.forward(features, **inputs))


classifier_types = {'basic': BasicClassifier}


def get_classifier(type, **kwargs):
    return classifier_types[type](**kwargs)


