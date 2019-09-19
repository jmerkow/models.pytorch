from efficientnet_pytorch.model import EfficientNet
from efficientnet_pytorch.utils import get_model_params, relu_fn, url_map

backbone_indices = {
    'efficientnet-b0': [0, 2, 4, 10],
    'efficientnet-b1': [1, 4, 7, 15],
    'efficientnet-b2': [1, 4, 7, 15],
    'efficientnet-b3': [1, 4, 7, 17],
    'efficientnet-b4': [1, 5, 9, 21],
    'efficientnet-b5': [2, 7, 12, 26],
    'efficientnet-b6': [2, 8, 14, 30],
    'efficientnet-b7': [3, 10, 17, 37]
}


class EfficientNetEncoder(EfficientNet):

    def __init__(self, model_name, **override_params):
        blocks_args, global_params = get_model_params(model_name, override_params)
        self.backbone_indices = backbone_indices[model_name]
        super().__init__(blocks_args, global_params)
        del self._fc

    def forward(self, inputs):
        """ Returns output of the final convolution layer """

        backbone_indices = getattr(self, 'backbone_indices', None)
        if not backbone_indices:
            raise ValueError('no backbone indices, something went wrong!')

        # Stem
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

        features = []
        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if idx in backbone_indices:
                features.insert(0, x)

        # Head
        x = relu_fn(self._bn1(self._conv_head(x)))
        features.insert(0, x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('_fc.bias')
        state_dict.pop('_fc.weight')
        super().load_state_dict(state_dict, **kwargs)


pretrained_settings = {'efficientnet-b0': {'imagenet': {'url': url_map['efficientnet-b0'],
                                                        'input_space': 'RGB',
                                                        'input_size': [3, 244, 244],
                                                        'input_range': [0, 1],
                                                        'mean': [0.485, 0.456, 0.406],
                                                        'std': [0.229, 0.224, 0.225],
                                                        'num_classes': 1000,
                                                        'scale': 0.8975}},
                       'efficientnet-b1': {'imagenet': {'url': url_map['efficientnet-b1'],
                                                        'input_space': 'RGB',
                                                        'input_size': [3, 244, 244],
                                                        'input_range': [0, 1],
                                                        'mean': [0.485, 0.456, 0.406],
                                                        'std': [0.229, 0.224, 0.225],
                                                        'num_classes': 1000,
                                                        'scale': 0.8975}},
                       'efficientnet-b2': {'imagenet': {'url': url_map['efficientnet-b2'],
                                                        'input_space': 'RGB',
                                                        'input_size': [3, 244, 244],
                                                        'input_range': [0, 1],
                                                        'mean': [0.485, 0.456, 0.406],
                                                        'std': [0.229, 0.224, 0.225],
                                                        'num_classes': 1000,
                                                        'scale': 0.8975}},
                       'efficientnet-b3': {'imagenet': {'url': url_map['efficientnet-b3'],
                                                        'input_space': 'RGB',
                                                        'input_size': [3, 244, 244],
                                                        'input_range': [0, 1],
                                                        'mean': [0.485, 0.456, 0.406],
                                                        'std': [0.229, 0.224, 0.225],
                                                        'num_classes': 1000,
                                                        'scale': 0.8975}},
                       'efficientnet-b4': {'imagenet': {'url': url_map['efficientnet-b4'],
                                                        'input_space': 'RGB',
                                                        'input_size': [3, 244, 244],
                                                        'input_range': [0, 1],
                                                        'mean': [0.485, 0.456, 0.406],
                                                        'std': [0.229, 0.224, 0.225],
                                                        'num_classes': 1000,
                                                        'scale': 0.8975}},
                       'efficientnet-b5': {'imagenet': {'url': url_map['efficientnet-b5'],
                                                        'input_space': 'RGB',
                                                        'input_size': [3, 244, 244],
                                                        'input_range': [0, 1],
                                                        'mean': [0.485, 0.456, 0.406],
                                                        'std': [0.229, 0.224, 0.225],
                                                        'num_classes': 1000,
                                                        'scale': 0.8975}},
                       'efficientnet-b6': {'imagenet': {'url': url_map['efficientnet-b6'],
                                                        'input_space': 'RGB',
                                                        'input_size': [3, 244, 244],
                                                        'input_range': [0, 1],
                                                        'mean': [0.485, 0.456, 0.406],
                                                        'std': [0.229, 0.224, 0.225],
                                                        'num_classes': 1000,
                                                        'scale': 0.8975}},
                       'efficientnet-b7': {'imagenet': {'url': url_map['efficientnet-b7'],
                                                        'input_space': 'RGB',
                                                        'input_size': [3, 244, 244],
                                                        'input_range': [0, 1],
                                                        'mean': [0.485, 0.456, 0.406],
                                                        'std': [0.229, 0.224, 0.225],
                                                        'num_classes': 1000,
                                                        'scale': 0.8975}}}

efficientnet_encoders = {'efficientnet-b0': {'encoder': EfficientNetEncoder,
                                             'out_shapes': [1280, 112, 40, 24, 16],
                                             'pretrained_settings': pretrained_settings['efficientnet-b0'],
                                             'params': {'model_name': 'efficientnet-b0'}},
                         'efficientnet-b1': {'encoder': EfficientNetEncoder,
                                             'out_shapes': [1280, 112, 40, 24, 16],
                                             'pretrained_settings': pretrained_settings['efficientnet-b1'],
                                             'params': {'model_name': 'efficientnet-b1'}},
                         'efficientnet-b2': {'encoder': EfficientNetEncoder,
                                             'out_shapes': [1408, 120, 48, 24, 16],
                                             'pretrained_settings': pretrained_settings['efficientnet-b2'],
                                             'params': {'model_name': 'efficientnet-b2'}},
                         'efficientnet-b3': {'encoder': EfficientNetEncoder,
                                             'out_shapes': [1536, 136, 48, 32, 24],
                                             'pretrained_settings': pretrained_settings['efficientnet-b3'],
                                             'params': {'model_name': 'efficientnet-b3'}},
                         'efficientnet-b4': {'encoder': EfficientNetEncoder,
                                             'out_shapes': [1792, 160, 56, 32, 24],
                                             'pretrained_settings': pretrained_settings['efficientnet-b4'],
                                             'params': {'model_name': 'efficientnet-b4'}},
                         'efficientnet-b5': {'encoder': EfficientNetEncoder,
                                             'out_shapes': [2048, 176, 64, 40, 24],
                                             'pretrained_settings': pretrained_settings['efficientnet-b5'],
                                             'params': {'model_name': 'efficientnet-b5'}},
                         'efficientnet-b6': {'encoder': EfficientNetEncoder,
                                             'out_shapes': [2304, 200, 72, 40, 32],
                                             'pretrained_settings': pretrained_settings['efficientnet-b6'],
                                             'params': {'model_name': 'efficientnet-b6'}},
                         'efficientnet-b7': {'encoder': EfficientNetEncoder,
                                             'out_shapes': [2560, 224, 80, 48, 32],
                                             'pretrained_settings': pretrained_settings['efficientnet-b7'],
                                             'params': {'model_name': 'efficientnet-b7'}}}
