from pretrainedmodels.models.inceptionv4 import InceptionV4, pretrained_settings


class InceptionV4Encoder(InceptionV4):
    output_indices = [2, 4, 9, 17]

    def __init__(self):
        super().__init__()
        del self.last_linear

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('last_linear.bias')
        state_dict.pop('last_linear.weight')
        super().load_state_dict(state_dict, **kwargs)

    def forward(self, x):

        features = []
        for ix, l in enumerate(self.features[:-1]):
            x = l(x)
            if ix in self.output_indices:
                features.insert(0, x)
        features.insert(0, x)
        return features


inceptionv4_encoders = {
    'inceptionv4': {
        'encoder': InceptionV4Encoder,
        'out_shapes': (1536, 1024, 384, 192, 64),
        'pretrained_settings': pretrained_settings['inceptionv4'],
        'params': {},
    }
}
