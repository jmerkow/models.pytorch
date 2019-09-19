import six
import torch
from torch import nn as nn

from .classifiers import get_classifier
from ..base import Model
from ..encoders import get_encoder


class ClassificationModel(Model):

    def __init__(self, encoder='resnet34', activation='sigmoid',
                 encoder_weights="imagenet", nclasses=1,
                 tasks='cls',
                 classifier_params=None, model_dir=None,
                 **kwargs):
        super().__init__()

        self.name = encoder
        self.encoder = get_encoder(encoder, encoder_weights=encoder_weights, model_dir=model_dir)

        if isinstance(tasks, six.string_types):
            tasks = [tasks]

        classifier_params = classifier_params or {'type': 'basic'}
        task_params = {}

        for task in tasks:
            task_params[task] = classifier_params.pop(task, {})

        for d in task_params.values():
            d.setdefault('activation', activation)
            d.setdefault('nclasses', nclasses)
            for k, v in classifier_params.items():
                d.setdefault(k, v)
            d['encoder_channels'] = self.encoder.out_shapes
            d.update(**kwargs)
        self.classifiers = nn.ModuleDict({name: get_classifier(**args) for name, args in task_params.items()})

    def output_info(self):
        return {name: {'nclasses': c.nclasses, 'activation': c.activation_type} for name, c in self.classifiers.items()}

    @property
    def tasks(self):
        return list(self.output_info().keys())

    def forward(self, x, **args):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
        features = self.encoder(x)
        outputs = {name: classifier.forward(features) for name, classifier in self.classifiers.items()}
        return outputs

    def predict(self, x, **args):

        if self.training:
            self.eval()
        with torch.no_grad():
            features = self.encoder(x, **args)
            return {name: classifier.predict(features) for name, classifier in self.classifiers.items()}

    def get_encoder_params(self):
        return self.encoder.parameters()

    def get_classifier_params(self):
        return self.classifiers.parameters()
