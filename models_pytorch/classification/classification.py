import torch

from .classifiers import get_classifier
from ..base import Model
from ..encoders import get_encoder


class ClassificationModel(Model):

    def __init__(self,
                 encoder='resnet34',
                 encoder_weights="imagenet",
                 model_dir=None,

                 classifier='basic',
                 nclasses=6,
                 activation='sigmoid',
                 **kwargs):
        super().__init__()

        self.name = encoder
        self.encoder = get_encoder(encoder, encoder_weights=encoder_weights, model_dir=model_dir)
        self.classifier = get_classifier(classifier, encoder_channels=self.encoder.out_shapes,
                                         nclasses=nclasses, activation=activation,
                                         **kwargs)

        self.is_multi_task = self.classifier.is_multi_task

    def output_info(self):
        return self.classifier.output_info()

    @property
    def tasks(self):
        return self.classifier.tasks

    def forward(self, x, **args):
        """Sequentially pass `x` trough model`s `encoder` and `decoder` (return logits!)"""
        features = self.encoder(x)
        output = self.classifier(features)
        return output

    def predict(self, x, **args):

        if self.training:
            self.eval()
        with torch.no_grad():
            features = self.encoder(x, **args)
            output = self.classifier.predict(features)
            return output

    def get_encoder_params(self):
        return self.encoder.parameters()

    def get_classifier_params(self):
        return self.classifier.parameters()
