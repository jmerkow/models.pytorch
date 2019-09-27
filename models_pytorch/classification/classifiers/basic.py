import six
from torch import nn as nn

from ._classifiers import get_classifier


class BasicClassifier(nn.Module):

    def __init__(self, encoder_channels, tasks='cls', classifier_params=None, activation='sigmoid', nclasses=6,
                 **kwargs):
        super().__init__()
        classifier_params = classifier_params or {'type': 'basic'}
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
            d['encoder_channels'] = encoder_channels
            d.update(**kwargs)
        self.classifiers = nn.ModuleDict({name: get_classifier(**args) for name, args in task_params.items()})
        self.is_multi_task = len(self.classifiers) > 1

    def output_info(self):
        return {name: {'nclasses': c.nclasses, 'activation': c.activation_type} for name, c in self.classifiers.items()}

    @property
    def tasks(self):
        return list(self.output_info().keys())

    def forward(self, features, **kwargs):
        output = [(name, classifier(features)) for name, classifier in self.classifiers.items()]
        if not self.is_multi_task:
            output = output[0][1]
        else:
            output = dict(output)

        return output

    def predict(self, features, **kwargs):
        output = [(name, classifier.predict(features)) for name, classifier in self.classifiers.items()]

        if not self.is_multi_task:
            output = output[0][1]
        else:
            output = dict(output)

        return output
