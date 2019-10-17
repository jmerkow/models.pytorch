from .basic import BasicClassifier
from .conv_lstm import ConvLSTMClassifier
from .srn import SpatialRegularizationClassifier

classifier_map = {
    'basic': BasicClassifier,
    'srn': SpatialRegularizationClassifier,
    'conv_lstm': ConvLSTMClassifier,
}


def get_classifier(classifier, **kwargs):
    Model = classifier_map[classifier]
    return Model(**kwargs)
