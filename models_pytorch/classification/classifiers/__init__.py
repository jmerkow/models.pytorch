from .basic import BasicClassifier
from .srn import SpatialRegularizationClassifier

classifier_map = {
    'basic': BasicClassifier,
    'srn': SpatialRegularizationClassifier,
}


def get_classifier(classifier, **kwargs):
    Model = classifier_map[classifier]
    return Model(**kwargs)
