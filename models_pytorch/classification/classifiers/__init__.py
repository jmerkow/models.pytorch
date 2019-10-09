from .basic import BasicClassifier
from .extra_inputs import ExtraScalarInputsClassifier
from .srn import SpatialRegularizationClassifier

classifier_map = {
    'basic': BasicClassifier,
    'srn': SpatialRegularizationClassifier,
    'extra_inputs': ExtraScalarInputsClassifier,
}


def get_classifier(classifier, **kwargs):
    Model = classifier_map[classifier]
    return Model(**kwargs)
