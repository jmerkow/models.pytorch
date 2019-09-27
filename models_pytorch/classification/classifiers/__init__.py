from .basic import BasicClassifier

classifier_map = {
    'basic': BasicClassifier
}


def get_classifier(classifier, **kwargs):
    Model = classifier_map[classifier]
    return Model(**kwargs)
