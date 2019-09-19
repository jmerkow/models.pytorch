from .classification import ClassificationModel
from ..encoders import get_preprocessing_fn


def get_model(base='', **model_params):
    base = base.lower()
    model_cls = ClassificationModel  ## right now only this one exists, so we can figure that out later
    model = model_cls(**model_params)

    encoder_weights = model_params['encoder_weights']
    preprocessing = None
    if encoder_weights is not None:
        preprocessing = get_preprocessing_fn(model_params['encoder'], pretrained=encoder_weights)

    return model, preprocessing
