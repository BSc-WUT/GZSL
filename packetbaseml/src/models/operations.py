import torch
import os
from src.vars import MODELS_PATH
from src.models.generic import GenericModel


def save_model(model_name: str, model: GenericModel) -> None:
    model_path: str = os.path.join(MODELS_PATH, model_name)
    torch.save(model, model_path)


def load_model(model_name: str) -> GenericModel:
    model_path: str = os.path.join(MODELS_PATH, model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError()
    model: GenericModel = torch.load(model_path)
    return model


"""

        
        
        f1 = 2 * precision * sensitivity / (precision + sensitivity)
"""


def precision(true_predictions: int, false_positive: int) -> float:
    try:
        precision: float = (
            100.0 * true_predictions / (true_predictions + false_positive)
        )
    except:
        precision = 0.0
    return precision


def accuracy(true_predictions: int, predicitons_amount: int) -> float:
    try:
        accuracy: float = 100.0 * true_predictions / predicitons_amount
    except:
        accuracy = 0.0
    return accuracy


def sensitivity(true_predictions: int, false_negative: int) -> float:
    try:
        sensitivity: float = (
            100.0 * true_predictions / (true_predictions + false_negative)
        )
    except:
        sensitivity = 0.0
    return sensitivity


def f1(precision: float, sensitivity: float) -> float:
    try:
        f1: float = 2 * precision * sensitivity / (precision + sensitivity)
    except:
        f1 = 0.0
    return f1
