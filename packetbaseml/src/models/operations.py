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
    precision: float = 100.0 * true_predictions / (true_predictions + false_positive)
    return precision


def accuracy(true_predictions: int, predicitons_amount: int) -> float:
    accuracy: float = 100.0 * true_predictions / predicitons_amount
    return accuracy


def sensitivity(true_predictions: int, false_negative: int) -> float:
    sensitivity: float = 100.0 * true_predictions / (true_predictions + false_negative)
    return sensitivity


def f1(precision: float, sensitivity: float) -> float:
    f1: float = 2 * precision * sensitivity / (precision + sensitivity)
    return f1
