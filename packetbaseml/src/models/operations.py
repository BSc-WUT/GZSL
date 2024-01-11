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


def precision(true_predictions: int, false_positive: int) -> float:
    try:
        prec: float = 100.0 * true_predictions / (true_predictions + false_positive)
    except:
        prec = 0.0
    return prec


def accuracy(true_predictions: int, predicitons_amount: int) -> float:
    try:
        acc: float = 100.0 * true_predictions / predicitons_amount
    except:
        acc = 0.0
    return acc


def sensitivity(true_predictions: int, false_negative: int) -> float:
    try:
        sens: float = 100.0 * true_predictions / (true_predictions + false_negative)
    except:
        sens = 0.0
    return sens


def f1(prec: float, sens: float) -> float:
    try:
        f1: float = 2 * prec * sens / (prec + sens)
    except:
        f1 = 0.0
    return f1
