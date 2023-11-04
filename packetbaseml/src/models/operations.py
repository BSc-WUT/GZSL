import torch
import os
from src.vars import MODELS_PATH
from src.models.generic import GenericModel


def save_model(model_name: str, model: GenericModel) -> None:
    model_path: str = os.path.join(MODELS_PATH, model_name)
    torch.save(model, model_path)


def load_model(model_name: str) -> GenericModel:
    model_path: str = os.path.join(MODELS_PATH, model_name)
    model: GenericModel = torch.load(model_path)
    return model
