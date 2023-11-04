from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from typing import List
import torch
import os

from src.vars import MODELS_PATH, INPUT_SIZE
from src.models.generic import GenericModel
from .utils import get_env_vars, model_summary_to_json
from .models import NetworkFlow, Model


app = FastAPI()
ENV_VARS = get_env_vars()


""" MODELS """


@app.get("/models")
def get_models_list() -> List[Model]:
    models_desc: list = []
    for model_path in os.listdir(MODELS_PATH):
        model: GenericModel = torch.load(os.path.join(MODELS_PATH, model_path))
        model_desc: Model = model_summary_to_json(model, INPUT_SIZE)
        models_desc.append(model_desc)
    return models_desc


@app.get("/models/{model_name}")
def get_model(model_name: str) -> JSONResponse:
    try:
        model: GenericModel = torch.load(os.path.join(MODELS_PATH, model_name))
        model_desc: Model = model_summary_to_json(model, INPUT_SIZE)
        return model_desc
    except:
        return {"error": f"Model: {model_name} was not found"}


@app.post("/models/{model_name}/predict")
def model_predict(model_name: str, flow: NetworkFlow) -> JSONResponse:
    try:
        model: GenericModel = torch.load(os.path.join(MODELS_PATH, model_name))
    except:
        return {"error": f"Model: {model_name} was not found"}

    input: torch.Tensor = torch.Tensor([flow.dict()])
    prediction: dict = model(input)
    return {"prediction": prediction}


@app.post("/models/upload")
async def upload_model(model_file: UploadFile, model_name: str) -> JSONResponse:
    with open(os.join(MODELS_PATH, model_name), "+bw") as file_handler:
        file_handler.write(model_file.read())
    return {"result": f"Successfully uploaded file with model: {model_name}"}


@app.delete("models/{model_name}/delete")
def delete_model(model_name: str) -> JSONResponse:
    file_path: str = os.path.join(MODELS_PATH, model_name)
    if os.path.exists(file_path):
        os.remove(path=file_path)
        return {"result": f"Successfully deleted file with model: {model_name}"}
    else:
        return {"result": f"Could not delete file {model_name}. File does not exists."}
