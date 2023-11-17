from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import torch
import os

from src.vars import MODELS_PATH, INPUT_SIZE
from src.models.generic import GenericModel
from src.models.operations import load_model
from .utils import get_env_vars, model_summary_to_json
from .file_metadata import set_is_active_flag, get_is_active_flag
from .models import NetworkFlow, Model


app = FastAPI()
ENV_VARS = get_env_vars()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def set_model_flag(model_name: str, flag_value: bool) -> dict:
    model_path: str = os.path.join(MODELS_PATH, f"{model_name}.pt")
    set_is_active_flag(model_path, flag_value)
    return {
        'result': f'Sucessfully set is_active flag to {flag_value} for model: {model_name}'
    }


""" MODELS """


@app.get("/models")  # ✅
def get_models_list() -> List[Model]:
    models_desc: list = []
    for file_name in os.listdir(MODELS_PATH):
        if not "__init__" in file_name:
            model: GenericModel = load_model(file_name)
            model_desc: dict = model_summary_to_json(model, INPUT_SIZE)
            model_desc["name"] = file_name.split(".")[0]
            model_path: str = os.path.join(MODELS_PATH, file_name)
            model_desc['is_active'] = get_is_active_flag(model_path)
            models_desc.append(model_desc)
    return models_desc



@app.get('/models/activate/{model_name}')
def activate_model(model_name: str) -> JSONResponse:
    return set_model_flag(model_name, True)


@app.get('/models/deactivate/{model_name}')
def activate_model(model_name: str) -> JSONResponse:
    return set_model_flag(model_name, False)


@app.get("/models/{model_name}")  # ✅
def get_model(model_name: str) -> JSONResponse:
    try:
        model_path: str = os.path.join(MODELS_PATH, f"{model_name}.pt")
        model: GenericModel = torch.load(model_path)
        model_desc: dict = model_summary_to_json(model, INPUT_SIZE)
        model_desc["name"] = model_name.split(".")[0]
        model_desc['is_active'] = get_is_active_flag(model_path)
        return model_desc
    except:
        return {"error": f"Model: {model_name} was not found"}


@app.post("/models/predict/{model_name}")
def model_predict(model_name: str, flow: NetworkFlow) -> JSONResponse:
    model_file_name: str = f"{model_name}.pt"
    model: GenericModel = load_model(model_file_name)
    input: list = [float(value) for value in flow.dict().values()]
    input_tensor: torch.Tensor = torch.Tensor(input)
    prediction: dict = model(input_tensor)
    return {"prediction": prediction}


@app.post("/models/upload/{model_name}")  # ✅
async def upload_model(model_file: UploadFile, model_name: str) -> JSONResponse:
    with open(os.path.join(MODELS_PATH, f"{model_name}.pt"), "+bw") as file_handler:
        file_handler.write(model_file.file.read())
    return {"result": f"Successfully uploaded file with model: {model_name}"}


@app.delete("/models/delete/{model_name}")  # ✅
def delete_model(model_name: str) -> JSONResponse:
    file_path: str = os.path.join(MODELS_PATH, f"{model_name}.pt")
    if os.path.exists(file_path):
        os.remove(path=file_path)
        return {"result": f"Successfully deleted file with model: {model_name}"}
    else:
        return {"result": f"Could not delete file {model_name}. File does not exists."}
