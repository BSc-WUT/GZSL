from fastapi import FastAPI

from .utils import get_env_vars
from .models import NetworkFlow


app = FastAPI()
ENV_VARS = get_env_vars()


""" MODELS """


@app.get("/models")
def get_models_list():
    pass


@app.get("/models/{model_name}")
def get_model(model_name: str):
    pass


@app.post("/models/{model_name}/predict")
def model_predict(model_name: str, flow: NetworkFlow):
    pass


@app.post("/models/upload")
def upload_model():
    pass


@app.delete("models/{model_name}/delete")
def delete_model(model_name: str):
    pass

