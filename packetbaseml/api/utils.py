from dotenv import load_dotenv
import os


def get_env_vars() -> dict:
    load_dotenv()
    return {"API_PORT": os.getenv("API_PORT")}
