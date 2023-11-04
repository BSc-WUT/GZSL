from dotenv import load_dotenv
from torchinfo import summary, ModelStatistics
import os
import re

from src.models.generic import GenericModel


def get_env_vars() -> dict:
    load_dotenv()
    return {"API_PORT": os.getenv("API_PORT")}


def get_model_summary(model: GenericModel, input_size: tuple) -> ModelStatistics:
    model_stats: ModelStatistics = summary(
        model,
        input_size,
        verbose=0,
        col_names=("input_size", "output_size", "num_params", "kernel_size"),
    )
    return model_stats


def split_summary_into_sections(summary: str) -> list:
    section: list = []
    for line in summary.split("\n")[3:]:
        if line.count("=") == len(line):
            yield section
            section = []
        else:
            section.append(line)


def parse_model_summary_to_json(model_summary: ModelStatistics) -> dict:
    model_summary_str: str = str(model_summary)
    parsed_summary: dict = {}
    sections: GeneratorExit = split_summary_into_sections(model_summary_str)
    model_layers: list = []
    for line in next(sections):
        parsed_line: str = re.sub(" +", " ", line).replace(", ", ",").replace(": ", ":")
        parsed_line_params: list = [
            param if param != "--" else 0 for param in parsed_line.split(" ")
        ]
        layer_name, input_shape, output_shape, params, kernel_shape = parsed_line_params
        model_layers.append(
            {
                "layer_name": layer_name,
                "input_shape": input_shape,
                "output_shape": output_shape,
                "params": int(params.replace(",", "")) if params else params,
                "kernel_size": kernel_shape if kernel_shape else "",
            }
        )
    parsed_summary["layers"] = model_layers

    for section in sections:
        for line in section:
            key, value = line.split(": ")
            parsed_value = 0
            if "," in value:
                parsed_value = int(value.replace(",", ""))
            elif "." in value:
                parsed_value = float(value)
            parsed_summary[
                key.split(" (")[0]
                .lower()
                .replace("size", "size_MB")
                .replace(" ", "_")
                .replace("/", "_")
                .replace("-", "_")
            ] = parsed_value

    return parsed_summary


def model_summary_to_json(model: GenericModel, input_size: tuple) -> dict:
    model_summary: str = get_model_summary(model, input_size)
    parsed_summary: dict = parse_model_summary_to_json(model_summary)
    return parsed_summary
