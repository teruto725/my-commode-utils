import subprocess
from os.path import dirname, join
from typing import Dict, List

from omegaconf import DictConfig


def download_dataset(url: str, dataset_dir: str, dataset_name: str):
    download_dir = dirname(dataset_dir)
    print(f"Downloading {dataset_name} from {url} to {dataset_dir}")
    download_command_result = subprocess.run(["wget", url, "-P", download_dir], capture_output=True, encoding="utf-8")
    if download_command_result.returncode != 0:
        raise RuntimeError(f"Failed to download dataset. Error:\n{download_command_result.stderr}")
    tar_name = join(download_dir, f"{dataset_name}.tar.gz")
    print(f"Extracting from {tar_name}")
    untar_command_result = subprocess.run(
        ["tar", "-xzvf", tar_name, "-C", download_dir], capture_output=True, encoding="utf-8"
    )
    if untar_command_result.returncode != 0:
        raise RuntimeError(f"Failed to untar dataset. Error:\n{untar_command_result.stderr}")


def print_table(data: Dict[str, List[str]]):
    row_lens = [max(len(header), max([len(s) for s in values])) for header, values in data.items()]
    row_template = " | ".join(["{:<" + str(i) + "}" for i in row_lens])
    headers = [key for key in data.keys()]
    max_data_per_col = max([len(v) for v in data.values()])
    row_data = []
    for i in range(max_data_per_col):
        row_data.append([v[i] if len(v) > i else "" for k, v in data.items()])

    header_line = row_template.format(*headers)
    delimiter_line = "-" * len(header_line)
    row_lines = [row_template.format(*row) for row in row_data]
    print("", header_line, delimiter_line, *row_lines, sep="\n")


def print_config(config: DictConfig, fields: List[str]):
    config_data = {}
    for column in fields:
        if column not in config:
            continue
        config_data[column] = [f"{k}: {v}" for k, v in config[column].items()]
    print_table(config_data)
