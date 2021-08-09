from re import template
from fastcore.basics import annotations
import typer
import PIL
from pathlib import Path
from lib.connectors import *
from torch_snippets import readlines
from lib.connectors.yolo import read_yolo

CONNECTORS = {
    'yolo': generate_data_in_yolo_format,
    'pascal': generate_data_in_pascal_format
}

app = typer.Typer()

@app.command()
def generate(
    template_image_path: Path, 
    annotations_path: Path,
    output_folder: Path, 
    output_format: str, 
    n_images: int,
    checkboxes_folder: Path
):
    template_image = PIL.Image.open(template_image_path)
    lines = readlines(annotations_path)
    lines = [l.split() for l in lines]
    CLSS = [int(l) for l,*_ in lines]
    BBS = [[int(pt) for pt in pts] for _,*pts in lines]

    if output_format not in CONNECTORS:
        raise NotImplementedError('Other formats are not yet supported')
    else:
        CONNECTORS[output_format](template_image, BBS, CLSS, n_images, checkboxes_folder, output_folder)

@app.command()
def train(
    training_data_folder
):
    pass

if __name__ == "__main__":
    app()