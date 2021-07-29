from re import template
import typer
import PIL
from pathlib import Path
from lib.generate_images import generate_yolo_data
from torch_snippets import show
from lib.connectors.yolo import read_yolo

app = typer.Typer()

@app.command()
def generate(
    template_folder: Path, 
    output_folder: Path, 
    output_format:str, 
    n_images:int,
    checkboxes_folder: Path
):
    template_image = PIL.Image.open(template_folder/'image.jpg')
    W, H = template_image.size
    if output_format == 'yolo':
        classes, bbs = read_yolo(template_folder/'annotations.txt', image=template_folder/'image.jpg')
    else:
        raise NotImplementedError('Other formats are not yet supported')
    show(template_image, bbs=bbs)
    generate_yolo_data(template_image, bbs, output_folder, n_images, checkboxes_folder)
    print(
        template_folder,
        output_folder,
        output_format,
        n_images
    )

@app.command()
def train(
    training_data_folder
):
    pass

if __name__ == "__main__":
    app()