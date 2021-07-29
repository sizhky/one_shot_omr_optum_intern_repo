from torch_snippets import unzip_file, P
import os

def create_project_template_from_yolo_cvat(zip_file, project_folder):
    project = P(project_folder)
    unzip_file(zip_file, project)

    image_extns = 'jpg,png,jpeg'.split(',')
    imgs = list((project/'obj_train_data').glob('*'))
    img = [f for f in imgs if any([f.suffix == f'.{ext}' for ext in image_extns])][0]
    txt = next((project/'obj_train_data').glob('*.txt'))

    (project/'template').mkdir(exist_ok=True)
    os.rename(img, project/'template/image.jpg')
    os.rename(txt, project/'template/annotations.txt')

    (project/'obj_train_data').rmdir()
    os.remove(project/'obj.data')
    os.remove(project/'obj.names')
    os.remove(project/'train.txt')