"""Function heirarchy

generate_data
    -> load_checkbox_image_paths
    -> generate_datum
        -> augment
            -> patch
inspect_data
"""
__all__ = ['generate_data']
from functools import lru_cache
import PIL
from PIL import Image
from matplotlib import pyplot as plt
from torch_snippets import show, Glob, np, choose, read, resize, bbfy, BB
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from functools import lru_cache

ia.seed(1)

@lru_cache()
def load_checkbox_image_paths(folder):
    all_paths = {
        "ticked": Glob(f"{folder}/YES/*"),
        "not_ticked": Glob(f"{folder}/NO/*")
    }
    return all_paths

def patch(image, bbs, class_names, checkbox_image_paths):
    checkbox_types = list(checkbox_image_paths.keys())
    bbs = bbfy(bbs)
    image = image.copy()
    classes = []
    for class_name, bb in zip(class_names, bbs):
        clss = choose(checkbox_types)
        fpath = choose(checkbox_image_paths[clss])
        _chkbx = read(fpath)
        _chkbx = resize(_chkbx, (bb.w, bb.h))
        _chkbx = Image.fromarray(_chkbx)
        origin = (bb.x, bb.y)
        image.paste(_chkbx, origin)
        classes.append(f'{class_name}_{clss}')
    return image, classes

seq = iaa.Sequential([
    iaa.Crop(px=(0, 16)), 
    #iaa.Fliplr(0.5), 
    iaa.GaussianBlur(sigma=(0, 3.0)), 
    iaa.Multiply((1.2, 1.5)), # change brightness, doesn't affect BBs
    iaa.Affine(
        translate_px=(-15, 15),
        scale=(0.75, 1.25),
        cval=(255,255),
        #rotate=(-10,10),
        fit_output=True
    ),
])

def augment(image, bbs, class_names, checkbox_image_paths):
    image = PIL.Image.fromarray(image) if isinstance(image, np.ndarray) else image
    image, new_classes = patch(image, bbs, class_names, checkbox_image_paths)
    image = np.array(image)
    image_aug, bbs_aug = seq(
        image=image, bounding_boxes=BoundingBoxesOnImage([
            BoundingBox(x1=x, y1=y, x2=X, y2=Y)
            for x,y,X,Y in bbs
        ], shape=image.shape)
    )
    return image_aug, bbfy([(bb.x1, bb.y1, bb.x2, bb.y2) for bb in bbs_aug]), new_classes
    
def generate_datum(IMAGE, BBS, class_names, checkbox_image_paths):
    new_image, new_bbs, new_classes = augment(IMAGE, BBS, class_names, checkbox_image_paths)
    return new_image, new_bbs, new_classes

def generate_data(template_image, template_bbs, class_names, n, checkbox_folder):
    checkbox_image_paths = load_checkbox_image_paths(checkbox_folder)
    records = []
    for i in range(n):
        records.append(generate_datum(template_image, template_bbs, class_names, checkbox_image_paths))
    return records

def inspect_records(records):
    for record in records:
        im, bbs, clss = record
        show(im, bbs=bbs, texts=clss)
