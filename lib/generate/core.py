"""Function heirarchy

generate_data
    -> load_checkbox_image_paths
    -> generate_datum
        -> augment
            -> patch
inspect_data
"""
__all__ = ['generate_data', 'inspect_records']
from functools import lru_cache
import PIL
from PIL import Image
from matplotlib import pyplot as plt
from torch_snippets import show, Glob, np, choose, read, resize, bbfy, lzip
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


from copy import deepcopy
def augment(image, bbs, class_names, seq, checkbox_image_paths, tolerance=15):
    image = PIL.Image.fromarray(image).copy() if isinstance(image, np.ndarray) else image.copy()
    image, new_classes = patch(image, bbs, class_names, checkbox_image_paths)
    image = np.array(image)
    image_aug, bbs_aug = seq(
        image=image, bounding_boxes=BoundingBoxesOnImage([
            BoundingBox(x1=x, y1=y, x2=X, y2=Y)
            for x,y,X,Y in bbs
        ], shape=image.shape)
    )
    h, w = image_aug.shape[:2]
    bbs = bbfy([(bb.x1, bb.y1, bb.x2, bb.y2) for bb in bbs_aug])
    BBS, CLSS = deepcopy(bbs), deepcopy(new_classes)
    to_remove = []
    for ix, (cls, bb) in enumerate(zip(new_classes, bbs)): 
        x,y,X,Y = bb
        if X <= tolerance or Y <= tolerance or x >= w-tolerance or y >= h-tolerance:
            # print(f'removing: {bb} {cls}')
            BBS.remove(bb)
            CLSS.remove(cls)
    return image_aug, BBS, CLSS
 
def generate_datum(IMAGE, BBS, class_names, seq, checkbox_image_paths):
    new_image, new_bbs, new_classes = augment(IMAGE, BBS, class_names, seq, checkbox_image_paths)
    return new_image, new_bbs, new_classes

def generate_data(template_image, template_bbs, class_names, n, seq, checkbox_folder):
    checkbox_image_paths = load_checkbox_image_paths(checkbox_folder)
    records = []
    for _ in range(n):
        records.append(generate_datum(template_image.copy(), template_bbs, class_names, seq, checkbox_image_paths))
    return records

def inspect_records(records):
    for record in records:
        im, bbs, clss = record
        show(im, bbs=bbs, texts=clss)
