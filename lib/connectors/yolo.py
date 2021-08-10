__all__ = ['generate_data_in_yolo_format',  'inspect_yolo_data']
from lib.generate.core import *
from torch_snippets import flatten, unique, writelines, cv2, makedir, readlines, P, read, stems, show
import PIL
import numpy as np

def save_record_in_yolo_format(record, filename, mapping):
    im, bbs, clss = record
    clss = [mapping[cls] for cls in clss]
    H,W = im.shape[:2]
    lines = []
    for bb, cls in zip(bbs, clss):
        xmax, xmin = [func(bb.x, bb.X) for func in (max, min)]
        ymax, ymin = [func(bb.y, bb.Y) for func in (max, min)]
        xc = (xmin + xmax)/2.0
        yc = (ymin + ymax)/2.0
        w = xmax-xmin
        h = ymax-ymin
        xc = xc/W; yc = yc/H
        h = h/H; w = w/W
        lines.append('{} {} {} {} {}'.format(cls,xc,yc,w,h))
    writelines(lines, filename+'.txt')
    cv2.imwrite(filename+'.png', im)

def generate_data_in_yolo_format(template_image, template_bbs, class_names, n, checkbox_folder, output_folder):
    makedir(output_folder)
    makedir(f'{output_folder}/data')
    records = generate_data(template_image, template_bbs, class_names, n, checkbox_folder)
    mapping = unique(flatten([clss for im,bbs,clss in records]))
    writelines(mapping, f'{output_folder}/obj.names')
    mapping = {cls:ix for ix,cls in enumerate(mapping)}
    writelines([
        f'classes = {len(mapping)}',
        'train = data/train.txt',
        'names = data/obj.names',
        'backup = backup/'
    ], f'{output_folder}/obj.data')

    for ix, record in enumerate(records):
        filename = f'{output_folder}/data/{ix}'
        save_record_in_yolo_format(record, filename, mapping)

def read_yolo(text_file, image=None):
    if isinstance(image, (str, P)):
        image = read(str(image))
        H, W = image.shape
    elif isinstance(image, (np.ndarray, PIL.Image.Image)):
        H, W = np.array(image).shape

    lines = readlines(text_file, silent=True)
    lines = [l.split() for l in lines]
    classes = [clss for clss, *_ in lines]
    bbs = [[float(i) for i in [xc,yc,w,h]] for _,xc,yc,w,h in lines]
    bbs = [(xc-w/2, yc-h/2, xc+w/2, yc+h/2) for xc,yc,w,h in bbs]
    bbs = [(x*W,y*H,X*W,Y*H) for x,y,X,Y in bbs]
    bbs = [[int(i) for i in bb] for bb in bbs]
    return classes, bbs

def inspect_yolo_data(folder):
    folder = P(folder)
    mapping = readlines(folder/'obj.names')
    for f in unique(stems(folder/'data')):
        im = read(f'{folder}/data/{f}.png')
        txt = f'{folder}/data/{f}.txt'
        clss, bbs = read_yolo(txt, im)
        clss = [mapping[int(c)] for c in clss]
        show(im, bbs=bbs, texts=clss, sz=10)