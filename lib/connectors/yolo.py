from torch_snippets import readlines, P, read
import PIL
import numpy as np

def read_yolo(text_file, image=None):
    if isinstance(image, (str, P)):
        image = read(str(image))
        H, W = image.shape
    elif isinstance(image, (np.ndarray, PIL.Image.Image)):
        H, W = np.array(image).shape

    lines = readlines(text_file)
    lines = [l.split() for l in lines]
    classes = [clss for clss, *_ in lines]
    bbs = [[float(i) for i in [xc,yc,w,h]] for _,xc,yc,w,h in lines]
    bbs = [(xc-w/2, yc-h/2, xc+w/2, yc+h/2) for xc,yc,w,h in bbs]
    bbs = [(x*W,y*H,X*W,Y*H) for x,y,X,Y in bbs]
    bbs = [[int(i) for i in bb] for bb in bbs]
    return classes, bbs


def write_yolo(classes, bbs, text_file):
    with open(text_file, 'w') as file:
        for cls, bbs in zip(classes, bbs):
            xc,yc,w,h = bbs
            row = f'{cls} {xc} {yc} {w} {h}\n'
            file.write(row)