#!pip install imgaug
#!pip install torch_snippets
from io import StringIO
import PIL
from matplotlib import pyplot as plt
from PIL import Image
from torch_snippets import show, Glob, np, choose
import array
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpim
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

ia.seed(1)
yes_checkbox_paths=[""]*2
no_checkbox_paths=[""]*2

root="C:\\Users\\Hp\\"
j=0
all_paths={}
for i in range(2):
    yes_checkbox_paths[i] = Glob(f"{root}/dataset1/dataset/YES/*")
    no_checkbox_paths[i] = Glob(f"{root}/dataset1/dataset/NO/*")
#yes1_checkbox_paths = Glob(f"{root}/dataset1/dataset/YES/*")
#no1_checkbox_paths = Glob(f"{root}/dataset1/dataset/NO/*")

    all_paths[i+j]=yes_checkbox_paths[i]
    all_paths[i+j+1]=no_checkbox_paths[i]
    
    
    j+=1

def patch(image,img1):
    image = image.copy()
    bbs = [(14,32), (82,32)]
    classes = []
    i=0
    for bb in bbs:
        clss = choose([0+i,1+i])
        i+=1
        i+=1
        fpath = choose(all_paths[clss])
        img = Image.open(fpath)
        image.paste(img, bb)
        classes.append(clss)
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

def augment(image, bbs):
    image = PIL.Image.fromarray(image)
    image, new_classes = patch(image,image)
    image = np.array(image)
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    H,W = image_aug.shape[:2]

    bbs_aug_yolo_format = []
    for i in range(len(bbs.bounding_boxes)):
        bb_ = bbs_aug.bounding_boxes[i]
        xmax, xmin = [func(bb_.x1, bb_.x2) for func in (max, min)]
        ymax, ymin = [func(bb_.y1, bb_.y2) for func in (max, min)]

        xc = (xmin + xmax)/2.0
        yc = (ymin + ymax)/2.0
        h = xmax-xmin
        w = ymax-ymin

        xc = xc/W; yc = yc/H
        h = h/H; w = w/W
        bbs_aug_yolo_format.append((xc,yc,w,h))
    return image_aug, bbs_aug_yolo_format, new_classes

def generate_image(IMAGE, BBS, filename):
    new_image, new_bbs, new_classes = augment(IMAGE, BBS)
    #print(new_classes,new_bbs)
    file = open("{}.txt".format(filename), 'w')
    
    #res=str(new_bbs).strip('[]')
    for i in range(2):
        res=""
        res+=str(new_classes[i])
        res+=" "
        for j in range(4):
            res+=str(new_bbs[i][j])
            res+=" "
        file.write(res)
        file.write('\n')
    file.close()
    #'''
    mybbs = [(xc-w/2,yc-h/2,xc+w/2,yc+h/2) for xc,yc,w,h in new_bbs]
    #show(new_image, bbs=mybbs)
    import scipy.misc
    scipy.misc.imsave('{}.png'.format(filename), new_image)


def generate_images(IMAGE,BBS,folder,n):
    for i in range(n):
        filename=folder+str(i)
        generate_image(IMAGE,BBS,filename)
