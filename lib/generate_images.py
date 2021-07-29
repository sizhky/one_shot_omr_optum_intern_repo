from functools import lru_cache
import PIL
from matplotlib import pyplot as plt
from PIL import Image
from torch_snippets import show, Glob, np, choose, makedir, parent, unique, stems
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from lxml.etree import Element, SubElement, tostring
from functools import lru_cache

from xml.dom.minidom import parseString
from lib.connectors.yolo import read_yolo, write_yolo

from torch_snippets.loader import readlines

ia.seed(1)

@lru_cache()
def load_checkbox_image_paths(folder):
    all_paths = {
        "ticked": Glob(f"{folder}/YES/*"),
        "not_ticked": Glob(f"{folder}/NO/*")
    }
    return all_paths

def patch(image, origins, checkbox_image_paths):
    checkbox_types = list(checkbox_image_paths.keys())
    image = image.copy()
    classes = []
    for origin in origins:
        clss = choose(checkbox_types)
        fpath = choose(checkbox_image_paths[clss])
        _chkbx = Image.open(fpath)
        image.paste(_chkbx, origin)
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

def augment(image, bbs, checkbox_image_paths):
    image = PIL.Image.fromarray(image) if isinstance(image, np.ndarray) else image
    origins = [(bb.x1, bb.y1) for bb in bbs]
    image, new_classes = patch(image, origins, checkbox_image_paths)
    image = np.array(image)
    image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)
    # show(image_aug, bbs=[[bb.x1, bb.y1, bb.x2, bb.y2] for bb in bbs_aug])
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

import scipy.misc
def generate_yolo_datum(IMAGE, BBS, filename, checkbox_image_paths):
    new_image, new_bbs, new_classes = augment(IMAGE, BBS, checkbox_image_paths)
    try:
        image_file_name ='{}.png'.format(filename) 
        makedir(parent(image_file_name))
        scipy.misc.imsave(image_file_name, new_image)
    except:
        import imageio
        imageio.imwrite('{}.png'.format(filename), new_image)

    with open("{}.txt".format(filename), 'w') as file:
        for cls, bbs in zip(new_classes, new_bbs):
            xc,yc,w,h = bbs
            row = f'{cls} {xc} {yc} {w} {h}\n'
            file.write(row)

def generate_yolo_data(template_image, template_bbs, folder, n, checkbox_folder):
    checkbox_image_paths = load_checkbox_image_paths(checkbox_folder)
    template_bbs = BoundingBoxesOnImage([
            BoundingBox(x1=x, y1=y, x2=X, y2=Y)
            for x,y,X,Y in template_bbs
        ], shape=np.array(template_image).shape
    )
    for i in range(n):
        filename = f'{folder}/{i}'
        generate_yolo_datum(template_image, template_bbs, filename, checkbox_image_paths)

def inspect_yolo_data(folder):
    files = unique(stems(folder))
    for f in files:
        im_path = f'{folder}/{f}.png'
        clss, bbs = read_yolo(f'{folder}/{f}.txt', im_path)
        show(im_path, texts=clss, bbs=bbs)

def patch_icevision(image,img1):
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

def augment_icevision(image,pts,bbs):
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

        bbs_aug_yolo_format.append((xmin,ymin,xmax,ymax))
    return image_aug, bbs_aug_yolo_format, new_classes

def generate_image_icevision(IMAGE, BBS, filename):
    IMAGE=PIL.Image.fromarray(IMAGE)
    for _ in range(2):
        bbs = [(14,32), (82,32)]
        a,b=patch_icevision(IMAGE,IMAGE)
        new_image, new_bbs, new_classes = augment_icevision(np.array(a),bbs,BBS)
        #print(new_classes)


        file = open("{}.txt".format(filename), 'w')
        sum=[]
        for i in range(len(new_bbs)):
            tup=()
            for j in range(len(new_bbs[i])):
                if(new_bbs[i][j]<0):
                    tup=tup+(0,)
                else:
                    tup=tup+(int(new_bbs[i][j]),)
            sum.append(tup)

        #print(sum)
    
    
        node_root = Element('annotation')
 
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'images'
 
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = f"{1+_}.png"

        node_filename = SubElement(node_root, 'path')
        node_filename.text = f"../images/{1+_}.png"

    
        node_source = SubElement(node_root, 'source')
        node_source = SubElement(node_source, 'database')
        node_source.text = 'Unknown'

        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = '499'
 
        node_height = SubElement(node_size, 'height')
        node_height.text = '666'
 
        node_depth = SubElement(node_size, 'depth')
        node_depth.text = '3'

        node_segmented = SubElement(node_root, 'segmented')
        node_segmented.text = '0'
        for i in range(2):
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            if(new_classes[i]==1):
                node_name.text = 'NO'
            elif(new_classes[i]==0):
                node_name.text = 'YES'
            node_pose = SubElement(node_object, 'pose')
            node_pose.text = 'Unspecified'
            node_truncated = SubElement(node_object, 'truncated')
            node_truncated.text = '0'
            node_difficult = SubElement(node_object, 'difficult')
            node_difficult.text = '0'
            node_bndbox = SubElement(node_object, 'bndbox')
            node_xmin = SubElement(node_bndbox, 'xmin')
            node_xmin.text = str(sum[i][0]).encode()
            node_ymin = SubElement(node_bndbox, 'ymin')
            node_ymin.text = str(sum[i][1]).encode()
            node_xmax = SubElement(node_bndbox, 'xmax')
            node_xmax.text = str(sum[i][2]).encode()
            node_ymax = SubElement(node_bndbox, 'ymax')
            node_ymax.text = str(sum[i][3]).encode()

            #print(sum[i][0])
            Xml = tostring(node_root, pretty_print=True) #Formatted display, the newline of the newline
            dom = parseString(Xml)
        #print(Xml)
        f = open('{}.xml'.format(filename), 'wb')
        f.write(Xml)
        f.close()
    
    #res=str(new_bbs).strip('[]')
        for i in range(2):
            res=""
            res+=str(new_classes[i])
            res+=" "
            for j in range(4):
                res+=str(sum[i][j])
                res+=" "
            file.write(res)
            file.write('\n')
        file.close()
    #mybbs = [(xc-w/2,yc-h/2,xc+w/2,yc+h/2) for xc,yc,w,h in new_bbs]
        #show(new_image)
        import scipy.misc
        scipy.misc.imsave('{}.png'.format(filename), new_image)

def generate_images_icevision(IMAGE,BBS,folder,n):
    for i in range(n):
        filename=folder+str(i)
        generate_image_icevision(IMAGE,BBS,filename)
