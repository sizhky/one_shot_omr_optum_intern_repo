__all__ = ['generate_pascal_data']
from torch_snippets.paths import makedir, parent, fname
from lib.generate.core import *
import cv2
from lxml.etree import Element, SubElement, tostring

def generate_pascal_datum(image, bbs, classes, image_path, xml_path):
    h, w = image.shape[:2]

    for ix in range(len(bbs)):
        sum=[]
        for i in range(len(bbs)):
            tup=()
            for j in range(len(bbs[i])):
                if(bbs[i][j]<0):
                    tup=tup+(0,)
                else:
                    tup=tup+(int(bbs[i][j]),)
            sum.append(tup)

        node_root = Element('annotation')
 
        node_folder = SubElement(node_root, 'folder')
        node_folder.text = 'images'
 
        node_filename = SubElement(node_root, 'filename')
        node_filename.text = f"{fname(image_path)}"

        node_filename = SubElement(node_root, 'path')
        node_filename.text = f"{image_path}"

        node_source = SubElement(node_root, 'source')
        node_source = SubElement(node_source, 'database')
        node_source.text = 'Unknown'

        node_size = SubElement(node_root, 'size')
        node_width = SubElement(node_size, 'width')
        node_width.text = str(w)
 
        node_height = SubElement(node_size, 'height')
        node_height.text = str(h)
 
        node_depth = SubElement(node_size, 'depth')
        node_depth.text = '3'

        node_segmented = SubElement(node_root, 'segmented')
        node_segmented.text = '0'

        for i in range(len(bbs)):
            node_object = SubElement(node_root, 'object')
            node_name = SubElement(node_object, 'name')
            node_name.text = classes[i]
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
            Xml = tostring(node_root, pretty_print=True) #Formatted display, the newline of the newline

        makedir(parent(xml_path))
        makedir(parent(image_path))
        f = open(xml_path, 'wb')
        f.write(Xml)
        f.close()

        cv2.imwrite(image_path, image)

def generate_data_in_pascal_format(template_image, template_bbs, class_names, n,seq, checkbox_folder, output_folder):
    records = generate_data(template_image, template_bbs, class_names, n,seq, checkbox_folder)
    for ix, record in enumerate(records):
        im, bbs, clss = record
        image_path = f'{output_folder}/images/{ix}.png'
        xml_path = f'{output_folder}/annotations/{ix}.xml'
        generate_pascal_datum(im, bbs, clss, image_path, xml_path)

"""To inspect pascal data

```python
from icevision.all import *
data_dir = Path(output_folder)
images_dir = data_dir / 'images'
annotations_dir = data_dir / 'annotations'
class_map = flatten([[f'{i}_ticked', f'{i}_not_ticked'] for i in range(1, 22)])
class_map = ClassMap(class_map)
print(class_map)

parser = parsers.voc(annotations_dir=annotations_dir, images_dir=images_dir, class_map=class_map)
data_splitter = RandomSplitter((.8, .2))
# data_splitter
train_records, valid_records = parser.parse(data_splitter)
show_records(train_records[0:2], ncols=1, class_map=class_map)
```

"""
