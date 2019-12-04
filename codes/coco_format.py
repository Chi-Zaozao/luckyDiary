import os
from lxml import etree
from detectron2.structures import BoxMode
from datetime import datetime
import json

def get_simimage_dicts(img_dir):
    '''
    Convert the dataset annotations into a format which detectron2 can accept and register
    Args:
        img_dir(string): the path of the dataset which stores the pictures and annitations, e.g. 'simImage/val/'
    Returns:
        list[dict]: in the required format as declared in https://detectron2.readthedocs.io/tutorials/datasets.html
    '''
    xml_files=os.listdir(os.path.join(img_dir,'Annotations'))
    dataset_dicts=[]
    for file in xml_files:
        if file == '.ipynb_checkpoints':
            continue
        record={}
        tree=etree.parse(os.path.join(img_dir,'Annotations',file))
        root=tree.getroot()
        tags=[]
        texts=[]
        bboxes=[]
        for c in root.iter():
            tags.append(c.tag)
            texts.append(c.text)
            if c.tag=='bndbox':
                bboxes.append(c)
        filename = texts[tags.index('filename')]
        record['file_name']=os.path.join(img_dir,filename)
        record['width']=int(texts[tags.index('width')])
        record['height']=int(texts[tags.index('height')])
        record['image_id']=int(filename[:-4].replace('-',''))
        objs=[]
        for bbox in bboxes:
            box=[]
            for b in bbox.iterchildren():
                box.append(int(b.text))
            assert(box[2]>box[0])
            assert(box[3]>box[1])
            obj={
                'bbox':box,
                'bbox_mode':BoxMode.XYXY_ABS,
                'category_id':0,
                'iscrowd':0,
            }
            objs.append(obj)
        record['annotations']=objs
        dataset_dicts.append(record)
    return dataset_dicts

def xml2json(img_dir, output_dir = None):
    '''
    Convert the dataset annotations into a COCO format ground truth json file so as to call COCOEvaluator
    Args:
        img_dir(string): the path of the dataset which stores the pictures and annitations, e.g. 'simImage/val/'
        output_dir(string): the directory to save the json file
    '''
    xml_files=os.listdir(os.path.join(img_dir,'Annotations'))
    info = {}
    info['version'] = 'v0'
    info['description'] = 'a tiny dataset for learning detectron2'
    info['contributor'] = 'chizao'
    now = datetime.now()
    info['date_created'] = str(now)
    info['year'] = now.year
    images = []
    categories = []
    annotations = []
    category = {}
    category['id'] = 0
    category['name'] = 'danger'
    categories.append(category)
    anno_id = 0
    for file in xml_files:
        if file == '.ipynb_checkpoints':
            continue
        img_record = {}
        
        img_record['width'] = 190
        img_record['height'] = 380
             
        tree=etree.parse(os.path.join(img_dir,'Annotations',file))
        root=tree.getroot()
        bboxes=[]
        for c in root.iter():
            if c.tag == 'filename':
                filename = c.text
                img_record['file_name'] = filename
                img_record['id']=int(filename[:-4].replace('-',''))
            if c.tag=='bndbox':
                bboxes.append(c)
                
        for bbox in bboxes:
            anno_record ={}
            anno_record['id'] = anno_id
            anno_id += 1
            anno_record['image_id'] = img_record['id']
            anno_record['category_id'] = 0
            anno_record['iscrowd'] = 0
            box=[]
            for b in bbox.iterchildren():
                box.append(int(b.text))
            assert(box[2] > box[0])
            assert(box[3] > box[1])
            width = box[2] - box[0]
            height = box[3] - box[1]
            anno_record['bbox'] = [box[0], box[1], width, height]
            anno_record['area'] = float(width*height)
            annotations.append(anno_record)
        images.append(img_record)
    anno ={}
    anno['info'] = info
    anno['images'] = images
    anno['annotations'] = annotations
    anno['categories'] = categories
    anno_json = json.dumps(anno)
    if output_dir == None:
        output_dir = img_dir
    with open(os.path.join(output_dir,'annotations.json'),'w') as f:
        f.write(anno_json)