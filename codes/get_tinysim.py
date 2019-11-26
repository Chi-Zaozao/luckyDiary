from detectron2.structures import BoxMode
from lxml import etree
import os

def get_tinysim(data_dir):
    '''
    Convert the dataset annotations into a format which detectron2 can accept and register
    Args:
        data_dir(string): the path of the dataset which stores the pictures and annitations, e.g. 'simImage/val'
    Returns:
        list[dict]: in the required format as declared in https://detectron2.readthedocs.io/tutorials/datasets.html
    '''
    xml_files=os.listdir(os.path.join(data_dir,'Annotations'))
    dataset_dicts=[]
    for file in xml_files:
        if file == '.ipynb_checkpoints':
            continue
        record={}
        tree=etree.parse(os.path.join(data_dir,'Annotations',file))
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
        image_id = ''
        for i in filter(str.isdigit, filename):
            image_id += i
        record['image_id']=int(image_id)
        record['file_name']=os.path.join(data_dir,filename)
        record['width']=int(texts[tags.index('width')])     # optional for object detection
        record['height']=int(texts[tags.index('height')])   # optional for object detection
        if int(filename[:3]) == 2:
            category_id = 0
        else:
            category_id = 1
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
                'category_id':category_id,
#                 'iscrowd':0,    # optional for object detection
            }
            objs.append(obj)
        record['annotations']=objs
        dataset_dicts.append(record)
    return dataset_dicts