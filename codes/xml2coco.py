from lxml import etree
import json
import os

def xml2coco(data_dir, output_dir = None):
    '''
    Convert the tiny_simdata dataset annotations into a COCO format ground truth json file so as to call COCOEvaluator
    Args:
        data_dir(string): the path of the dataset which stores the pictures and annitations, e.g. 'tiny_simdata/val'
        output_dir(string): the directory to save the json file
    '''
    xml_files=os.listdir(os.path.join(data_dir,'Annotations'))
    
    json_anno = {}
    
    categories = [{'id': 2, 'name': 'gun'}, {'id':17, 'name': 'lighter'}]
#     for i in range(78):
#         cat = {}
#         cat['id'] = i+80
#         cat['name'] = str(i)+'aaaa'
#         categories.append(cat)
    json_anno['categories'] = categories
    
    images = []
    annotations = []
    
    anno_id = 0
    for file in xml_files:
        # becasue I'm using Jupyterlab
        # which always creates the boring file/directory
        if file == '.ipynb_checkpoints':
            continue
        img_record = {}
        
        img_record['width'] = 190     # optional for object detection
        img_record['height'] = 380     # optional for object detection
             
        tree=etree.parse(os.path.join(data_dir,'Annotations',file))
        root=tree.getroot()
        bboxes=[]
        for c in root.iter():
            if c.tag == 'filename':
                filename = c.text
                image_id = ''
                for i in filter(str.isdigit, filename):
                    image_id += i
                img_record['file_name'] = filename
                img_record['id']=int(image_id)
            if c.tag=='bndbox':
                bboxes.append(c)
                
        for bbox in bboxes:
            anno_record ={}
            anno_record['id'] = anno_id
            anno_id += 1
            anno_record['image_id'] = img_record['id']
            anno_record['category_id'] = int(filename[:3])
            anno_record['iscrowd'] = 0
            box=[]
            for b in bbox.iterchildren():
                box.append(int(b.text))
            assert(box[2] > box[0])
            assert(box[3] > box[1])
            width = box[2] - box[0]
            height = box[3] - box[1]
            anno_record['bbox'] = [bbox[0], bbox[1], width, height]
            anno_record['area'] = float(width*height)
            annotations.append(anno_record)
        images.append(img_record)

    json_anno['images'] = images
    json_anno['annotations'] = annotations
    
    json_file = json.dumps(json_anno)
    if output_dir == None:
        output_dir = data_dir
    with open(os.path.join(output_dir,'annotations.json'),'w') as f:
        f.write(json_file)