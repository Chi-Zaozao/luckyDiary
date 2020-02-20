from lxml import etree
import json
import os
import matplotlib.image as mpimg

# def xml2dicts(anno_dir,file,anno_id,categories_list):
# def xml2dicts(anno_dir,img_dir,file,anno_id,categories_list,y):
def xml2dicts(anno_dir,img_dir,file,anno_id,categories_list):
    '''
    Used by function `split_dataset', 
    Convert the xml annotation file into the format function `split_dataset' accept
    Args:
        anno_dir(string): the path of the dataset annitations, e.g. 'simdata/val/Annotations'
        img_dir(string): the path of the dataset images, 
            to read the image shape
            (hardcoded as (380,190) in xml files but that's not ture)
        file: the filename of the xml annotation file
        anno_id: the annotation id of the first annotation
        categories_list: the list of the categories
    Returns:
        tuple: the tuple contains 1 dictionary, 1 list and 1 int:
            dictionary: the image dictionary in COCO format
            list: the list of the annotation dictionaries in COCO format
            e.g. ({image_dict}, [{anno_dict1}, {anno_dict2}, ...])
    '''
    img_record = {}
    
    img_name = file[:-3]+'jpg'
    
    image_id = ''
    for i in filter(str.isdigit, img_name):
        image_id += i
    img_record['file_name'] = img_name
    img_record['id']=int(image_id)
    
#     if img_name in y:
#         img_record['width'] = 380
#         img_record['height'] = 760
#     else:
    img_record['width'] = 190
    img_record['height'] = 380
#     img = mpimg.imread(os.path.join(img_dir, img_name))
#     (height, width) =  img.shape
#     img_record['width'] = width
#     img_record['height'] = height
             
    tree=etree.parse(os.path.join(anno_dir,file))
    root=tree.getroot()
    bboxes=[]
    for c in root.iter():
        if c.tag=='object':
            bboxes.append(c)
    
    annotations = []           
    for bbox in bboxes:
        anno_record ={}
        anno_record['id'] = anno_id
        anno_id += 1
        anno_record['image_id'] = img_record['id']
        anno_record['iscrowd'] = 0
        box=[]
        for b in bbox.iter(): 
            if b.tag == 'name':
                anno_record['category_id'] = categories_list.index(b.text)
            elif b.tag == 'xmin':
                xmin = int(b.text)-1
            elif b.tag == 'ymin':
                ymin = int(b.text)-1
            elif b.tag == 'xmax':
                xmax = int(b.text)-1
            elif b.tag == 'ymax':
                ymax = int(b.text)-1
        if file[:5]=='Train':
            xmin = int(xmin/2)
            ymin = int(ymin/2)
            xmax = int(xmax/2)
            ymax = int(ymax/2)
        width = xmax - xmin +1
        height = ymax - ymin + 1
        anno_record['bbox'] = [xmin, ymin, width, height]
        anno_record['area'] = float(width*height)
        annotations.append(anno_record)
    return img_record, annotations
    
def split_dataset(anno_dir, img_dir, output_dir = None):
    '''
    Convert the xml dataset annotations into 2 COCO format ground truth [train, test] annotations json files,
    so as to call `COCOEvaluator` and function `register_coco_instances`
    Args:
        anno_dir(string): the path of the dataset annitations, e.g. 'simdata/val/Annotations'
        img_dir(string): the path of the dataset images, to make sure that there is a image corresponding to the xml file
        output_dir(string): the directory to save the 2 json files: train_anno.json & test_anno.json
    '''
#     with open('./projects/TridentNet_5class/dataset/big_image_760x380.txt', 'r') as f:
#         x = f.readlines()
#     y = []
#     for i in x:
#         y.append(i[:-27])
        
    xml_files=os.listdir(anno_dir)
    img_files=os.listdir(img_dir)
    
    json_train = {}
    json_test = {}
    
    categories = []
    categories_list = ['gun', 'knife', 'cellphone', 'explosive', 'object']
    for i,j in enumerate(categories_list):
        category_dict = {}
        category_dict['id'] = i
        category_dict['name'] = j
        categories.append(category_dict)
    json_train['categories'] = categories
    json_test['categories'] = categories
    
    train_images = []
    test_images = []

    train_annotations = []
    test_annotations = []
    
    anno_id = 0
    for file in xml_files:
        # becasue I'm using Jupyterlab
        # which always creates the boring cache file/directory
        if not file[-3:] == 'xml':
            continue
        elif file[:-3]+'jpg' not in img_files:
            continue
        elif file[:8] == 'Train_12':
            anno_single_file = xml2dicts(anno_dir,img_dir,file,anno_id,categories_list)
            test_images.append(anno_single_file[0])
            test_annotations.extend(anno_single_file[1])
            anno_id += len(anno_single_file[1])
        else:
            anno_single_file = xml2dicts(anno_dir,img_dir,file,anno_id,categories_list)
            train_images.append(anno_single_file[0])
            train_annotations.extend(anno_single_file[1])
            anno_id += len(anno_single_file[1])
            
    json_train['images'] = train_images
    json_test['images'] = test_images
    json_train['annotations'] = train_annotations
    json_test['annotations'] = test_annotations
    
    json_train_file = json.dumps(json_train)
    json_test_file = json.dumps(json_test)
    if output_dir == None:
        output_dir = anno_dir
    with open(os.path.join(output_dir,'train_annotations.json'),'w') as f:
        f.write(json_train_file)
    with open(os.path.join(output_dir,'test_annotations.json'),'w') as f:
        f.write(json_test_file)
