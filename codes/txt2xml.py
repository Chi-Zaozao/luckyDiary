#run in the same path of the folder 'name'
"""
Created on Wed Sep 25 08:44:35 2019

@author: LiHuaiqian
"""

import os
from lxml import etree

def create_xml(info):
    root=etree.Element('annotation')
    folder=etree.SubElement(root,'folder')
    # root.append(etree.Element('filename')) This is another syntax 
    filename=etree.SubElement(root,'filename')
    source=etree.SubElement(root,'source')
    owner=etree.SubElement(root,'owner')
    size=etree.SubElement(root,'size')
    seg=etree.SubElement(root,'segmented')
    obj=etree.SubElement(root,'object')
    
    folder.text='simi-data-201710'
    
    filename.text=info['filename']+'jpg'
    
    db=etree.SubElement(source,'database')
    anno_s=etree.SubElement(source,'annotation')
    img=etree.SubElement(source,'image')
    flid_s=etree.SubElement(source,'flickrid')
    db.text='simi-data-201710'
    anno_s.text='simi-data-201710'
    img.text='flickr'
    flid_s.text='201701018'
    
    flid_o=etree.SubElement(owner,'flickrid')
    name_o=etree.SubElement(owner,'name')
    flid_o.text='Random'
    name_o.text='SimImage'
    
    width=etree.SubElement(size,'width')
    height=etree.SubElement(size,'height')
    depth=etree.SubElement(size,'depth')
    width.text=info['width']
    height.text=info['height']
    depth.text=info['depth']
    
    seg.text='0'

    name_ob=etree.SubElement(obj,'name')
    pose=etree.SubElement(obj,'pose')
    trun=etree.SubElement(obj,'truncated')
    hard=etree.SubElement(obj,'difficult')
    name_ob.text='object'
    pose.text='Unspecified'
    trun.text='0'
    hard.text='0'
    if not len(info['bboxes'])%4:
        for i in range(0,len(info['bboxes']),4):
            bbox=etree.SubElement(obj,'bndbox')
            xmin=etree.SubElement(bbox,'xmin')
            xmin.text=info['bboxes'][i]
            ymin=etree.SubElement(bbox,'ymin')
            ymin.text=info['bboxes'][i+1]
            xmax=etree.SubElement(bbox,'xmax')
            xmax.text=info['bboxes'][i+2]
            ymax=etree.SubElement(bbox,'ymax')
            ymax.text=info['bboxes'][i+3]
    file=etree.ElementTree(root)
    file.write('Annotations/'+info['filename']+'xml',encoding='utf-8',xml_declaration=True,pretty_print=True)
            
    
if __name__=='__main__':   
    anno='Annotations'
    if not os.path.exists(anno):
        os.mkdir(anno)
    files=os.listdir(os.getcwd()+'/name')
    for file in files:
        with open(os.getcwd()+'/name/'+file,'r') as f:
            lines=f.readlines()
            try:
                s=''
                for line in lines[:-1]:
                    s+=line[:-1]
                s+=lines[-1]
                bboxes=s.split(',')
                print(bboxes)
                info={}
                info['width']='190'
                info['height']='380'
                info['depth']='1'
                info['bboxes']=bboxes
                info['filename']=file[:-3]
                create_xml(info)
                if len(bboxes)%4:
                    100/0
            except ZeroDivisionError:
                print('error: file '+ file + ' has wrong data number! ')  
            except:
                print('error: file '+ file + ' has no object! ')                       
    print('Done!')