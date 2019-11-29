import os
import cv2

def get_body(img_dir):
    '''
    give the bboxes of the human of the given images
    Args:
        img_dir(string): the directory of the images
    Return:
        bboxes(list of dictionary[{}])：the dictionary includes 2 items:
             bbox(list): [xmin, ymin, xmax, ymax]
             filename(string): the path of the image
    '''
    files = os.listdir(img_dir)
    bboxes = []
    for file in files:
        if not file[-3:] == 'jpg':
            continue
        record = {}
        bbox = []
        filename = os.path.join(img_dir, file)
        record['filename'] = filename
        source_img = cv2.imread(filename)
        down_img = cv2.pyrDown(source_img, cv2.IMREAD_UNCHANGED)
        blur = cv2.GaussianBlur(down_img[:,:,0],(9,9),0)
        ret,thresh  = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        xmin = []
        ymin = []
        xmax = []
        ymax = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w*h<40 or y<25:
                continue
            xmin.append(x)
            ymin.append(y)
            xmax.append(x+w)
            ymax.append(y+h)
        bbox.append(2*min(xmin))
        bbox.append(2*min(ymin))
        bbox.append(2*max(xmax))
        bbox.append(2*max(ymax))
        record['bbox'] = bbox
        bboxes.append(record)
    return bboxes
