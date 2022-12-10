import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import traceback
from Dissect import AbstractLine
import time
import seaborn as sns
import pretty_midi
import utils

def page_detect_line(img, k_blur=25):
    blurred = cv.medianBlur(img, k_blur)
    gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)
    edges = cv.dilate(cv.Canny(gray, 30, 150), np.ones((3, 3), np.uint8))

    lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/360, threshold=100, minLineLength=300, maxLineGap=50)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
    
    blurred[edges > 0] = (0, 255, 0)
    plt.imshow(img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def dissect_rows(img, binary, low_bound=2, min_height=60):
    assert binary.ndim == 2, 'binary must be negative binary'
    assert binary.shape[:2] == img.shape[:2], 'binary and img must have same height and width'
    
    projection_y = binary.sum(axis=1) / 255
    row_ranges = []
    top = -1
    
    for i, size in enumerate(projection_y):
        size = int(size)
        if top == -1:
            if size > low_bound:
                top = i
        elif size <= low_bound:
            if i-top >= min_height  and projection_y[i-1]<=low_bound and projection_y[i+1]<=low_bound and projection_y[i-2]<=low_bound and projection_y[i+2]<=low_bound:
                row_ranges.append((top, i))
                top = -1
    
    row_binaries, row_imgs = [], []
    for top, bottom in row_ranges:
        row_binaries.append(binary[top:bottom+1, :binary.shape[1]])
        cv.line(adjusted,(0,bottom),(binary.shape[1],bottom),(0,255,0),1,1)
        row_imgs.append(img[top:bottom+1, :img.shape[1]])

    return row_imgs, row_binaries, row_ranges

def fill_object(img, binary, seed, expand=1):
    assert isinstance(seed, tuple)
    mask = np.zeros(tuple(s+2 for s in binary.shape), np.uint8)
    area, binary, mask, (x, y, w, h) = cv.floodFill(binary, mask, seed, (127), (0), (0), flags=(8 | 255 << 8))
    mask = mask[1:-1, 1:-1]
    mask = cv.dilate(mask, np.ones((2*expand+1, 2*expand+1), np.uint8)) \
        [max(0, y-expand) : y+h+expand, max(0, x-expand) : x+w+expand]
    img = img[max(0, y-expand) : y+h+expand, max(0, x-expand) : x+w+expand]
    assert img.shape == mask.shape, 'img and mask must have same shape'
    result = np.ones(mask.shape, np.uint8) * 255
    cropped = cv.bitwise_and(img, img, mask=mask)
    result[mask == 255] = cropped[mask == 255]
    return (x, y, w, h), result

def dissect_objects(img, binary):
    obj_dict = {}
    for pos, pixel in np.ndenumerate(binary):
        if pixel > 250:
            xywh, obj = fill_object(img, binary, (pos[1], pos[0]))
            obj_dict[xywh] = obj
            x,y,w,h=xywh
            row_add=row_ranges[index][0]
    return obj_dict


def has_interset(x1,x2,X1,X2):
    if x2 < X1 or X2<x1:
        return False
    return True

def reconstruction(line):
    global t
    chars,dots,underlines,tuplet,overlines=line
    for kkey in tuplet:
        chars[kkey]=-1
    chars=sorted(chars.items(),key=lambda x:x[0][0])
    for (x,y,w,h),num in chars:
        note_length = 0.5
        note_pitch = 60
    # underlines
        for (tx,ty,tw,th) in underlines:
            if has_interset(x , x+w , tx, tx+tw):
                note_length = note_length/2


    # dots
        for (tx,ty,tw,th) in dots:        
            if has_interset(x,x+w,tx,tx+tw)==True:
                if ty<y:
                    note_pitch+=12
                elif ty>y+h:
                    note_pitch-=12
            elif has_interset(y,y+h,ty,ty+th) and tx>x and tx-x<3*w:
                 note_length = note_length + note_length/2
        
        if num<=0:
            t+=note_length
    # convert into midi info
        else:
            if num == 2:
                note_pitch += 2
            elif num == 3:
                note_pitch += 4
            elif num == 4:
                note_pitch += 5
            elif num == 5:
                note_pitch += 7
            elif num == 6:
                note_pitch += 9
            elif num == 7:
                note_pitch += 11
            
            note = pretty_midi.Note(100, note_pitch + tone_skewing , t , t+note_length)
            piano.notes.append(note)
            t+=note_length
    

def tone_verrify(img,obj_dict):
    global tone_skewing
    obj = list(obj_dict.keys())
    obj = sorted(obj,key=lambda x:x[0])
    plt.figure()
    TO = {}
    # for item in obj:
    #     utils.display(img, item[0], item[1], item[2], item[3],show=True)
    temp=obj[3]
    TO[temp] = -1
    utils.display(img,temp[0],temp[1],temp[2],temp[3],show=True)
    utils.pattern_match(img, TO,font="Font 2")
    if TO[temp]==0:
        tone_skewing=0
    elif TO[temp]==1:
        tone_skewing=2
    elif TO[temp]==2:
        tone_skewing=7    
    elif TO[temp]==3:
        tone_skewing=5    
    
    
if __name__ == '__main__':
    img=cv.imread("media/experiment/img1.jpg")
    adjusted = cv.cvtColor(img, cv.COLOR_BGR2GRAY)            
    _, binarized = cv.threshold(adjusted, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    binarized = cv.bitwise_not(binarized)   #二值化
    row_imgs, row_binaries, row_ranges = dissect_rows(adjusted, binarized)  
    index = 0 
    line_id = 0       
    plt.figure()  
    enc_res=pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=1)
    tone_skewing = 0
    t=0
    for img, binary in zip(row_imgs, row_binaries): 
        line_id +=1
        print(line_id)
        obj_dict = dissect_objects(img, binary)
        line = AbstractLine.construct(img, obj_dict)
        if line=="NULL":
            if line_id==2:
                tone_verrify(img,obj_dict)
                print("shift:",tone_skewing)
            continue
        else:
            reconstruction(line)
        index += 1

    enc_res.instruments.append(piano)
    enc_res.write("media\\encoded midi files\\ans.mid".format("ans"))
    