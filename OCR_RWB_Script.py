from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from PyPDF2 import PdfFileReader, PdfFileWriter

def get_text(page):
    img = np.array(page)
    original = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    threshl, img_bin = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(img_bin).shape[1]//150))
    eroded_image = cv2.erode(img_bin, vertical_kernel, iterations=5)
    ver_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=3)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(img_bin).shape[0]//150, 1))
    eroded_image1 = cv2.erode(img_bin, horizontal_kernel, iterations=2) # Number of iterations can be adjusted
    hor_lines = cv2.dilate(eroded_image1, horizontal_kernel, iterations=3)

    grid = cv2.addWeighted(hor_lines, 0.5, ver_lines, 0.5, 240)
    grid = cv2.dilate(grid, horizontal_kernel, iterations=1)
    grid = cv2.dilate(grid, vertical_kernel, iterations=1)

    contours, hierarchy = cv2.findContours(255-grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    blank = np.zeros(original.shape)
    
    roi = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if y > original.shape[0]/8 and y < original.shape[0]/5:
            image = cv2.rectangle(blank,(x,y),(x+w,y+h),(0,255,0),2)
            roi = img_bin[y:y+h, x:x+w]
        boxes.append([x, y, w, h])
    #    cv2.rectangle(blank,(x,y),(x+w,y+h),(0,255,0),2) #Debug
    #cv2.imshow('bl', blank) # Debug
    #cv2.waitKey(0)
    if roi is None:
        return None

    out = pytesseract.image_to_string(roi, lang='rus')
    result = tokenizer.tokenize(out)
    
    return result

def cosine_similarity(a, b):
    l1=[]
    l2=[]
    a_set = set(a)
    b_set = set(b)
    r_vector = a_set.union(b_set)
    for w in r_vector:
        if w in a_set: 
            l1.append(1)
        else:
            l1.append(0)
        if w in b_set:
            l2.append(1)
        else:
            l2.append(0)

    if sum(l1) == 0: 
        return 0

    c = np.dot(l1, l2)
    
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    return cosine

# Calculate closest match
def find_best_match(result): # Result should be a tokenized output of OCR
    sim = 0 
    rec = ''
    
    for r in t_receivers:
        s = cosine_similarity(result, r)
        if s > sim:
            sim = s
            rec = r

    if rec == '':
        return "Not found..."

    return receivers.receiver[t_receivers.index(rec)]

def solve(page):
    result = get_text(page)
    if result is None:
        return 'Not found...'
        
    receiver = find_best_match(result)
    return receiver

def get_page_ranges(file):
    images = convert_from_path(file)
    ranges = {}
    for page in images:
        result = solve(page)
        n = images.index(page)
        if result != 'Not found...':
            rec = result
        if rec in ranges.keys():
            ranges[rec].append(n)
        else:
            ranges[rec] = [n]
    return ranges

def devide_file(file):
    ranges = get_page_ranges(file)
    reader = PdfFileReader(file)
    for r in ranges:
        writer = PdfFileWriter()
        for n in ranges[r]:
            writer.add_page(reader.pages[n])
        with open(f"{r}_{file}", "wb") as fp:
            writer.write(fp)    

def get_receivers(rec_file):
    receivers = pd.read_excel(rec_file)

    # Tockenize list of receivers
    t_receivers = []
    for r in receivers.name.values:
        t_receivers.append(tokenizer.tokenize(str(r)))
    return t_receivers, receivers

if __name__ == '__main__':
    file = "КИТАЙ 24.01.2023.pdf"
    rec_file = "receivers.xlsx"
    tokenizer = RegexpTokenizer(r'\w+')
    t_receivers, receivers = get_receivers(rec_file)
    devide_file(file)
