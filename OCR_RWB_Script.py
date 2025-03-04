import os
from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from PyPDF2 import PdfReader, PdfWriter
from time import perf_counter
import re
import tkinter as tk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo, showerror
from tkinter import ttk
import traceback

def timeit(inner):
    def timeit_wrapper(*args, **kwargs):
        t1 = perf_counter()
        result = inner(*args, **kwargs)
        t2 = perf_counter()
        delta = t2 - t1
        print(f"It took {delta} seconds")
        return result
    return timeit_wrapper

def rotate_table(image):
        
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding to obtain a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (presumably the table)
    contours = sorted(contours, key = cv2.contourArea)
    largest_contour = contours[-1] 
    #largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the minimum area rectangle enclosing the contour
    rect = cv2.minAreaRect(largest_contour)
    
    # Determine the angle of rotation
    angle = rect[2]
    #print("angle = ", angle)
    rot_angle = 0
    if angle<2:
        rot_angle = angle
    if angle>88:
        rot_angle = angle - 90
    # Rotate the image around the center
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), rot_angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))
    
    # Display the original and rotated images
    #cv2.imshow("Original", image)
    #cv2.imshow("Rotated", rotated)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return rotated

def get_text(idx, page):
    img = np.array(page)
    ### Rotate
    #original = img[0:int(img.shape[1]/1), 0:int(img.shape[0]/1)] #Might increase efficiency
    original=img
    original = rotate_table(original)
    # Convert image from RGB to BGR 
    original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
    #original = cv2.convertScaleAbs(original, alpha=2)
    
    # Convert image to grayscale
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    #threshl, img_bin = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    img_bin = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,33,10)

    # Get vertical and horizontal lines of the grid with erosion and dilation
    ver_d_iter = int(spinbox1.get())
    hor_d_iter = int(spinbox2.get())
    erod_v_iter = int(spinbox3.get())
    erod_h_iter = int(spinbox4.get()) 

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, np.array(img_bin).shape[0]//150))
    eroded_image = cv2.erode(img_bin, vertical_kernel, iterations=erod_v_iter) # Number of iterations can be adjusted
    ver_lines = cv2.dilate(eroded_image, vertical_kernel, iterations=ver_d_iter)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (np.array(img_bin).shape[1]//150, 1)) #change kernel!!!!!!!! increase devider 
    eroded_image1 = cv2.erode(img_bin, horizontal_kernel, iterations=erod_h_iter) # Number of iterations can be adjusted
    hor_lines = cv2.dilate(eroded_image1, horizontal_kernel, iterations=hor_d_iter)

    #cv2.imshow("hor lines", hor_lines)

    grid = cv2.addWeighted(hor_lines, 1, ver_lines, 1, 0)
    grid = cv2.dilate(grid, vertical_kernel, iterations=ver_d_iter)
    grid = cv2.dilate(grid, horizontal_kernel, iterations=hor_d_iter)
    grid = 255 - grid
    
    contours, hierarchy = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    blank = np.zeros(original.shape)
    
    roi = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        if y > img.shape[0]/8 and y < img.shape[0]/5 and w > img.shape[1]/3 and w < img.shape[1]/2 and h < img.shape[0]/11 and h > img.shape[0]/16:
            #print(y, x, original.shape[0], original.shape[1], w, h)
            #image = cv2.rectangle(blank,(x,y),(x+w,y+h),(0,255,0),2)
            roi = img_bin[y:y+h, x:x+w]
        boxes.append([x, y, w, h])
        #if w > original.shape[1]/3 and w < original.shape[1]/2 and h < original.shape[0]/11 and h > original.shape[0]/15: #Debug
        cv2.rectangle(original,(x,y),(x+w,y+h),(0,255,0),2) #Debug
    
    # For debugging
    if roi is None:
        if idx == 0:
            
            cv2.imshow("BL", original) 
            cv2.waitKey(0)
            

        return None

#######################
   # cv2.imshow('bl', roi) 
   # cv2.waitKey(0)    
##############################

    out = pytesseract.image_to_string(roi, lang='rus')
    result = tokenizer.tokenize(out)

    #####################
 #   print(result)
    ##################
    
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

#Clean strings
def clean_string(input_string):
    input_string = input_string.lower()
    return re.sub(r'[^A-Za-z0-9]', '', input_string)

#longest sub-string distance
def sub_string(a,b):
    a , b = "".join(a), "".join(b)
    a, b = clean_string(a), clean_string(b)

    for i in range(len(a)-1,0,-1):
        left = (a[:i+1])
        right = (a[len(a)+1-i:])
        if left in b:
            #print(left, a, b)
            return len(left)/len(a)
            
        if right in b:
            #print(right, a, b)
            return len(right)/len(a)
    return 0

# Calculate closest match
def find_best_match(result): # Result should be a tokenized output of OCR
    sim = 0 
    rec = ''
    
    for r in t_receivers:
        l_sub = sub_string(result, r) # FOR LATER!!!!
        s = cosine_similarity(result, r)
        if s > sim:
            sim = s
            rec = r

    if rec == '':
        return "Not found..."

    return receivers.receiver[t_receivers.index(rec)]

def solve(idx, page):
    result = get_text(idx, page)


    if result is None and idx==0:
        raise Exception('Change parameters')

    if result is None:
        return 'Not found...'
        
    receiver = find_best_match(result)
    return receiver

def get_page_ranges(file):
    images = convert_from_path(file)
    ranges = {}

    #######
    current_operation.set(f"Getting page ranges {progress['value']:0.0f}%")
    root.update_idletasks()
    #######
    rec = ''
    for idx, page in enumerate(images):
        result = solve(idx, page)
        n = images.index(page)
        
        if result != 'Not found...':
            rec = result
        if rec in ranges.keys():
            ranges[rec].append(n)
        else:
            ranges[rec] = [n]
        
        #######Progress bar and label activation####
        progress['value'] += 1 / len(images) * 100
        current_operation.set(f"Getting page ranges {progress['value']:0.0f}%")
        root.update_idletasks()
        #######
        
    ###Reset bar###
    progress['value']+= 1 / len(images) * 100
    progress['value'] = 0
    return ranges
    
@timeit
def devide_file(file):
    ranges = get_page_ranges(file)
    reader = PdfReader(file)
    for r in ranges:
        
        writer = PdfWriter()
        for n in ranges[r]:
            writer.add_page(reader.pages[n])
        with open(f"{file[:-4]}_{r}.pdf", "wb") as fp:
            writer.write(fp)
        #### Progress bar and label activation####
        progress['value']+= 1 / len(ranges) * 100
        current_operation.set(f"Devidig file {progress['value']:0.0f}%")
        root.update_idletasks()
        #######
    ###Reset bar###
    progress['value'] = 0    

def get_receivers(rec_file):
    receivers = pd.read_excel(rec_file, engine='openpyxl')

    # Tockenize list of receivers
    t_receivers = []
    for r in receivers.name.values:
        t_receivers.append(tokenizer.tokenize(str(r)))
    return t_receivers, receivers

def open_file_dialog():
    global file
    file = fd.askopenfile(title="Select file for OCR").name
    print(file)
    
def OCR():
    try:
        devide_file(file)
        all_good = tk.messagebox.showinfo("Result", "All good")
    except Exception as err:   
        err_box = tk.messagebox.showerror("Error", str(err) + traceback.format_exc())
          

if __name__ == "__main__":
    rec_file = os.path.dirname(__file__)+"\\receivers.xlsx"
    tokenizer = RegexpTokenizer(r'\w+')
    t_receivers, receivers = get_receivers(rec_file)

    root = tk.Tk()

    # Set the window size and position it in the center of the screen
    window_width = 400
    window_height = 200
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width // 2) - (window_width // 2)
    y = (screen_height // 2) - (window_height // 2)
    root.geometry(f'{window_width}x{window_height}+{x}+{y}')

    # Create a label for the first Spinbox
    label1 = tk.Label(root, text="Ver dilate iter:")
    label1.grid(row=0, column=0, pady=5, padx=5)

    # Create the first Spinbox for integer input
    spinbox1 = tk.Spinbox(root, from_=1, to=9, increment=1, width=10)
    spinbox1.grid(row=1, column=0, pady=5, padx=5)
    #spinbox1.insert(tk.END,3)

    # Create a label for the second Spinbox
    label2 = tk.Label(root, text="Hor dilate iter:")
    label2.grid(row=0, column=1, pady=5, padx=5)

    # Create the second Spinbox for integer input
    spinbox2 = tk.Spinbox(root, from_=1, to=9, increment=1, width=10, textvariable=tk.DoubleVar(value=2))
    spinbox2.grid(row=1, column=1, pady=5, padx=5)
    #spinbox2.insert(tk.END, "2")

    # Create the third Spinbox for integer input
    label3 = tk.Label(root, text='Erode v iter:')
    label3.grid(row=0, column=2, pady=5, padx=5)
    spinbox3 = tk.Spinbox(root, from_=1, to=9, increment=1, width=10, textvariable=tk.DoubleVar(value=1) )
    spinbox3.grid(row=1, column=2, pady=5, padx=5)

    # Create the third Spinbox for integer input
    label4 = tk.Label(root, text='Erode h iter:')
    label4.grid(row=0, column=3, pady=5, padx=5)
    spinbox4 = tk.Spinbox(root, from_=1, to=9, increment=1, width=10, textvariable=tk.DoubleVar(value=2) )
    spinbox4.grid(row=1, column=3, pady=5, padx=5)

    # Create a button in the center of the window
    button1 = tk.Button(root, text="Open File", command=open_file_dialog)
    button1.place(relx=0.25, rely=0.5, anchor=tk.CENTER)

    # Create a button in the center of the window
    button2 = tk.Button(root, text="OCR", command=OCR)
    button2.place(relx=0.75, rely=0.5, anchor=tk.CENTER)

    # Create a label for the current operation
    current_operation = tk.StringVar()
    label = tk.Label(root, textvariable=current_operation)
    label.place(relx=0.5, rely=0.35, anchor=tk.CENTER)

    # Create a progress bar
    progress = ttk.Progressbar(root, mode='determinate')
    progress.place(relx=0.5, rely=0.7, anchor=tk.CENTER)

    root.mainloop()
