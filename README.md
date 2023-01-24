
# Splitting PDF files based on the conten

## Problem setup

On daily basis operations staff receives numerous documents in *PDF* format. At least 50% of the information is comprised of scans, which may have completely different quality. One of the documents I work with is a *Railway bill*, which is a document issued by a Railroad Authority for each wagon or group of wagons accepted for the transportation. This document is quite remarkable in a sense that it is an irregular table with some hand-written remarks, stamps and other features.  

![First three pages](/assets/img/3pages.png)

Most import for us is the first page, which contains information about receiver. Here I provide a picture, where the *region of interest* (ROI) is marked by hand.  

![First page with the mark](/assets/img/1page_marked.png)  

The main technical goal is not only to find the ROI, but also to define, which receiver is specified in it. For that reason we will be using simple *OCR* procedure with [Tesseract](https://github.com/tesseract-ocr/tesseract).  

The last step is classification, which shall be performed as a smple [*cosine similarity*](https://en.wikipedia.org/wiki/Cosine_similarity) procedure.

---

## Solution

> Present solution was tested on Windows 10 machine with Python 3.11

First of all let's list the packages used:
* Pandas
* Numpy
* OpenCV-Python
* PyPDF2
* pdf2image
* nltk
* pytesseract
> **IMPORTANT:** don't forget to install Tesseract prior to installation of its Python wrapper.

```Python
from pdf2image import convert_from_path
import cv2
import numpy as np
import pytesseract
from nltk.tokenize import RegexpTokenizer
import pandas as pd
from PyPDF2 import PdfFileReader, PdfFileWriter
```

Most importan part of the script is an OCR part, which is wrapped in a function **```get_text(page)```**. Since page is an image, generated internally by **pdf2image** package, we need to convert it to Numpy array. 

The following sequence of transformations is applied: 

<div class="mermaid">
sequenceDiagram
    participant Alice
    participant Bob
    Alice->>John: Hello John, how are you?
    loop Healthcheck
        John->>John: Fight against hypochondria
    end
    Note right of John: Rational thoughts <br/>prevail!
    John-->>Alice: Great!
    John->>Bob: How about you?
    Bob-->>John: Jolly good!
</div>
