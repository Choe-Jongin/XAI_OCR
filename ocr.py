from matplotlib import pyplot as plt
from easyocr import Reader
import math
import cv2
import pytesseract
import numpy as np
import json
import pandas as pd
import argparse
import os
from pdf2image import convert_from_path
from date_detector.detector import Parser
import pyap
import sys
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QLabel, QHBoxLayout, QGridLayout
from PyQt5.QtGui import QImage, QPixmap
from tkinter import filedialog

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-ocr\\tesseract.exe'


                    
class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('XAI OCR')
        self.move(100, 100)
        self.resize(900, 1000)
        self.show()

class box():
    def __init__(self, l, t, r, b):
        self.l = l
        self.t = t
        self.r = r
        self.b = b
        self.type = ""
        self.block = 0
        self.text = ""
        self.real_value = ""
        self.child = []

    def collision(self, other, x_margin = 0, y_margin = 0):
        if self.r < other.l - x_margin:
            return False
        if self.l > other.r + x_margin:
            return False
        if self.b < other.t - y_margin:
            return False
        if self.t > other.b + y_margin:
            return False
        return True
    
    def merge(self, other):
        self.l = min(self.l, other.l)
        self.t = min(self.t, other.t)
        self.r = max(self.r, other.r)
        self.b = max(self.b, other.b)

    def distance(self, other):
        self_center = [(self.l + self.r)/2, (self.t + self.b)/2]
        other_center = [(other.l + other.r)/2, (other.t + other.b)/2]
        dist = [self_center[0] - other_center[0], self_center[1] - other_center[1]]
        return math.sqrt(dist[0]**2 + dist[1]**2)
    
    def add_child(self, other):
        self.child.append(other)

def plt_imshow(title='image', img=None, figsize=(30, 15)):
    plt.figure(figsize=figsize)
 
    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []
 
            for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
 
        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()

def get_table_blocks(org_image):
    gray_scale=cv2.cvtColor(org_image, cv2.COLOR_BGR2GRAY)
    _, img_bin = cv2.threshold(gray_scale, 150, 225, cv2.THRESH_BINARY)
    img_bin = ~ img_bin

    line_min_width = 30
    kernel_h = np.ones((1, line_min_width), np.uint8)
    kernel_v = np.ones((line_min_width, 1), np.uint8)

    img_bin_h = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel_h)
    img_bin_v = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel_v)
    img_bin_final = img_bin_h | img_bin_v

    final_kernel = np.ones((7,7), np.uint8)
    img_bin_final = cv2.dilate(img_bin_final, final_kernel, iterations=3)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=8, ltype=cv2.CV_32S)

    n1 = np.array(stats[2:])
    box_image = org_image.copy()
    table_list = []
    for x,y,w,h,area, in stats[2:]:
        table_list.append(box(x,y, x+w, y+h))
    
    return table_list

def merge_boxs(box_list, x_margin = 0, y_margin = 0):
    i = 0
    while i < len(box_list):
        j = i+1
        colision = False
        while j < len(box_list):
            box1 = box_list[i]
            box2 = box_list[j]
            if box1.collision(box2, x_margin, y_margin) :
                box1.merge(box2)
                del box_list[j]
                colision = True
            else:
                j+=1
        if colision :
            i = 0
        else :
            i += 1

def read_image(file="sample0.jpg"):
    if ".pdf" in file or "PDF" in file:
        image_file_name = file.replace(".pdf", ".png").replace(".PDF", ".png")
        # convert to jpg
        if not os.path.isfile(image_file_name):
            print("convert pdf to png")
            pages = convert_from_path(file, poppler_path=r'C:\poppler\Library\bin')
            print("save", image_file_name)
            pages[0].save(image_file_name, "png")
        file = image_file_name
    print("read file :", file)
    image = cv2.imread(file)
    return image

def main(path = ""):
    if path == "" :
        print("No directory or file")
        return
    
    files = path
    # 폴더일 경우 폴더 내 모든 파일을 대상으로 실행
    if len(files) == 1 and os.path.isdir(files[0]):
        dirname = files[0]
        files = os.listdir(dirname)
        if dirname[-1] != "/":
            dirname = dirname+"/"
        files = [dirname + file for file in files]

    for file in files:
        # Image read & search the text(box)

        org_image = read_image(file)
        reader = Reader(lang_list=['en'], gpu=True)
        results = reader.readtext(org_image)
        box_list = []
        for line in results:
            (x1,y1) = int(line[0][0][0]), int(line[0][0][1])
            (x2,y2) = int(line[0][2][0]), int(line[0][2][1])
            box_list.append(box(x1,y1,x2,y2))
        

        # Merge adjacent boxes
        original_box_list = box_list.copy()
        margin_finish = False
        x_margin = 5

        while not margin_finish :
            table_list = get_table_blocks(org_image)
            merge_boxs(box_list, x_margin, 0)

            show_image = org_image.copy()
            text_image = np.zeros(org_image.shape, np.uint8)

            print("table",len(table_list))

            # text recog
            final_image = org_image.copy()
            for b in box_list:
                (x1,y1) = (b.l-1, b.t-1)
                (x2,y2) = (b.r+1, b.b+1)
                roi_image = org_image[y1:y2, x1:x2]  
                if roi_image.shape[0] == 0 or roi_image.shape[1] == 0:
                    continue
                text = pytesseract.image_to_string(cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB), config="--psm 4")
                if text == "" :
                    continue
                if text[-1] == "\n":
                    text = text[:-1]
                text = text.replace("\t"," ").replace("\n"," ")
                
                if len(text) == 0 :
                    del b
                    continue
        
                b.text = text
                
                # final_image = cv2.rectangle(final_image, (x1,y1), (x2,y2), (0,0,255), 2)
                # if  b.type=="address":
                #     final_image = cv2.rectangle(final_image, (x1,y1), (x2,y2), (255,0,0), 2)
                # text_image = cv2.putText(text_image, text,(x1,y1), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1)

            # block detection
            for i, box1 in zip(range(len(table_list)), table_list):
                for box2 in box_list:
                    if box1.collision(box2):
                        box1.add_child(box2)
                        box2.block = i+1

            for table in table_list :
                table.child.sort(key=lambda x : x.t)

            # type check
            date_parser = Parser()
            for b in box_list:
                
                # date
                dates = list(date_parser.parse(b.text))
                if dates != []:
                    b.type="date"
                    b.real_value = dates[0]
                    print(b.text)

                # address
                addresses = pyap.parse(b.text, country='US')
                if addresses != []:
                    b.type="address"
                    b.real_value = addresses[0]
                    print(b.text)

            for table in table_list :
                addr_text = ""
                if len(table.child) > 0 :
                    for child_box in table.child :
                        addr_text = addr_text + " " + child_box.text
                addr_text = pyap.parse(addr_text, country='US')
                if addr_text != [] :
                    print(addr_text)
                    table.type = "address"
                    table.real_value = addr_text
            
            ## show 
            # box check
            for b in box_list:
                (x1,y1) = (b.l, b.t)
                (x2,y2) = (b.r, b.b)
                show_image = cv2.rectangle(show_image, (x1,y1), (x2,y2), (0,255,0), 1)
                show_image = cv2.putText(show_image, str(b.block), (x1,y1+10), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
                show_image = cv2.putText(show_image, str(b.type), (x1+30,y1+10), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)
                if b.type == "address" :
                    show_image = cv2.rectangle(show_image, (x1,y1), (x2,y2), (0,0,255), 2)

            # block check
            for b in table_list:   
                (x1,y1) = (b.l, b.t)
                (x2,y2) = (b.r, b.b)
                show_image = cv2.rectangle(show_image, (x1,y1), (x2,y2), (255,0,0), 1)
                if b.type == "address" :
                    show_image = cv2.rectangle(show_image, (x1,y1), (x2,y2), (0,0,255), 2)

            
            ######      MACTHING       #######  
            label_text = ""     
            date_parser = Parser()
            for b in box_list :
                # date matching
                if ("date" in b.text.lower() or 
                    "dat" in b.text.lower() or 
                    "eta" in b.text.lower() ) :
                    dates = list(date_parser.parse(b.text))
                    if dates != []:
                        print( b.text, ":", dates[0].date)
                        label_text +=  b.text.replace(":","") +" : " + str(dates[0].date) + "\n"
                    else :
                        closer_box = None
                        for date_box in box_list :
                            date_parser = Parser()
                            dates = list(date_parser.parse(date_box.text))
                            if dates != [] and b.distance(date_box) < 300:
                                if closer_box == None:
                                    closer_box = date_box
                                elif b.distance(closer_box) > b.distance(date_box):
                                    closer_box = date_box
                        if closer_box != None:
                            dates = list(date_parser.parse(closer_box.text))
                            print( b.text, ":", dates[0].date)
                            label_text +=  b.text.replace(":","") + " : " + str(dates[0].date) + "\n"

                # address matching
                if ("deliver" in b.text.lower() or 
                    "pickup" in b.text.lower() or 
                    "ship" in b.text.lower() or 
                    "address" in b.text.lower()) :

                    if b.type == "address" and b.text.find(":") != -1:
                        print(b.text[:b.text.find(":")], b.real_value)
                    else :
                        closer_box = None
                        for addr_box in box_list :
                            if addr_box.type == "address" and b.distance(addr_box) < 300:
                                if closer_box == None:
                                    closer_box = addr_box
                                elif b.distance(closer_box) > b.distance(addr_box):
                                    closer_box = addr_box
                        for addr_box in table_list :
                            if addr_box.type == "address" and b.distance(addr_box) < 300:
                                if closer_box == None:
                                    closer_box = addr_box
                                elif b.distance(closer_box) > b.distance(addr_box):
                                    closer_box = addr_box

                        if closer_box != None:
                            print( b.text, ":", closer_box.real_value)
                            label_text +=  b.text.replace(":","") +" : " + str(closer_box.real_value) + "\n"
            
            reszie_rate = 800/show_image.shape[0]
            show_image = cv2.resize(show_image, dsize=(int(show_image.shape[1]*reszie_rate), int(show_image.shape[0]*reszie_rate)))
            app = QApplication([])
            main_window = QMainWindow()
            main_window.setWindowTitle('XAI OCR')
            qiamge = QImage(show_image,show_image.shape[1], show_image.shape[0], show_image.strides[0], QImage.Format.Format_BGR888)
            frame = QLabel()
            frame.setPixmap(QPixmap.fromImage(qiamge))

            text_qlabel = QLabel()
            font1 = text_qlabel.font()
            font1.setPointSize(20)

            text_qlabel.setText(label_text)
            main_layout = QGridLayout()
            main_layout.addWidget(frame, 0, 0)
            main_layout.addWidget(text_qlabel, 1, 0)
            central_widget = QWidget()
            central_widget.setLayout(main_layout)
            main_window.setCentralWidget(central_widget)
        
            main_window.show()

            box_list = original_box_list.copy()
            x_margin += 5

            sys.exit(app.exec_())

        json_data = {}
        json_data['boundary'] = []
        for b in box_list:
            json_data['boundary'].append({
                "text" : b.text,
                "type" : b.type,
                "block_num" : b.block,
                "left_top" : (b.l, b.t),
                "right_bot" : (b.r, b.b),
            })

        # save data
        with open("label.data", 'w') as outfile:
            json.dump(json_data, outfile, indent=4)

if __name__ == '__main__':
   
    list_file = [] 
    files = list(filedialog.askopenfilenames(initialdir="./", title = "파일을 선택 해 주세요"))
    print(files)
    main(path = files)
