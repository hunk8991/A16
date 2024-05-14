import tensorflow as tf
import numpy as np
import cv2
import tkinter as tk
from PIL import  ImageTk, Image

graph_def = tf.compat.v1.GraphDef()
labels = []

# 載入訓練模型及標籤
filename = "/Users/hunk/Desktop/C03 Project/Training/model.pb"
labels_filename = "/Users/hunk/Desktop/C03 Project/Training/labels.txt"

# Import the TF graph
with tf.io.gfile.GFile(filename, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

# 建立標籤列
with open(labels_filename, 'r') as lf:
    for l in lf:
        labels.append(l.strip())

captrue = cv2.VideoCapture(0) # 開啟相機，0為預設內建相機
captrue.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 設置影像參數
captrue.set(3,350) # 像素
captrue.set(4,500) # 像素

img_viode = '/Users/hunk/Desktop/C03 Project/VideoCapture/identify.jpg' # 影像存放位置

def check():
    global captrue
    if captrue.isOpened(): # 判斷相機是否有開啟
        open_c()
    else:
        captrue = cv2.VideoCapture(0) 
        captrue.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')) # 設置影像參數
        captrue.set(3,350) # 像素
        captrue.set(4,500) # 像素
        open_c()

def open_c():
    try:
        global s
        ret,frame = captrue.read() # 取得相機畫面
        cv2.imwrite(img_viode,frame) # 儲存圖片
        img_right = ImageTk.PhotoImage(Image.open(img_viode)) # 讀取圖片
        label_right.imgtk = img_right # 換圖片
        label_right.config(image = img_right) # 換圖片
        s = label_right.after(1, open_c) # 持續執行open方法，1000為1秒
    except:
        print('error')
        captrue.release() # 關閉相機
        label_right.after_cancel(s) # 結束拍攝
        check()

def close():
    captrue.release() # 關閉相機
    label_right.after_cancel(s) # 結束拍攝
    
def identify():
    label_right.config(image = cover) # 換圖片
    print(labels) # 輸出所有label

    #載入並設定圖片參數
    test_image = cv2.imread('/Users/hunk/Desktop/C03 Project/VideoCapture/identify.jpg',cv2.IMREAD_COLOR)
    new_image = cv2.resize(test_image,(224, 224),interpolation=cv2.INTER_AREA)
    new_image = np.array(new_image).reshape(224,224,3)

    # These names are part of the model and cannot be changed.
    output_layer = 'loss:0'
    input_node = 'Placeholder:0'
    #detect image
    sess = tf.compat.v1.Session()  
    
    prob_tensor = sess.graph.get_tensor_by_name(output_layer)
    predictions = sess.run(prob_tensor, {input_node: [new_image] })
    
    # Print the highest probability label
    highest_probability_index = np.argmax(predictions)
    bag = labels[highest_probability_index]
    print(bag)   
        
    def top():
        top = tk.Toplevel()
        
        text1 = tk.Text(top, width=50, height=26)
        top_img = Image.open("/Users/hunk/Desktop/C03 Project/intro/"+bag+"/"+bag+".jpg")
        photo = ImageTk.PhotoImage(top_img)
        text1.insert(tk.END, '\n')
        text1.image_create(tk.END, image=photo)
        text1.pack(side=tk.LEFT)

        text2 = tk.Text(top, width=80, height=26)
        text2.tag_configure('name', foreground='#FF9933', font=('Verdana', 30, 'bold'), justify="center")
        text2.tag_configure('intro', foreground='#666666', font=('Tempus Sans ITC', 16, 'bold'))
        text2.insert(tk.END, bag+'\n', 'name')
        f = open("/Users/hunk/Desktop/C03 Project/intro/"+bag+"/"+bag+".txt", "r")
        intro = f.read()
        text2.insert(tk.END, '\n'+intro, 'intro')
        text2.pack(side=tk.LEFT)
        
        top.mainloop()
    top()
    
window = tk.Tk() 
window.title('包包辨識') 
window.geometry('600x500')

cover = ImageTk.PhotoImage(Image.open('/Users/hunk/Desktop/C03 Project/VideoCapture/cover.jpg'))

label_right = tk.Label(window, height=360, width=557, bg='gray94', fg='blue', image=cover)

button_1 = tk.Button(window,text = '開啟', bd=3, height=3, width=8, bg='gray94', foreground='#666666', command=check)
button_2 = tk.Button(window,text = '拍攝', bd=3, height=3, width=8, bg='gray94', foreground='#666666', command=close)
button_3 = tk.Button(window,text = '辨識', bd=3, height=3, width=8, bg='gray94', foreground='#666666', command=identify)

label_right.grid(row=1, column=0, padx=20, pady=20, sticky="nw") 

button_1.grid(row=1, column=0, padx=100, pady=410, sticky="nw")  
button_2.grid(row=1, column=0, padx=250, pady=410, sticky="nw")
button_3.grid(row=1, column=0, padx=400, pady=410, sticky="nw")

window.mainloop()