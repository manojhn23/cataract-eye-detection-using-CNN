import os

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.python.keras.models import load_model
#from cataract.modEye import model


def check(res):
    #p1=["benign","malignant"]
    p2=["normal","cataract",]
    path=p2
    model = load_model('model.h5',compile=False)
    #model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    pred=model.predict(res)
    res=np.argmax(pred)
    res=path[res]
    print(res)

def convert_img_to_tensor2(fpath):
    img = cv2.imread(fpath)
    img = cv2.resize(img,(256,256))
    res = img_to_array(img)
    res = np.array(res, dtype=np.float16)/ 255.0
    res = res.reshape(-1,256,256,3)
    res = res.reshape(1,256,256,3)
    return res

#t1=f"{dataset}/ModerateDemented/27 (2).jpg"
t2="C:\\Users\\manoj\\PycharmProjects\\python1\\cataract\\dataset\\2_cataract\\cataract_007.png"

res=convert_img_to_tensor2(t2)

check(res)

# folder_dir = "D:\\eye\\eyedataset\\2_cataract"
# for images in os.listdir(folder_dir):
#     t2 = f"{folder_dir}//{images}"
#     res = convert_img_to_tensor2(t2)
#     print(images, check(res))