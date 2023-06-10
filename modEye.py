from tensorflow.keras.layers import Conv2D,Flatten,Dense,MaxPooling2D
from os import listdir
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
from sklearn.model_selection import train_test_split
r2="C:\\Users\\manoj\\PycharmProjects\\python1\\Eye\\dataset"
r1="C:\\Users\\manoj\\PycharmProjects\\python1\\Eye\\dataset\\2_cataract"
import numpy as np
np.random.seed(42)
root_dir=r2
def convert_img_to_tensor(fpath):
    img = cv2.imread(fpath)
    img = cv2.resize(img,(256,256))
    res = img_to_array(img)
    # res = np.array(res, dtype=np.float16)/ 255.0
    # res = res.reshape(-1,256,256,3)
    # res = res.reshape(1,256,256,3)
    return res
def get_img_data_and_label(root_dir):
  dire=listdir(root_dir)
  image_dataset=[]
  image_label=[]

  binary_label=[]
  i=0
  for subdir in dire:
    binary_label.append(i)
    i+=1
  index=0
  for subdir in dire:
    eye_img_list=listdir(f"{root_dir}/{subdir}")
    for imgfile in eye_img_list:
          filepath=f"{root_dir}/{subdir}/{imgfile}"
          #print(filepath)
          res=convert_img_to_tensor(filepath)
          image_dataset.append(res)
          image_label.append(binary_label[index])
    index+=1
  return image_dataset,image_label

image_dataset,image_label= get_img_data_and_label(root_dir)

len(image_label),len(image_dataset),image_dataset[0].shape

xtrain,xtest,ytrain,ytest=train_test_split(image_dataset,image_label,test_size=0.2,random_state=100)

from tensorflow.keras.utils import to_categorical
ytrain=to_categorical(ytrain)
ytest=to_categorical(ytest)

len(ytest)

xtrain = np.array(xtrain, dtype=np.float16)/ 255.0
xtrain = xtrain.reshape(-1,256,256,3)
xtest = np.array(xtrain, dtype=np.float16)/ 255.0
xtest = xtrain.reshape(-1,256,256,3)

data={}
data["xtr"]=xtrain
data["ytr"]=ytrain
data["xts"]=xtest
data["yts"]=ytest

for k in data:
  print(k)

import pickle

xtrain.shape,ytrain.shape

ytrain[19]

file =open('C:\\Users\\manoj\\PycharmProjects\\python1\\cataract\\data.dat','wb')
pickle.dump(data,file)
file.close

file=open('C:\\Users\\manoj\\PycharmProjects\\python1\\cataract\\data.dat','rb')
data=pickle.load(file)

from tensorflow.keras.models import Sequential

model = Sequential()

model.add(Conv2D(32,(3,3), activation = 'relu' , input_shape = (256,256,3)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(32,(3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dense(2, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam' , metrics = ['accuracy'])

model.fit(xtrain,ytrain,epochs=10,batch_size=32)
model.save("model.h5")