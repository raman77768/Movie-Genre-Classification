# -*- coding: utf-8 -*-
"""movie_classifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ME2jLIFF4pmO_GvTufqvV_sx6KgN4JFj
"""

!unzip '/content/drive/My Drive/Multi_Label_dataset.zip'

# Commented out IPython magic to ensure Python compatibility.
!pip install BeautifulSoup4
!pip install google
!pip install requests
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
# %matplotlib inline
import gc
import IPython
import bs4
from bs4 import BeautifulSoup
from googlesearch import search 
import requests
from urllib.request import urlopen
import re
import os
#os.makedirs('recommended')

train = pd.read_csv("drive/My Drive/movie classifier/train.csv")
#print(train.head()) 
#print(len('Multi_Label_dataset/Images/'))
gc.collect()
train=train[0:7254]

#train.columns[3:]
counts=train.sum(axis=0,skipna=True)
counts=counts[3:]
counts
plt.plot(counts[0:24])
fig = plt.gcf()
fig.set_size_inches(24, 10)
fig.savefig('test2png.png', dpi=100)

train_image = []
for i in train[0:7254]["Id"]:
    img = image.load_img('Multi_Label_dataset/Images/{}'.format(i)+'.jpg',target_size=(260,260,3))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)

gc.collect()

X = np.array(train_image)
y = np.array(train.drop(['Id', 'Genre'],axis=1))
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(260,260,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(25, activation='sigmoid'))

model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), batch_size=64)

classes = np.array(train.columns[2:])

def test(i):
  img = image.load_img(i,target_size=(400,400,3))
  img = image.img_to_array(img)
  img = img/255
  res=[]
  proba = model.predict(img.reshape(1,400,400,3))
  top = np.argsort(proba[0])[:-4:-1]
  for i in range(3):
    print("{}".format(classes[top[i]])+" ({:.3})".format(proba[0][top[i]]))
    res.append(("{}".format(classes[top[i]])+" ({:.3})".format(proba[0][top[i]])))
  plt.imshow(img)
  return res

model=load_model('drive/My Drive/movie classifier/model.h5')

try:
  query = "karwaan movie"
  res=[j for j in search(query, tld="com", num=1, stop=1, pause=2)]
  print(res)
  r = requests.get(res[0])
  content = r.content
  soup = BeautifulSoup(content, "html.parser")
  try:
    if 'svg' in str(soup.findAll('img')[0]):
      if 'svg' in str(soup.findAll('img')[1]):images = soup.findAll('img')[2]
      else:images = soup.findAll('img')[1]
    else:images = soup.findAll('img')[0]
    s=images['src']
    flag=0
    print(s)
  except(IndexError):
    flag=1
    pass

  if "https:" in s:flag=1
  elif s[1]!='/':flag=2
  if flag==0:
    s="https:"+s
    r = requests.get(s) 
    with open("image.jpg",'wb') as f:f.write(r.content)
  elif flag==1:
    r=requests.get(res[0])
    with open("image.jpg",'wb') as f:f.write(r.content)
  elif flag==2:
    s="https:/"+s
    r = requests.get(s) 
    with open("image.jpg",'wb') as f:f.write(r.content)
except:
  print("Invalid Query")
  pass

result=test('image.jpg')

query2 = "karwaan imdb"
res2=[j for j in search(query2, tld="com", num=1, stop=1, pause=2)]
print(res2)
soup= BeautifulSoup(urlopen(res2[0]).read())
#print(soul)
div = soup.findAll('div', attrs = {'class' : 'rec_item'})
#print(div[1])
title=[]
image=[]
links=[]
for i in range(5):
  for ptag in div[i].find_all('img'):
      image.append(ptag['loadlate'])
      title.append(ptag['title'])
      #print(ptag['title'])
  for ptag in div[i].find_all('a'):
    links.append('https://www.imdb.com'+ptag['href'])

for i in range(5):
  #r=requests.get(image[i])
  #with open("recommended/image{}.jpg".format(i),'wb') as f:f.write(r.content)
  print(title[i],links[i])