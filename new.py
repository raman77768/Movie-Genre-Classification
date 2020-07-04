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

#print("loaded")

def test(i):
  train = pd.read_csv("C:/Users/Raman/Desktop/Moviegenre/Dataset/train.csv")
  train=train[0:7254]
  classes = np.array(train.columns[2:])
  img = image.load_img(i,target_size=(400,400,3))
  img = image.img_to_array(img)
  img = img/255
  res=[]
  proba = model.predict(img.reshape(1,400,400,3))
  top = np.argsort(proba[0])[:-4:-1]
  for i in range(3):
    #print("{}".format(classes[top[i]])+" ({:.3})".format(proba[0][top[i]]))
    res.append(("{}".format(classes[top[i]])+" ({:.3})".format(proba[0][top[i]])))
  return res

def modelloading():
  return load_model('C:/Users/Raman/Desktop/Moviegenre/Pretrained_model/model.h5')
print("model loaded")
model=modelloading()

def image_fetch(inp):
  try:
    #inp= input()
    query=inp+"movie"
    res=[j for j in search(query, tld="com", num=1, stop=1, pause=2)]
    #print(res)
    r = requests.get(res[0])
    content = r.content
    soup = BeautifulSoup(content,features= "html.parser",from_encoding="iso-8859-1")
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
      with open("testimage.jpg",'wb') as f:f.write(r.content)
    elif flag==1:
      r=requests.get(res[0])
      with open("testimage.jpg",'wb') as f:f.write(r.content)
    elif flag==2:
      s="https:/"+s
      r = requests.get(s) 
      with open("testimage.jpg",'wb') as f:f.write(r.content)
  except:
    return "Not Found"
    pass
  try:
    result=test("testimage.jpg")
    return result
  except:return "Not Found"

def recommend(inp):
  query2 = inp+"imdb"
  res2=[j for j in search(query2, tld="com", num=1, stop=1, pause=2)]
  #print(res2)
  soup= BeautifulSoup(urlopen(res2[0]).read())
  #print(soul)
  div = soup.findAll('div', attrs = {'class' : 'rec_item'})
  #print(div[1])
  title=[]
  images=[]
  links=[]
  for i in range(5):
    for ptag in div[i].find_all('img'):
        images.append(ptag['loadlate'])
        title.append(ptag['title'])
        #print(ptag['title'])
    for ptag in div[i].find_all('a'):
      links.append('https://www.imdb.com'+ptag['href'])
  recommended_list=[]
  for i in range(5):
  #r=requests.get(images[i])
  #with open("recommended/image{}.jpg".format(i),'wb') as f:f.write(r.content)
    recommended_list.append([title[i],links[i]])
  return recommended_list