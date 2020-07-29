!unzip Multi_Label_dataset.zip
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
%matplotlib inline
import gc

train = pd.read_csv("train.csv")
print(train.head()) 
print(len('Multi_Label_dataset/Images/'))
gc.collect()
train

train_image = []
for i in train[:]["Id"]:
    img = image.load_img('Multi_Label_dataset/Images/{}'.format(i)+'.jpg',target_size=(400,400,3))
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)

gc.collect()

del img
gc.collect()

X = np.array(train_image)
y = np.array(train.drop(['Id', 'Genre'],axis=1))
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(400,400,3)))
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
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=64)

model.save('model.h5')

gc.collect()

model=load_model("model.h5")
def test(testing_photo):
  img = image.load_img(testing_photo,target_size=(400,400,3))
  img = image.img_to_array(img)
  img = img/255
  classes = np.array(train.columns[2:])
  proba = model.predict(img.reshape(1,400,400,3))
  top_3 = np.argsort(proba[0])[:-4:-1]
  res=[]
  for i in range(3):
      print("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))
  return res

res=test('/content/1-thapped.jpeg')