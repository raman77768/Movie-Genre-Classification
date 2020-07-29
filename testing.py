import pandas as pd
from keras.models import load_model
import keras
import numpy as np
from keras.preprocessing import image
def test(testing_photo):
  train = pd.read_csv("train.csv")
  model=load_model("model.h5")
  img = image.load_img(testing_photo,target_size=(400,400,3))
  img = image.img_to_array(img)
  img = img/255
  classes = np.array(train.columns[2:])
  proba = model.predict(img.reshape(1,400,400,3))
  top_3 = np.argsort(proba[0])[:-4:-1]
  res=[]
  for i in range(3):
      res.append("{}".format(classes[top_3[i]])+" ({:.3})".format(proba[0][top_3[i]]))
  return '\n'.join(res)
