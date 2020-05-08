import numpy as np
from keras.models import load_model


class Classifier:
  def __init__(self,model_filename: str):
    self.model = None
    try:
      self.model = load_model(model_filename)
    except:
      raise Exception("Model didn't load successfully")

  def load_model(self,model_filename):
    try:
      self.model = load_model(model_filename)
    except:
      raise Exception("Model didn't load successfully")


  def eval(self, features):
    if(self.model):
      sample=features.reshape(1,features.shape[0],features.shape[1])
      pred=self.model.predict(sample)
      return pred[0]
    else:
      raise  Exception("Model isn't loaded yet")