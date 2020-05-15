import collections
import numpy as np
import os.path
import re
import sys
from keras.models import model_from_json

class Classifier_cntk:
    def __init__(self,model_filename: str):
        import cntk
        self.model = None
        try:
          self.load_model(model_filename)
        except:
            raise Exception("Model didn't load successfully")

    def load_model(self,model_filename):
        self.model_filename = model_filename

        cntk_model = cntk.load_model(model_filename)

        #  First try and find output by name
        model_output = cntk_model.find_by_name('ScaledLogLikelihood')

        #  Fall back to first defined output
        if model_output is None:
            model_output = cntk_model.outputs[0]

        #  Create an object restricted to the desired output.
        cntk_model = cntk.combine(model_output)

        #  Optimized RNN models won't run on CPU without conversion.
        if 0 == cntk.use_default_device().type():
            cntk_model = cntk.misc.convert_optimized_rnnstack(cntk_model)

        self.model = cntk_model
        return self



    def stack_features(self, feature_vectors,context_frames=11):
        return np.column_stack([
            feature_vectors[np.minimum(len(feature_vectors) - 1, np.maximum(
                0, np.arange(len(feature_vectors), dtype=np.int) + d))]
            for d in range(-context_frames, context_frames + 1)
        ])


    def eval(self, feature_vectors,do_stack_features = False):
        if(do_stack_features):
            feature_vectors = self.stack_features(feature_vectors)

        if(self.model):
            return self.model.eval(feature_vectors.astype('f'))[0]
        else:
            raise  Exception("Model isn't loaded yet")

class Classifier_keras:
  def __init__(self,model_arch: str,model_weight: str, prior_file: str):
    self.model = None
    self.priori_logproba = None
    try:
        with open(model_arch, 'r') as json_file:
            json_savedModel= json_file.read()
            self.model = model_from_json(json_savedModel)
            print(self.model.summary())
        self.model.compile(loss='categorical_crossentropy',
                optimizer='adam',metrics=["categorical_accuracy"])
        self.model.load_weights(model_weight)
        self.model._make_predict_function()
        print("Model Loadded successfully")

        with open(prior_file,"r") as f:
            priori = [line.split()[1] for line in f.readlines()]
            self.priori_logproba = np.log(np.array(list(map(float,priori))))
    except:
        raise Exception("Model didn't load successfully")

  def eval(self, features):
    if(self.model):
        sample=features.reshape(1,features.shape[0],features.shape[1])
        print(sample.shape)
        pred=self.model.predict(sample)
        
        return np.log(pred[0]) - self.priori_logproba
    else:
        raise  Exception("Model isn't loaded yet")
