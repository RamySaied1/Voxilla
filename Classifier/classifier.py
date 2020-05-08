import collections
import numpy as np
import os.path
import re
import sys
import cntk

class Classifier:
    def __init__(self,model_filename: str):
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
