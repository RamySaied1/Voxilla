import sys; sys.path.append("../M2_Speech_Signal_Processing/"); sys.path.append("../performance_measure/")
from htk_featio import read_htk_user_feat
import cntk
import re
import os.path
import numpy as np
import collections

class Classifier:
    def stack_features(self,context_frames=11):
        self.feature_vectors = np.column_stack([
        self.feature_vectors[np.minimum(len(self.feature_vectors) - 1, np.maximum(0,np.arange(len(self.feature_vectors), dtype=np.int) + d))]
        for d in range(-context_frames, context_frames + 1)
    ])

    def parse_script_line(self,script_line: str, script_path: str):
        """This function parses utterance specifications from a script file.

        It is expected the format obeys the following structure:
            (utterance name).feat=(feature filename)[(start frame), (end_frame)]

        If the (feature filename) starts with the magic string ".../", then this prefix will be replaced
        with the string represented by script_path. This functionality allows scripts and acoustic feature files
        to be agnostic to their absolute location in the filesystem.
        """
        m = re.match(r'(.*)\.feat=(.*)\[(\d+),(\d+)\]', script_line)
        assert(m)
        audio_features_filename = m.group(1)
        audio_features_filepath = m.group(2)
        frame_start = int(m.group(3))
        frame_end = int(m.group(4))
        m = re.match(r'\.\.\.[/\\](.*)', audio_features_filepath)
        if m:
            audio_features_filepath = os.path.join(script_path, m.group(1))
        return audio_features_filename, audio_features_filepath, frame_start, frame_end

    def load_parameters(self,script_line: str, script_path: str):
        audio_features_filename, audio_features_filepath, frame_start, frame_end = self.parse_script_line(
            script_line, script_path)
        feature_vectors = read_htk_user_feat(audio_features_filepath)
        assert(frame_start == 0)
        assert(frame_end + 1 - frame_start == len(feature_vectors))
        self.feature_vectors, self.audio_features_filename = feature_vectors, audio_features_filename
        return audio_features_filename

    def load_model(self,model_filename: str):
        self.cntk_model = cntk.load_model(model_filename)

        #  First try and find output by name
        model_output = self.cntk_model.find_by_name('ScaledLogLikelihood')

        #  Fall back to first defined output
        if model_output is None:
            model_output = self.cntk_model.outputs[0]

        #  Create an object restricted to the desired output.
        self.cntk_model = cntk.combine(model_output)

        #  Optimized RNN models won't run on CPU without conversion.
        if 0 == cntk.use_default_device().type():
            self.cntk_model = cntk.misc.convert_optimized_rnnstack(
                self.cntk_model)

    def eval(self):
        return self.cntk_model.eval(self.feature_vectors.astype('f'))[0]
