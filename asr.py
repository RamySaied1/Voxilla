from os.path import join as path_join
import sys
sys.path.append("./Decoder/C++/")
sys.path.append("./Classifier/")
sys.path.append("./FeatureExtraction/")
from decoder_wrapper import PyDecoder as Decoder
from classifier import Classifier
from itertools import groupby
from Wav2Feat import wav_to_feat
import numpy as np

class ASR:
    def __init__(self, model_file, fst_folder, fst_input_labels_file):
        if fst_folder and fst_folder[-1] != '/': fst_folder+="/"
        self.decoder = Decoder(fst_folder, path_join(fst_folder,fst_input_labels_file)) 
        self.classifier = Classifier(model_file)

    def speech_to_text(self, wav_file, max_active_tokens, beam_width, acoustic_model_weigh):
        features = np.transpose(np.array(self._get_features(wav_file)))
        states_seq = self._decode(features, max_active_tokens,beam_width, acoustic_model_weigh)
        return ' '.join(self._get_words_from_states_seq(states_seq))

    def speech_to_words_aligned(self, wav_file, max_active_tokens, beam_width, acoustic_model_weigh):
        pass

    def _decode(self, features, max_active_tokens, beam_width, acoustic_model_weight):
        activations = self.classifier.eval(features, do_stack_features=False)
        return self.decoder.decode(activations, max_active_tokens, beam_width, 1/acoustic_model_weight)
    
    def _get_words_from_states_seq(self, states_seq):
        # remove self loops
        states_seq = [state for i,state in enumerate(states_seq) if not i or states_seq[i][-1] != states_seq[i-1][-1]]
        return [outlabel for _,outlabel,_ in states_seq if not self.decoder.is_special_sym(outlabel)]

    def _get_features(self, wav_file):
        return wav_to_feat(wav_file)