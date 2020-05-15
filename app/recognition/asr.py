from os.path import join as path_join
import sys
from config import Config
sys.path.append(Config.RECOGNITION_DIR+"Decoder/C++/PythonBinding/")
sys.path.append(Config.RECOGNITION_DIR+"Classifier/")
sys.path.append(Config.RECOGNITION_DIR+"FeatureExtraction/")
sys.path.append(Config.RECOGNITION_DIR+"ForcedAlignmnet/")
from decoder_wrapper import PyDecoder as Decoder
from classifier import Classifier_keras as Classifier
from Transcript import Transcript
from itertools import groupby
from Wav2Feat import wav_to_feat
import numpy as np

class ASR:
    def __init__(self, model_arch, model_weight, model_priori_proba_file, fst_folder, acoustic_model_labels_file, words_lexicon_file = Config.RECOGNITION_DIR+"ForcedAlignmnet/words_lexicon.txt", phones_lexicon_file = Config.RECOGNITION_DIR+"ForcedAlignmnet/phones_lexicon.txt"):
        if fst_folder and fst_folder[-1] != '/': fst_folder+="/"
        self.decoder = Decoder(fst_folder, acoustic_model_labels_file) 
        self.classifier = Classifier(model_arch, model_weight, model_priori_proba_file)
        self.trans = Transcript(words_lexicon_file, phones_lexicon_file)

    def speech_to_text(self, wav_file, max_active_tokens, beam_width, acoustic_model_weigh, include_alignment = False):
        features = np.transpose(np.array(self._get_features(wav_file)))
        states_seq = self._decode(features, max_active_tokens,beam_width, acoustic_model_weigh)
        if(not include_alignment):
           states_seq = self._remove_selfloops(states_seq)
           return ' '.join(self._get_words_from_states_seq(states_seq))
        else:
           frames_count = len(features)
           return self.trans.get_transcript(states_seq,frames_count)

    def _decode(self, features, max_active_tokens, beam_width, acoustic_model_weight):
        activations = self.classifier.eval(features)
        return self.decoder.decode(activations, max_active_tokens, beam_width, 1/acoustic_model_weight)
    
    def _remove_selfloops(self,states_seq):
        return [state for i,state in enumerate(states_seq) if not i or states_seq[i][-1] != states_seq[i-1][-1]] # remove self loops

    def _get_words_from_states_seq(self, states_seq):
        return [outlabel for _,outlabel,_ in states_seq if not self.decoder.is_special_sym(outlabel)]

    def _get_features(self, wav_file):
        return wav_to_feat(wav_file)