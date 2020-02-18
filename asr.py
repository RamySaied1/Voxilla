from htk_featio import read_htk_user_feat


class ASR:

    def __init__(self, model_file, fst_file, fst_input_labels_file):
        self.fst = FST(fst_file, fst_input_labels_file) 
        self.classifier = Classifier(model)

    def decode(self, feature_vectors, beam_width, lmweight):
        activations = classifier.eval(feature_vectors)
        return fst.decode(activations, beam_width=beam_width, lmweight=lmweight)
    
    def speech_to_text(self, voice_file, beam_width=500, lmweight = 30):
        feature_vectors = self._get_features(voice_file)
        states_seq = self._decode(feature_vectors, beam_width, lmweight)
        return ' '.join(self._get_words_from_path(states_seq))

    def speech_to_words_aligned(self, voice_file, beam_width=500, lmweight = 30):
        pass

    def _get_words_from_states_seq(self, states_seq):
        return [outlabel for _,outlabel,_ in states_seq if outlabel not in self.fst.special_symbols]

    def _get_features(self, voice_file):
        feature_vectors = read_htk_user_feat(voice_file)
        return feature_vectors