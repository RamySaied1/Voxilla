import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import htk_featio as htk
import speech_sigproc as sp

inv=np.array(htk.load_ascii_vector("./FeatureExtraction/feat_invstddev.ascii"))
m=np.array(htk.load_ascii_vector("./FeatureExtraction/feat_mean.ascii"))

def wav_to_feat(wav_file,samp_rate = 16000):
    x, s = sf.read(wav_file)
    if (s != samp_rate):
        raise RuntimeError("LibriSpeech files are 16000 Hz, found {0}".format(s))

    fe = sp.FrontEnd(samp_rate=samp_rate,mean_norm_feat=True)
    feat=fe.process_utterance(x)

    m_seq=np.repeat(m[:,np.newaxis],feat.shape[1],axis=1)
    inv_seq=np.repeat(inv[:,np.newaxis],feat.shape[1],axis=1)
    
    feat=feat-m_seq
    feat=np.multiply(inv_seq,feat)
    
    return feat


