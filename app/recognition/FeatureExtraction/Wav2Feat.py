import os
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
import htk_io as htk
from MfccFeatures import MFCC

from config import Config
inv=np.array(htk.load_ascii_vector(Config.RECOGNITION_DIR+"FeatureExtraction/feat_invstddev.ascii"))
m=np.array(htk.load_ascii_vector(Config.RECOGNITION_DIR+"FeatureExtraction/feat_mean.ascii"))
SAMP_RATE= Config.SAMP_RATE
FRAME_DURATION= Config.FRAME_DURATION
FRAME_SHIFT= Config.FRAME_SHIFT


def wav_to_feat(wav_file,samp_rate = SAMP_RATE,frame_duration=FRAME_DURATION,frame_shift=FRAME_SHIFT):
    x, s = sf.read(wav_file)
    if (s != samp_rate):
        raise RuntimeError("Our dataset files have sampling rate "+str(SAMP_RATE)+"Hz, found {0}".format(s))

    fe = MFCC(sampling_rate=samp_rate,normalize_feat=True, frame_duration=frame_duration, frame_shift=frame_shift)
    feat=fe.calculate_features(x)

    m_seq=np.repeat(m[:,np.newaxis],feat.shape[1],axis=1)
    inv_seq=np.repeat(inv[:,np.newaxis],feat.shape[1],axis=1)
    
    feat=feat-m_seq
    feat=np.multiply(inv_seq,feat)
    
    return feat


