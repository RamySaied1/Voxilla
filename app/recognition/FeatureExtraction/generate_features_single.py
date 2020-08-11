import soundfile as sf
import os
import numpy as np
from MfccFeatures import MFCC
import matplotlib.pyplot as plot
import htk_io as htk



wav_file='./app/recognition/FeatureExtraction/84-121123-0002.flac'
feat_file='./app/recognition/FeatureExtraction/84-121123-0002.feat'


samp_rate = 16000

audio, s = sf.read(wav_file)
if (s != samp_rate):
    raise RuntimeError("we only work with files are 16000 Hz, found {0}".format(s))

fe = MFCC(sampling_rate=samp_rate,normalize_feat=True)


features = fe.calculate_features(audio)



# plot mel filterbank
for i in range(0, fe.num_mel_filters):
    plot.plot(fe.filterbank[i, :])
plot.title('filterbank')
plot.savefig('filterbank.png', bbox_inches='tight')
plot.close()


# plot waveform
plot.plot(audio)
plot.title('wave_form')
plot.savefig('waveform.png', bbox_inches='tight')
plot.close()


# plot log mel spectrum (fbank)
plot.imshow(features, origin='lower', aspect=4) 
plot.title('log mel filterbank features (fbank)')
plot.savefig('fbank.png', bbox_inches='tight')
plot.close()

htk.write_user_feat(features, feat_file)
print("i have written {0} frames to {1}".format(features.shape[1], feat_file))

