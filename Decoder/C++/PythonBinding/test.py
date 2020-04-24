from decoder_wrapper import PyDecoder as Decoder
import numpy as np

decoder = Decoder("../../Graphs/200k-vocab/","../../Graphs/200k-vocab/labels.ciphones",1000,12.)

activations = np.loadtxt("../../Activations/BLSTM1_activations/1089-134686-0000_BLSTM1_activations.txt")
print(decoder.decode(activations,1./9))