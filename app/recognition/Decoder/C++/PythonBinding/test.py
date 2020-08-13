from decoder_wrapper import PyDecoder as Decoder
import numpy as np

decoder = Decoder("../../Graphs/200k-vocab/","../../Graphs/200k-vocab/labels.ciphones","G.fst")

activations = np.loadtxt("./activations.txt")
print(decoder.decode(activations,1000,10.,1./10,4))