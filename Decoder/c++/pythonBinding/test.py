from decoder_wrapper import PyDecoder as Decoder
import numpy as np

decoder = Decoder(b"../../DecodingGraph-large.txt",b"../../labels.ciphones",500)

activations = np.loadtxt("activations.txt")
print(decoder.decode(activations,30.))