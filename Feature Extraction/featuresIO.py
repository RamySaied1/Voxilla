import struct
import numpy as np
import sys

def writeHtkFeatures(x, fileName='filename'):
    frameShift = 100000 # 0.01ms shift
    numDimension = x.shape[0]
    numFrames = x.shape[1]
    hdr = struct.pack( '>iihh', # big-endian
        numFrames, frameShift, 4*numDimension, 9)

    outFile = open(fileName, 'wb')
    outFile.write(hdr)

    for t in range(numFrames):
        frame = np.array(x[:,t],'f')
        if sys.byteorder == 'little':
            frame.byteswap(True)
        frame.tofile(outFile)

    outFile.close()

def readHtkFeatures(fileName='filename'):
    featuresFile = open(fileName,'rb')
    hdr = featuresFile.read(12)
    numSamples, samplePeriod, sampleSize, numFeatures = struct.unpack(">IIHH", hdr)
    if numFeatures != 9:
        raise RuntimeError('file not supported, expect features to be 9, not found !!')

    numDimension = sampleSize//4

    features = np.zeros([numSamples, numDimension],dtype=float)
    for t in range(numSamples):
        features[t,:] = np.array(struct.unpack('>' + ('f' * numDimension), featuresFile.read(sampleSize)),dtype=float)

    featuresFile.close()
    return features


def writeStats(x,fileName='filename'):
    outFile = open(fileName,'w')
    for t in range(0, x.shape[0]):
        outFile.write(f"{x[t]}\n")
    outFile.close()