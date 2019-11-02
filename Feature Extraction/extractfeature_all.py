import soundfile as sf
from core import FeatureExtractor as FE
import argparse
import featuresIO as fio
import os

experimentsFolder = "../Experiments"

if __name__=='__main__':
    programDesc = '''
        extract features from audio files. you should specify one of 'dev, train or test' if train is set
        global mean and variance of features are calculated for use in training the acoustic model
    '''
    parser = argparse.ArgumentParser(description=programDesc)

    parser.add_argument('-d', '--dataset', help='Specify which data set type to process, must be one of {train,dev,test}', required=True, default=None)
    args = parser.parse_args()

    dataSetType = args.dataset
    shouldCalcStats = True if args.dataset == "train" else False

    audioList = os.path.join(experimentsFolder,f"lists/wav_{dataSetType}.list")
    featuresList = os.path.join(experimentsFolder,f"lists/feat_{dataSetType}.rscp")
    featureFolder = os.path.join(experimentsFolder,"feat")
    rscp_dir = "..." # [CNTK]: "relative to the location of the list of feature files"
    meanFile = os.path.join(experimentsFolder,"am/feat_mean.ascii")
    invstddev_file = os.path.join(experimentsFolder,"am/feat_invstddev.ascii")
    audioFolder = ".."

    if not os.path.exists(os.path.join(experimentsFolder,'am')):
        os.mkdir(os.path.join(experimentsFolder,'am'))


    defaultSampleRate = 16000
    featuresExtractor = FE(sampleRate=defaultSampleRate, shouldNormFeature=True, calcStats=shouldCalcStats)
    # read lines

    with open(audioList) as f:
        audioLines = f.readlines()
        audioLines = [x.strip() for x in audioLines]

    if not os.path.exists(featureFolder):
        os.makedirs(featureFolder)

    if not os.path.exists(os.path.dirname(featuresList)):
        os.makedirs(os.path.dirname(featuresList))
    outList = open(featuresList,"w")
    numProcessedFiles = 0
    for line in audioLines:

        signalId = os.path.basename(line)
        rootName, _ = os.path.splitext(signalId)
        signalFile = os.path.join(audioFolder, line)
        featureName = rootName + '.feat'
        featureFile = os.path.join(featureFolder , featureName)
        speechSignal, currentSampleRate = sf.read(signalFile)

        if (currentSampleRate != defaultSampleRate):
            raise RuntimeError("ops!!, all audio file should be 16 kHz")

        features = featuresExtractor.extract(speechSignal)
        fio.writeHtkFeatures(features, featureFile)
        feat_rscp_line = os.path.join(rscp_dir, '..', 'feat', featureName)
        print("Wrote", features.shape[1], "frames to", featureFile)
        outList.write("%s=%s[0,%d]\n" % (featureName, feat_rscp_line,features.shape[1]-1))
        numProcessedFiles += 1
    outList.close()

    print("Processed", numProcessedFiles, "files.")
    if (shouldCalcStats):
        m, p = featuresExtractor.computeStats() # m=mean, p=precision (inverse standard deviation)
        fio.writeStats(m, meanFile)
        print("Wrote global mean to", meanFile)
        fio.writeStats(p, invstddev_file)
        print("Word global inv stddev to ", invstddev_file)

