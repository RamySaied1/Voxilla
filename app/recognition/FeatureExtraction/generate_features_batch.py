import soundfile as sf
from MfccFeatures import MFCC
import argparse
import htk_io as htk
import os

data_folder = "../Experiments"

data_set="train"

if __name__=='__main__':
    
    if data_set == "train":
        compute_global_stats=True
    else:
        compute_global_stats=False
    wave_list = os.path.join(data_folder,"lists/wav_{0}.list".format(data_set))
    feature_list = os.path.join(data_folder,"lists/feat_{0}.rscp".format(data_set))
    features_dir = os.path.join(data_folder,"feat")
    rscp_dir = "..."
    mean_file = os.path.join(data_folder,"am/features_mean.ascii")
    invstddev_file = os.path.join(data_folder,"am/features_invstddev.ascii")
    wave_dir = ".."


    samp_rate = 16000
    fe = MFCC(sampling_rate=samp_rate, normalize_feat=True, compute_global_stats=compute_global_stats)
    # read lines

    with open(wave_list) as f:
        wav_files = f.readlines()
        wav_files = [x.strip() for x in wav_files]

    if not os.path.exists(os.path.dirname(feature_list)):
        os.makedirs(os.path.dirname(feature_list))

    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    out_list = open(feature_list,"w")
    files_count = 0
    for line in wav_files:

        wave_name = os.path.basename(line)
        root_name, wav_ext = os.path.splitext(wave_name)
        wave_file = os.path.join(wave_dir, line)
        feat_name = root_name + '.feat'
        feature_filename = os.path.join(features_dir , feat_name)
        audio, sam = sf.read(wave_file)

        if (sam != samp_rate):
            raise RuntimeError(" assume 16 kHz audio files!")

        feat_generated = fe.calculate_features(audio)
        htk.write_user_feat(feat_generated, feature_filename)
        feat_rscp_line = os.path.join(rscp_dir, '..', 'feat', feat_name)
        print("Wrote", feat_generated.shape[1], "frames to", feature_filename)
        out_list.write(" %s = %s [ 0 , %d ]\n" % (feat_name, feat_rscp_line,feat_generated.shape[1]-1))
        files_count += 1
    out_list.close()

    print("finish processing of ", files_count, "files.")
    if (compute_global_stats):
        m, p = fe.calculate_statistics() # m=mean, p= (inverse standard deviation)
        htk.write_stats(m, mean_file)
        print("Save global mean to", mean_file)
        htk.write_stats(p, invstddev_file)
        print("Save global inv stddev to ", invstddev_file)






