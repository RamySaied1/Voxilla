'''
    output the file that contains alignment using 40 phones format from only the textGrid files.
    iterate over all .textgrid files replace every phone with corresponding one.
    sort them with phones not with files to quickly train the hmm model
'''
from tgre import TextGrid
import os
import numpy as np
targetPhones = ["aa", "ae", "ah", "ao", "aw", "ay", "b", "ch", "d", "dh", "eh", "er", "ey", "f", "g", "hh", "ih", "iy", "jh", "k", "l", "m", "n", "ng", "ow", "oy", "p", "r", "s", "sh", "sil", "t", "th", "uh", "uw", "v", "w", "y", "z", "zh"] # 40 basic phones
targetPhones = [phone.upper() for phone in targetPhones]

# map the actual phones to target phones
phonesTargetMapper = {
    "AA0": "AA",
    "AA1": "AA",
    "AA2": "AA",
    "AE0": "AE",
    "AE1": "AE",
    "AE2": "AE",
    "AH0": "AH",
    "AH1": "AH",
    "AH2": "AH",
    "AO0": "AO",
    "AO1": "AO",
    "AO2": "AO",
    "AW0": "AW",
    "AW1": "AW",
    "AW2": "AW",
    "AY0": "AY",
    "AY1": "AY",
    "AY2": "AY",
    "EH0": "EH",
    "EH1": "EH",
    "EH2": "EH",
    "ER0": "ER",
    "ER1": "ER",
    "ER2": "ER",
    "EY0": "EY",
    "EY1": "EY",
    "EY2": "EY",
    "IH0": "IH",
    "IH1": "IH",
    "IH2": "IH",
    "IY0": "IY",
    "IY1": "IY",
    "IY2": "IY",
    "OW0": "OW",
    "OW1": "OW",
    "OW2": "OW",
    "OY0": "OY",
    "OY1": "OY",
    "OY2": "OY",
    "UH0": "UH",
    "UH1": "UH",
    "UH2": "UH",
    "UW0": "UW",
    "UW1": "UW",
    "UW2": "UW",
    "B": "B",
    "CH": "CH",
    "D": "D",
    "DH": "DH",
    "F": "F",
    "G": "G",
    "HH": "HH",
    "JH": "JH",
    "K": "K",
    "L": "L",
    "M": "M",
    "N": "N",
    "NG": "NG",
    "P": "P",
    "R": "R",
    "S": "S",
    "SH": "SH",
    "T": "T",
    "TH": "TH",
    "V": "V",
    "W": "W",
    "Y": "Y",
    "Z": "Z",
    "ZH": "ZH",

    "SIL": "SIL"
}


def getRelDir(path, back=1):
    dirs = path.split("\\")
    return "\\".join(dirs[len(dirs) - back:])

def getFilesPaths(rootDir):
    # print(os.path.basename(rootDir))
    acceptedExtensions = ["TextGrid"]
    return [os.path.abspath(os.path.join(root, filename)) for root, subdirs, files in os.walk(rootDir) if len(subdirs) == 0 for filename in files if (filename.split(".")[-1] in acceptedExtensions)]
    
def main(rootDir, outDir):
    print("number of phones appeared in mapping:", len(set(phonesTargetMapper.values())), ",", "number of target phones:", len(targetPhones))
    phonesLocations = {} # {"ph1": [(fpath1, xmin1, xmax1), (fpath2, xmin2, xmax2), ...], "ph2": [(fpath1, xmin1, xmax1), (fpath2, xmin2, xmax2), ...], ...}
    for targetPhone in targetPhones:
        phonesLocations[targetPhone] = []
    phonesLocations['others'] = []

    for filePath in getFilesPaths(rootDir):
        textGrid = TextGrid.from_file(filePath)
        phonesTier = next(filter(lambda interval: interval.name == "phones", textGrid), None)
        if (not phonesTier):
            print("the file has no phones tier !!")
        for interval in phonesTier:
            actualPhone = interval.text.upper()
            targetPhone = phonesTargetMapper.get(actualPhone, 'others')
            relDir = getRelDir(filePath, back=3)
            phonesLocations[targetPhone].append((relDir, interval.xmin, interval.xmax))
    for targetPhone in targetPhones:
        np.savetxt(f"{os.path.join(outDir, targetPhone)}.txt", phonesLocations[targetPhone], fmt='%a')

if __name__ == "__main__":
    main("data/librispeech_alignments-trainclean-100", "data/data-generated")