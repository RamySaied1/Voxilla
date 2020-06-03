import os

class Config(object):
    DEVELOPMENT = False
    DEBUG = False
    print(os.path.realpath("."),"++++++++++++++++++++++++++++++++++++++++++++++")
    DOWNLOAD_FILE=os.path.join(os.path.realpath("."),"downloaded","out.wav")
    UPLOAD_FILE="./uploaded/wave.flac"
    RECOGNITION_DIR='./app/recognition/'
    INSERT_DIR='./app/insert/'
    MAX_ACTIVE_TOKENS=1500
    BEAM_WIDTH=12
    ACOUSTIC_MODEL_WEIGHT=10

    SAMP_RATE=16000 
    FRAME_DURATION=0.025
    FRAME_SHIFT=0.010
    
    
    


class ProductionConfig(Config):
    DEVELOPMENT = False
    DEBUG = False

class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True

