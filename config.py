class Config(object):
    DEVELOPMENT = False
    DEBUG = False
    UPLOAD_FILE="./uploaded/wave.flac"
    RECOGNITION_DIR='./app/recognition/'
    MAX_ACTIVE_TOKENS=1500
    BEAM_WIDTH=12
    ACOUSTIC_MODEL_WEIGHT=10
    
    


class ProductionConfig(Config):
    DEVELOPMENT = False
    DEBUG = False

class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True

