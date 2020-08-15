import os

class Config(object):
    DEVELOPMENT = False
    DEBUG = False
    MOCK = False
    print(os.path.realpath("."),"++++++++++++++++++++++++++++++++++++++++++++++")
    # DOWNLOAD_FILE=os.path.join(os.path.realpath("."),"downloaded","out.wav")
    DOWNLOAD_FILE=os.path.join(os.path.realpath("."),"downloaded")
    UPLOAD_FILE="./uploaded/wave.flac"
    RECOGNITION_DIR='./app/recognition/'
    INSERT_DIR='./app/insert/'
    MAX_ACTIVE_TOKENS=1500
    BEAM_WIDTH=12
    ACOUSTIC_MODEL_WEIGHT=10

    SAMP_RATE=16000 
    FRAME_DURATION=0.025
    FRAME_SHIFT=0.010
    
    
    SECRET_KEY = '__easyMatchSecretKey(HEO5)__'
    SQLALCHEMY_DATABASE_URI = 'mssql+pyodbc://LAPTOP-UTJ2F2N1\SQLEXPRESS/voxilla?driver=ODBC+Driver+11+for+SQL+Server'


class ProductionConfig(Config):
    DEVELOPMENT = False
    DEBUG = False

class DevelopmentConfig(Config):
    DEVELOPMENT = True
    DEBUG = True

