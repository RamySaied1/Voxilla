import os

class Config(object):
    DEVELOPMENT = False
    DEBUG = True
    SERVICES_MOCK = False
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
    
    
    SECRET_KEY = '__voxilla is the best__'
    SQLALCHEMY_DATABASE_URI = 'mssql+pyodbc://LAPTOP-UTJ2F2N1\SQLEXPRESS/voxilla?driver=ODBC+Driver+11+for+SQL+Server'


class ProductionConfig(Config):
    DEVELOPMENT = False
    DEBUG = False

class DevelopmentConfig(Config):
    DEVELOPMENT = True
    debug = True

