from flask import Flask
from app.recognition.controller import recognition
from app.upload.controller import upload
from app.insert.controller import insert
from app.download.controller import download 

def create_app(db_uri='any'):
    app = Flask(__name__)
    app.config.from_object('config.DevelopmentConfig')
    app.register_blueprint(recognition, url_prefix='/recognition')
    app.register_blueprint(upload, url_prefix='/upload')
    app.register_blueprint(insert, url_prefix='/insert')
    app.register_blueprint(download,url_prefix='/download')

    return app