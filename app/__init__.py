from flask import Flask
from app.recognition.controller import recognition
from app.upload.controller import upload


def create_app(db_uri='any'):

    app = Flask(__name__)
    app.config.from_object('config.DevelopmentConfig')
    app.register_blueprint(recognition, url_prefix='/recognition')
    app.register_blueprint(upload, url_prefix='/upload')


    return app