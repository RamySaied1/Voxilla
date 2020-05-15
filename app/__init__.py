from flask import Flask
from app.recognition.controller import recognition


def create_app(db_uri='any'):

    app = Flask(__name__)
    app.config.from_object('config.DevelopmentConfig')
    app.register_blueprint(recognition, url_prefix='/recognition')

    return app