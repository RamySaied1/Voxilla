from flask import Flask
from flask_cors import CORS
from app.recognition.controller import recognition,getTranscript,getAlignment
from app.upload.controller import upload,upload_file
from app.insert.controller import insert,generate_word
from app.download.controller import download,download_file

def create_app(db_uri='any'):
    app = Flask(__name__)
    app.config.from_object('config.DevelopmentConfig')
    app.route('/insert', methods = ['POST'])(generate_word)
    app.route('/download', methods = ['GET','POST'])(download_file)
    app.route('/recognition/gettranscript', methods=['GET'])(getTranscript)
    app.route('/recognition/getalignment', methods=['GET'])(getAlignment)
    app.route('/upload', methods=['POST'])(upload_file)
    CORS(app)
    #app.register_blueprint(recognition, url_prefix='/recognition')
    #app.register_blueprint(upload, url_prefix='/upload')
    #app.register_blueprint(insert, url_prefix='/insert')
    #app.register_blueprint(download,url_prefix='/download')

    return app