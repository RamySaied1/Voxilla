from config import Config
Config.MOCK = True
from app import create_app
from flask_cors import CORS
from flask import jsonify, request, current_app, send_file

import os
if __name__ == '__main__':
    app = create_app()

    @app.route('/recognition$getalignment/', methods=['GET'])
    def mygetAlignment():
        return jsonify({
            "text": [
                ["this", [0.4425, 1.0125]],
                ["is", [1.0125, 1.4725]],
                ["fake", [1.5, 1.7]],
            ]
        })

    @app.route('/upload$', methods = ['POST'])
    def myupload():
        try:
            f = request.files
            print(f)
            f = f['file']
            print(os.path.abspath(os.curdir))
            f.save(current_app.config["UPLOAD_FILE"])
            return jsonify({ "message": "file uploaded sussessfully" }), 200
        except Exception as e:
            return jsonify({ "message": e.message }), 404

    
    @app.route('/insert$', methods = ['POST'])
    def myinsert():
        body = request.json
        print(body)
        print("save to", current_app.config["DOWNLOAD_FILE"])
        return jsonify({ "text": 'route accessed successfly' }), 200
    
    @app.route('/download$',methods = ['GET','POST'])
    def mydownload():
        try:
            print(current_app.config["DOWNLOAD_FILE"])
            return send_file(current_app.config["DOWNLOAD_FILE"], as_attachment=True)
        except Exception as e:
            return jsonify({ "message": e.message }), 404
    CORS(app)
    app.run()