import os

from flask import jsonify, request, send_file

from .app import app
from .wrappers import userRequiredJson

@app.route('/insert$', methods = ['POST'])
def insert_demo():
    body = request.json
    print(body)
    return jsonify({ "text": 'route accessed successfly' }), 200

@app.route('/recognize$')
def recognize_demo():
    return jsonify({
        "text": [
            ["this", [0.4425, 1.0125]],
            ["is1", [0.4425, 1.0125]],
            ["is2", [0.4425, 1.0125]],
            ["is3", [0.4425, 1.0125]],
            ["is4", [0.4425, 1.0125]],
            ["is5", [0.4425, 1.0125]],
            ["is6", [0.4425, 1.0125]],
            ["is7", [0.4425, 1.0125]],
            ["is8", [0.4425, 1.0125]],
            ["is9", [0.4425, 1.0125]],
            ["is0", [0.4425, 1.0125]],
            ["is11", [0.4425, 1.0125]],
            ["is12", [0.4425, 1.0125]],
            ["is21", [0.4425, 1.0125]],
            ["fake", [0.4425, 1.0125]],
        ]
    })

@app.route('/download$')
def download_demo():
        return send_file(f"audios/1/out.wav", as_attachment=True)

@app.route('/upload$', methods=["POST"])
def upload_demo():
        try:
            f = request.files
            # print(f)
            f = f['file']
            # print(os.path.abspath(os.curdir))
            f.save("demo.flac")
            return jsonify({ "message": "file uploaded sussessfully" }), 200
        except Exception as e:
            return jsonify({ "message": e.message }), 404