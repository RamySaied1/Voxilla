import os

from flask import jsonify, request

from .app import app, db
from .models import Clip
from .wrappers import userRequiredJson
from .utils import createFileDirectory, randomFileName

@app.route('/clips/<int:clipId>/recognize', methods=['GET'])
@userRequiredJson()
def recognizeWords(user, clipId): # speech recognition
    userId = user["id"]
    clip = Clip.query.filter_by(id=clipId).first()
    if(clip.project.user_id != userId):
        return jsonify({"err": "you can't make action on others clips"}), 401
    request.args.get("type")
    if(clip.fileUri == None):
        return jsonify({"err": "no file exist, upload first"}), 400
    print(clip.fileUri)
    # TODO: get real alignment

    return jsonify({
        "text": [
            ["this", [0.4425, 1.0125]],
            ["is", [0.4425, 1.0125]],
            ["fake", [0.4425, 1.0125]],
        ]
    })

@app.route('/clips/<int:clipId>/synthesize', methods=['GET'])
@userRequiredJson()
def synthesizeWords(user, clipId): # speech recognition
    userId = user["id"]
    clip = Clip.query.filter_by(id=clipId).first()
    if(clip.project.user_id != userId):
        return jsonify({"err": "you can't make action on others clips"}), 401
    if(clip.fileUri == None):
        return jsonify({"err": "no file exist, upload first"}), 400
    print(clip.fileUri)
    # TODO: create the wave file here, I will simulate new file by renaming the sample file
    fpath = os.path.join(f"audios/{userId}/synthesized/{randomFileName()}")
    createFileDirectory(fpath)

    return jsonify({ "uri": "out.wav", "text": 'route accessed successfly' }), 200
