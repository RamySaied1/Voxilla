from datetime import datetime, timedelta
from flask import jsonify, request
from jwt import encode as jwtEncode
from werkzeug.security import generate_password_hash, check_password_hash

from .app import app, db
from .models import User
from .wrappers import userRequiredJson

@app.route('/signup', methods=['post'])
def signup():
    data_json = request.get_json()
    hashedPassword = generate_password_hash(data_json['password'], method='sha256')
    interestingData = ('username', 'email')
    mappingInDb =     ('username', 'email')
    d = {dbKey: data_json[apiKey] for apiKey, dbKey in zip(interestingData, mappingInDb)}
    newUser = User(**d, password=hashedPassword)
    db.session.add(newUser)
    db.session.commit()
    token = jwtEncode({
        'userId': newUser.id,
        'username': newUser.username,
        'email': newUser.email,
        'exp': datetime.utcnow() + timedelta(hours=48)
    }, app.config['SECRET_KEY']).decode('UTF-8')
    return jsonify({'token': token, 'username': newUser.username, 'email':newUser.email}), 200

@app.route('/login')
def login():
    auth = request.authorization

    if not auth or not auth.username or not auth.password:
        return jsonify({'err': "you should enter username and password"}), 401

    userId = auth.username.strip()
    password = auth.password
    user = User.query.filter_by(username=userId).first()
    if (not user):
        return jsonify({'err': 'not a valid user'}), 400
    user = user.as_dict()
    if not check_password_hash(user['password'], password):
        return jsonify({'err': 'not a valid user (pass)'}), 400
    token = jwtEncode({
        'userId': user['id'],
        'username': user['username'],
        'email': user['email'],
        'exp': datetime.utcnow() + timedelta(hours=48)
    }, app.config['SECRET_KEY']).decode('UTF-8')
    return jsonify({'token': token, 'username': user['username'], 'email':user['email'] }), 200
