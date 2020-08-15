from flask import jsonify, request
from jwt import decode as jwtDecode
from functools import wraps
from config import Config
app_secret_key = Config.SECRET_KEY

def user_token_available(role='any'):
    def wrapper(fn):
        @wraps(fn)
        def decorated(*args, **kwargs):
            error = None
            currUser = None
            if 'x-access-token' in request.headers:
                token = request.headers['x-access-token']
                try:
                    data = jwtDecode(token, app_secret_key)
                    if (role == 'any' or data['role'] in role.split(', ')):
                        currUser = {'id': data.get('userId'),'username': data.get('username'), 'password': data.get('password')}
                        print('{} successfully found'.format(currUser['id']))
                    else:
                        error = 'Not allowed for this user', 403
                except:
                    error = 'Token is invalid!', 401
            else:
                error = 'Token is missing!', 401

            auth = {'user': currUser, 'error': error}
            return fn(auth, *args, **kwargs)
        decorated.__name__ = "{}_{}".format(decorated.__name__, fn.__name__)
        return decorated
    return wrapper


def userRequired(role='any'):
    def decorator(fn):
        @user_token_available(role)
        def decorated(auth, *args, **kwargs):
            if(auth['error']):
                return  'err', {'cause': auth['error'][0]}, auth['error'][1]
            return fn(auth['user'], *args, **kwargs)
        decorated.__name__ = "{}_{}".format(decorated.__name__, fn.__name__)
        return decorated
    return decorator

def userRequiredJson(role='any'):
    def decorator(fn):
        @user_token_available(role)
        def decorated(auth, *args, **kwargs):
            if(auth['error']):
                return jsonify({'err': auth['error'][0]}), auth['error'][1]
            return fn(auth['user'], *args, **kwargs)
        decorated.__name__ = "{}_{}".format(decorated.__name__, fn.__name__)
        return decorated
    return decorator