from flask import Flask
from flask_cors import CORS
#from flask_caching import Cache
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config.from_object('config.DevelopmentConfig')
#cache = Cache(app, config={'CACHE_TYPE': 'simple'})
db = SQLAlchemy(app)
CORS(app)