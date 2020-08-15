from .app import app as currentApp, db
from app.auth import *
from app.clips import *
from app.projects import *
from app.demo import *
from app.io import *
if currentApp.config["MOCK"]:
    from app.services_mock import *
else:
    from app.services import *