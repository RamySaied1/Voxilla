from config import Config
from app import currentApp

if __name__ == '__main__':
    currentApp.run(debug=Config.DEBUG, use_reloader=False)