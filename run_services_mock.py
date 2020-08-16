from config import Config
Config.SERVICES_MOCK = True

from app import currentApp


if __name__ == '__main__':
    currentApp.run(debug=Config.DEBUG, use_reloader=True)