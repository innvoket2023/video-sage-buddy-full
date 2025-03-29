import os
import secrets
from dotenv import load_dotenv

load_dotenv()
class Config:
    PORT = int(os.environ.get('PORT', 5000))
    HOST = '0.0.0.0'

class DevelopmentConfig:
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.getenv("SQLALCHEMY_DATABASE_URI")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(16))

class ProductionConfig:
    DEBUG = False

config_dict = {
    "development": DevelopmentConfig,
    "production": ProductionConfig
}
