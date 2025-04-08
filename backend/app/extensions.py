from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
import os
import cloudinary
import cloudinary.uploader
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize extensions
db = SQLAlchemy()
bcrypt = Bcrypt()

# Configure services that don't require app instance
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Constants
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", os.urandom(32))
JWT_EXPIRATION = int(os.getenv("JWT_EXPIRATION", 86400))
