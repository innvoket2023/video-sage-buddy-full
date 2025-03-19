from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text 
from sqlalchemy.dialects.postgresql import UUID
import dotenv
import uuid

test = Flask(__name__)

test.config["SQLALCHEMY_DATABASE_URI"]

test.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(test)

class User(db.Model):
    __tablename__ = 'users'
    
    user_id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    updated_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())
    last_login = db.Column(db.DateTime(timezone=True), nullable=True, onupdate=db.func.now())
    is_active = db.Column(db.Boolean, default=False)
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    #Helper function
    def to_dict(self):
        return {
            'id': self.user_id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'last_login': self.last_login,
            'is_active': self.is_active 
        }

with test.app_context():
    try:
        db.session.execute(text("SELECT 1"))
        print("Connection Successful")
    except Exception as e:
        print(f"This is the error i am getting: {e}")




