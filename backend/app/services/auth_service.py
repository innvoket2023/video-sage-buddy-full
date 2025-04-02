import jwt
import datetime
import os
import secrets
from app.extensions import db, bcrypt
from app.models import User
from app.extensions import JWT_SECRET_KEY, JWT_EXPIRATION

# JWT configuration
# JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))
# JWT_EXPIRATION = int(os.getenv("JWT_EXPIRATION", 86400))  # 1 day default

def generate_token(user_id):
    """Generate a JWT token for the user"""
    payload = {
        'exp': datetime.datetime.utcnow() + datetime.timedelta(seconds=JWT_EXPIRATION),
        'iat': datetime.datetime.utcnow(),
        'sub': str(user_id)
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm='HS256')

def decode_token(token):
    """Decode the JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
        return payload['sub']
    except jwt.ExpiredSignatureError:
        return None  # Token expired
    except jwt.InvalidTokenError:
        return None  # Invalid token

def hash_password(password):
    """Hash a password using bcrypt"""
    return bcrypt.generate_password_hash(password).decode('utf-8')

def verify_password(stored_hash, password):
    """Verify a password against its hash"""
    return bcrypt.check_password_hash(stored_hash, password)

def create_user(username, email, password):
    """Create a new user in the database"""
    hashed_password = hash_password(password)
    new_user = User(
        username=username,
        email=email,
        password=hashed_password
    )
    db.session.add(new_user)
    db.session.commit()
    return new_user

def get_user_by_id(user_id):
    """Get a user by ID"""
    return User.query.filter_by(user_id=user_id).first()

def get_user_by_email(email):
    """Get a user by email"""
    return User.query.filter_by(email=email).first()

def get_user_by_username(username):
    """Get a user by username"""
    return User.query.filter_by(username=username).first()

def get_user_by_identifier(identifier):
    """Get a user by email or username"""
    if '@' in identifier:
        return get_user_by_email(identifier)
    return get_user_by_username(identifier)

def activate_user(user_id):
    """Activate a user account"""
    user = get_user_by_id(user_id)
    if user:
        user.is_active = True
        db.session.commit()
        return True
    return False

def update_user_login_timestamp(user):
    """Update the last login timestamp for a user"""
    user.last_login = datetime.datetime.now(datetime.timezone.utc)
    db.session.commit()
