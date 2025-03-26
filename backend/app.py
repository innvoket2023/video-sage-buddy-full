from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import os
import tempfile
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai
from collections import defaultdict
import cloudinary
import cloudinary.uploader
import cloudinary.api
import requests
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text 
from sqlalchemy.dialects.postgresql import UUID
from flask_bcrypt import Bcrypt
from sqlalchemy.exc import IntegrityError
from functools import wraps
import re
import datetime
import secrets
import uuid
import jwt  # New import for JWT

load_dotenv()

# Access environment variables
openai_api = os.getenv("OPENAI_API_KEY")
gemini_api = os.getenv("GEMINI_API_KEY")
cloudinary_cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
cloudinary_api_key = os.getenv("CLOUDINARY_API_KEY")
cloudinary_api_secret = os.getenv("CLOUDINARY_API_SECRET")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", secrets.token_hex(32))  # Add JWT secret key
JWT_EXPIRATION = int(os.getenv("JWT_EXPIRATION", 86400))  # Token expiration in seconds (default: 1 day)

# Initialize Cloudinary
cloudinary.config(
    cloud_name=cloudinary_cloud_name,
    api_key=cloudinary_api_key,
    api_secret=cloudinary_api_secret,
    secure=True
)
# Initialize the Google Generative AI correctly
genai.configure(api_key=gemini_api)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", secrets.token_hex(16))
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("SQLALCHEMY_DATABASE_URI")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Move CORS setup here, before any route definitions
CORS(app, 
     resources={r"/*": {
         "origins": ["http://localhost:8080", "http://localhost:3000", "http://localhost:5000"],
         "supports_credentials": True,
         "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"]
     }})

db = SQLAlchemy(app)

# Add explicit CORS handling
# @app.after_request
# def after_request(response):
#     response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#     response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
#     response.headers.add('Access-Control-Allow-Credentials', 'true')
#     return response

#Below can be implemented in MODEL.PY
    
class User(db.Model):
    __tablename__ = 'users'
    
    user_id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    updated_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())
    last_login = db.Column(db.DateTime(timezone=True), nullable=True, onupdate=db.func.now())
    is_active = db.Column(db.Boolean, default=False)

    videos = db.relationship("Video", back_populates="user", cascade="all, delete-orphan")
    
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

class Video(db.Model):
    __tablename__ = 'videos'
    
    video_id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = db.Column(UUID(as_uuid=True), db.ForeignKey('users.user_id', ondelete='CASCADE'), nullable=False)
    safe_name_for_vectordb = db.Column(db.String(255), nullable=True)
    title = db.Column(db.String(255), nullable=False)
    transcript = db.Column(db.Text, nullable=True)
    description = db.Column(db.Text, nullable=True)
    cloudinary_public_id = db.Column(db.String(255), nullable=False)
    cloudinary_url = db.Column(db.Text, nullable=False)
    cloudinary_resource_type = db.Column(db.String(50), nullable=True)
    duration = db.Column(db.Numeric, nullable=True)
    upload_date = db.Column(db.DateTime(timezone=True), server_default=db.func.now())
    status = db.Column(db.String(50), default='active')
    
    # Add metadata fields similar to User
    updated_at = db.Column(db.DateTime(timezone=True), server_default=db.func.now(), onupdate=db.func.now())
    view_count = db.Column(db.Integer, default=0)
    
    # Define relationship back to user
    user = db.relationship("User", back_populates="videos")
    
    # Add validation for status field
    __table_args__ = (
        db.CheckConstraint(status.in_(['processing', 'active', 'archived']), name='status_check'),
    )
    
    def __repr__(self):
        return f'<Video {self.title} ({self.video_id})>'
    
    # Helper functions
    def to_dict(self):
        return {
            'id': str(self.video_id),
            'user_id': str(self.user_id),
            'title': self.title,
            'description': self.description,
            'cloudinary_public_id': self.cloudinary_public_id,
            'cloudinary_url': self.cloudinary_url,
            'cloudinary_resource_type': self.cloudinary_resource_type,
            'duration': float(self.duration) if self.duration else None,
            'upload_date': self.upload_date.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'status': self.status,
            'view_count': self.view_count,
            'username': self.user.username if self.user else None
        }
    
    def increment_view(self):
        """Increment the view count for this video"""
        self.view_count += 1
        return self.view_count
    
    @property
    def is_active(self):
        """Check if the video is in active status"""
        return self.status == 'active'
    
    @property
    def formatted_duration(self):
        """Return the duration in a human-readable format (minutes:seconds)"""
        if not self.duration:
            return "0:00"
        
        total_seconds = int(float(self.duration))
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}:{seconds:02d}"

# JWT Token functions
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

#Below are the CRUD operations for LOGIN/SIGNUP (USER TABLE)
#===========================================================

bcrypt = Bcrypt(app)

# Signup/Registration function
@app.route('/api/signup', methods=['POST'])
def signup():
    # if request.method == 'OPTIONS':
    #     return '', 200
        
    data = request.get_json()
    # Validate required fields
    if not all(k in data for k in ('username', 'email', 'password')):
        return jsonify({'error': 'Missing required fields'}), 400
    
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    # Validate email format
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    if not email_pattern.match(email):
        return jsonify({'error': 'Invalid email format'}), 400
    
    # Validate password strength (example policy)
    if len(password) < 8:
        return jsonify({'error': 'Password must be at least 8 characters long'}), 400
    
    # Hash the password
    password = bcrypt.generate_password_hash(password).decode('utf-8')
    print("It works fine before Table filling") 
    # Create a new user
    new_user = User(
        username=username,
        email=email,
        password=password
    )

    print("It works fine after table filling")
    
    try:
        db.session.add(new_user)
        print("It works fine after adding the changes (the new user)")
        db.session.commit()
        print("It works fine after the commit too")
        
        return jsonify({
            'message': 'Registration successful! Please check your email to activate your account.',
            'user': {
                'id': str(new_user.user_id),
                'username': new_user.username,
                'email': new_user.email
            },
            # 'token': token  # Include JWT token in response
        }), 201

    except IntegrityError as e:
        db.session.rollback()
        # Check what kind of integrity error
        if 'username' in str(e.orig):
            return jsonify({'error': 'Username already exists'}), 409
        elif 'email' in str(e.orig):
            return jsonify({'error': 'Email already exists'}), 409
        else:
            return jsonify({'error': 'Registration failed due to database error'}), 500
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/mock', methods=["GET"])
def mock():
    # Get token from Authorization header
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    user_id = decode_token(token)
    return jsonify({"user_id": user_id}), 200

# Login function
@app.route('/api/login', methods=['POST', 'OPTIONS'])
def login():
    if request.method == 'OPTIONS':
        return '', 200
        
    data = request.get_json()
    
    # Check if login is via username or email
    identifier = data.get('username', '') or data.get('email', '')
    password = data.get('password', '')
    
    if not identifier or not password:
        return jsonify({'error': 'Username/email and password are required'}), 400
    
    # Find the user
    try:
        # Check if the identifier is an email
        if '@' in identifier:
            user = User.query.filter_by(email=identifier).first()
        else:
            user = User.query.filter_by(username=identifier).first()
        
        if not user:
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Check password
        if not bcrypt.check_password_hash(user.password, password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Check if account is active
        if not user.is_active:
            return jsonify({'error': 'Account not activated. Please check your email.'}), 403
        
        # Update last_login timestamp
        user.last_login = datetime.datetime.now(datetime.timezone.utc)
        db.session.commit()
        
        # Generate JWT token
        token = generate_token(user.user_id)
        
        return jsonify({
            'message': 'Login successful',
            'user': user.to_dict(),
            'token': str(token)  # Include JWT token in response
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Login failed: {str(e)}'}), 500

# Account activation function
@app.route('/api/activate/<uuid:user_id>', methods=['GET'])
def activate_account(user_id):
    # if request.method == 'OPTIONS':
    #     return '', 200
        
    user = User.query.filter_by(user_id=user_id).first()
    
    if not user:
        return jsonify({'error': 'Invalid activation link'}), 404
    
    if user.is_active:
        return jsonify({'message': 'Account already activated'}), 200
    
    user.is_active = True
    db.session.commit()
    
    return jsonify({'message': 'Account activated successfully! You can now log in.'}), 200

# Password reset request
@app.route('/api/reset-password', methods=['POST'])
def request_password_reset():
    # if request.method == 'OPTIONS':
    #     return '', 200
        
    data = request.get_json()
    email = data.get('email')
    
    if not email:
        return jsonify({'error': 'Email is required'}), 400
    
    user = User.query.filter_by(email=email).first()
    
    if not user:
        # Don't reveal if email exists or not for security
        return jsonify({'message': 'If your email exists in our system, you will receive a password reset link'}), 200
    
    # In a real application, you would:
    # 1. Generate a secure token
    # 2. Store it with an expiration time
    # 3. Send an email with the reset link
    
    return jsonify({'message': 'If your email exists in our system, you will receive a password reset link'}), 200

# Logout function - just a placeholder since JWT is stateless
@app.route('/api/logout', methods=['POST'])
def logout():
    # if request.method == 'OPTIONS':
    #     return '', 200
    
    # JWT is stateless, so no server-side logout is needed
    # Client should discard the token
    return jsonify({'message': 'Logged out successfully'}), 200

#===========================================================
# JWT auth decorator to replace login_required
def jwt_required(function):
    @wraps(function)
    def decorated_function(*args, **kwargs):
        # if request.method == 'OPTIONS':
        #     return function(*args, **kwargs)
            
        # Get token from header
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"error": "Missing Authorization header"}), 401
            
        # Format: "Bearer <token>"
        try:
            token = auth_header.split(' ')[1]
        except IndexError:
            return jsonify({"error": "Invalid Authorization header format"}), 401
            
        # Decode and validate token
        user_id = decode_token(token)
        if not user_id:
            return jsonify({"error": "Invalid or expired token"}), 401
            
        # Add user_id to request context
        request.user_id = user_id
        return function(*args, **kwargs)
    return decorated_function

#===========================================================
#Below are the CRUD operations 

#Below can be implemented in either MAIN.PY or ROUTER.PY
# Initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api)

# Dictionary to store vector databases for each video
video_vector_dbs = {}
# Modified transcribe_video to accept file paths
def transcribe_video(video_path):  # Changed parameter name
    """
    Transcribes a video from a file path
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Open the file from the provided path
        with open(video_path, 'rb') as f:  # Use the path directly
            video_bytes = f.read()
            
        response = model.generate_content(
            contents=[
                {"mime_type": "video/mp4", "data": video_bytes},
                {"text": "Transcribe the video with timestamps. Make sure to add the white/black board content in the transcription too"}
            ]
        )
        
        # Clean up the temporary file
        os.unlink(video_path)  # Delete the downloaded file
        
        return response.text
    except Exception as e:
        print(f"Error transcribing video: {e}")
        return None

def create_documents(transcript, video_name):
    """Splits transcript into chunks with timestamps."""
    timestamp_pattern = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]")
    lines = transcript.split("\n")
    text_chunks, timestamps = [], []
    current_timestamp, current_text = None, ""
    
    for line in lines:
        match = timestamp_pattern.search(line)
        if match:
            if current_text:
                text_chunks.append(current_text.strip())
                timestamps.append(current_timestamp)
                current_text = ""
            current_timestamp = match.group(1)
            line = line.replace(match.group(0), "").strip()
        if line.strip():
            current_text += " " + line if current_text else line
    
    if current_text:
        text_chunks.append(current_text.strip())
        timestamps.append(current_timestamp)
    
    docs = [Document(page_content=chunk, metadata={"source": video_name, "timestamp": ts}) 
            for chunk, ts in zip(text_chunks, timestamps)]
    return docs

def _get_vector_db(user_id, video_name: str, safe_video_name):
    """Get or create the appropriate vector database."""
    if video_name == "all":
        if not os.path.exists(f"faiss_indexes/{user_id}/all"):
            print("There are no videos")
            return None
        return FAISS.load_local(f"faiss_indexes/{user_id}/all", embedding_model, allow_dangerous_deserialization=True)
    
    individual_vector_db_path =f"faiss_indexes/{user_id}/individual/{safe_video_name}" 

    if os.path.exists(individual_vector_db_path):
        return FAISS.load_local(individual_vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        raise ValueError(f"The video: {video_name} doesn't exist at {individual_vector_db_path}")
     
def _prepare_results(docs: list[Document], 
                    group_by_source: bool = False, 
                    get_majority_source: bool = False) -> tuple:
    """Return format depends on get_majority_source flag"""
    base_results = [{
        "content": doc.page_content,
        "timestamp": doc.metadata.get("timestamp", ""),
        "source": doc.metadata.get("source", "")
    } for doc in docs]

    if not group_by_source:
        return (base_results, None) if get_majority_source else base_results

    source_counts = defaultdict(int)
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        source_counts[source] += 1

    if not source_counts:
        return ([], None) if get_majority_source else []

    max_count = max(source_counts.values())
    filtered_results = [
        result for result in base_results
        if source_counts[result["source"]] == max_count
    ]
    
    majority_source = max(source_counts, key=source_counts.get) if source_counts else None

    if get_majority_source:
        return filtered_results, majority_source
    return filtered_results

def _gemini_fallback(query: str, transcript: str) -> dict:
    """Generate fallback response using Gemini."""
    model = genai.GenerativeModel('gemini-1.5-pro')
    # context = "\n\n".join([v["transcript"] for v in video_metadata.values() if v.get("transcript")]) if video_name == "all" \
    #     else video_metadata.get(video_name, {}).get("transcript", "")
    context = transcript
    
    if not context:
        return {
            "content": "No relevant information found in videos",
            "timestamp": "",
            "source": "System"
        }

    prompt = f"Based on this context: {context[:10000]}\\n\\nAnswer concisely: {query}\\n\\nImportant guidelines:\\n- Complete your response in a friendly, helpful tone\\n- If the question relates to video content, include up to 3 most important timestamps in strictly [HH:MM:SS] format at the end\\n- Do not repeat any timestamp\\n- Only provide timestamps for video-related questions\\n- If the question is completely unrelated to the video, do not include any timestamps\\n- Place timestamps at the very end of your response without any additional commentary\\n\\nExample good response with timestamps:\\n[Your concise answer to the query]\\n[HH:MM:SS]\\n[HH:MM:SS]\\n[HH:MM:SS]\\n\\nExample good response without timestamps:\\n[Your concise answer to the unrelated query]"
    response = model.generate_content(prompt)
    return response.text

# Global dictionary to store video metadata
video_metadata = {}
universal_docs = []

def download_video_from_cloudinary(video_url):
    """
    Downloads a video from Cloudinary URL to a temporary file
    """
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    
    try:
        # Download the file
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        # Write the file to the temporary location
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        
        temp_file.close()
        return temp_file.name
    except Exception as e:
        temp_file.close()
        os.unlink(temp_file.name)
        print(f"Error downloading video: {e}")
        return None

@app.route('/upload-and-store', methods=['POST'])
@jwt_required
def upload_and_store():
    data = request.get_json()

    # Get metadata from request
    title = data.get("title", data.get("public_id"))
    description = data.get("description", "")
    cloudinary_url = data.get("video_url")
    cloudinary_public_id = data.get("public_id")

    if not cloudinary_url or not cloudinary_public_id:
        return jsonify({"error": "Missing video URL or public ID"}), 400

    video_name = title

    # Get user ID from JWT token
    user_id = request.user_id

    # Process video transcription (keeping your existing functionality)
    video = download_video_from_cloudinary(cloudinary_url)
    transcript = transcribe_video(video)
    if not transcript:
        return jsonify({"error": "Transcription failed"}), 500

    # Check if video already exists
    if video_name in video_metadata:
        return jsonify({"error": "Video with this title already exists"}), 409

    # Create and store vector DB (keeping your existing functionality)
    docs = create_documents(transcript, video_name)
    vector_db = FAISS.from_documents(docs, embedding_model)
    video_vector_dbs[video_name] = vector_db
    global universal_docs
    universal_docs += docs

    # Handle universal vector DB (keeping your existing functionality)
    try:
        if os.path.exists(f"faiss_indexes/{user_id}/all"):
            combined_db = FAISS.load_local(f"faiss_indexes/{user_id}/all", embedding_model)
            combined_db.merge_from(vector_db)
        else:
            combined_db = FAISS.from_documents(docs, embedding_model)

        combined_db.save_local(f"faiss_indexes/{user_id}/all")
        video_vector_dbs["all"] = combined_db
    except Exception as e:
        print(f"Error updating combined index: {e}")
        #hope we never hit this exception block
        video_vector_dbs["all"] = FAISS.from_documents(universal_docs, embedding_model)
        video_vector_dbs["all"].save_local(f"faiss_indexes/{user_id}/all")

    # Save individual index to disk (keeping your existing functionality)
    safe_video_name = re.sub(r'[^a-zA-Z0-9_-]', '_', video_name)
    vector_db.save_local(f"faiss_indexes/{user_id}/individual/{safe_video_name}")

    # Create Video record in the database
    new_video = Video(
        user_id=user_id,
        title=title,
        safe_name_for_vectordb=safe_video_name,
        transcript=transcript,
        description=description,
        cloudinary_public_id=cloudinary_public_id,
        cloudinary_url=cloudinary_url,
        cloudinary_resource_type="video",
        status='active',
    )

    db.session.add(new_video)

    # Store metadata for search functionality (your existing mechanism)
    video_metadata[video_name] = {
        "publicID": cloudinary_public_id,
        "transcript": transcript,
        "description": description,
        "processed": True,
        "index_path": f"faiss_indexes/individual/{safe_video_name}", #make sure to always have unique video names
        "video_url": cloudinary_url,
        "video_id": str(new_video.video_id),  # Add the database ID for reference
    }

    db.session.commit()

    return jsonify({
        "message": "Video uploaded and stored successfully",
        "video_name": video_name,
        "video_id": str(new_video.video_id),
        "transcript": transcript
    })

@app.route('/videos', methods=['GET'])
@jwt_required
def get_videos():
    # if request.method == 'OPTIONS':
    #     return '', 200
        
    print(list(video_metadata.values()))
    return jsonify({"videos": list(video_metadata.keys())})

@app.route('/preview', methods=['GET'])
@jwt_required
def get_preview():
    """Retrieve a list of uploaded videos from Cloudinary."""
    # if request.method == 'OPTIONS':
    #     return '', 200
    user_videos = Video.query.filter_by(user_id = request.user_id).all()  
    user_videos_list = [{"publicID": video.cloudinary_public_id, "description": video.description, "video_url": video.cloudinary_url} for video in user_videos]
    try:
        return jsonify({"videos": user_videos_list})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query', methods=['POST'])
@jwt_required
def query_video():
    # if request.method == 'OPTIONS':
    #     return '', 200
    user_id = request.user_id 
    try:
        data = request.get_json()
        query = data['query']
        video_name = data['video_name']
        safe_video_name = Video.query.filter_by(user_id=user_id, title=video_name).first().safe_name_for_vectordb if video_name != "all" else None

        # Validate inputs
        if not query or not video_name:
            return jsonify({"error": "Missing query or video_name"}), 400

        # Handle "all videos" special case
        # if video_name == "all" and not video_vector_dbs.get("all"):
        #     if not universal_docs:
        #         return jsonify({"error": "No videos available"}), 404
        #     video_vector_dbs["all"] = FAISS.from_documents(universal_docs, embedding_model)

        # Get appropriate vector DB
        vector_db = _get_vector_db(user_id, video_name, safe_video_name)
        if not vector_db:
            return jsonify({"error": "Video data not available"}), 404

        # Perform similarity search
        search_k = 1 if video_name == "all" else 3
        docs = vector_db.similarity_search(query, k=search_k)

        # Prepare results in uniform format
        filtered_results, majority_source = _prepare_results(
            docs=docs,
            group_by_source=(video_name == "all"),
            get_majority_source=True
        )

        # Determine final video source for Gemini
        final_source = majority_source if video_name == "all" else video_name

        transcript = Video.query.filter_by(user_id = user_id, title = final_source).first().transcript
        # Generate response using the determined source
        content = _gemini_fallback(query, transcript)
        results = {
            "content": content,
            "timestamp": filtered_results[0]["timestamp"],
            "source": final_source
        }

        return jsonify({"results": [results]})

    except KeyError as e:
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400
    except Exception as e:
        app.logger.error(f"Query error: {str(e)}")

        return jsonify({"error": "Processing failed"}), 500

# @app.route('/query', methods=['POST'])
# @jwt_required
# def query_video():
#     # if request.method == 'OPTIONS':
#     #     return '', 200
#     user_id = request.user_id 
#     print(f"[DEBUG] Query request received for user_id: {user_id}")
#
#     try:
#         data = request.get_json()
#         print(f"[DEBUG] Request data: {data}")
#
#         query = data['query']
#         video_name = data['video_name']
#         print(f"[DEBUG] Query: '{query}', Video name: '{video_name}'")
#
#         if video_name != "all":
#             video_record = Video.query.filter_by(user_id=user_id, title=video_name).first()
#             if video_record:
#                 safe_video_name = video_record.safe_name_for_vectordb
#                 print(f"[DEBUG] Found safe_video_name: {safe_video_name} for video: {video_name}")
#             else:
#                 print(f"[DEBUG] WARNING: No video record found for title: {video_name}")
#                 safe_video_name = None
#         else:
#             safe_video_name = None
#             print(f"[DEBUG] Using 'all' videos mode, safe_video_name set to None")
#
#         # Validate inputs
#         if not query or not video_name:
#             print(f"[DEBUG] Error: Missing query or video_name")
#             return jsonify({"error": "Missing query or video_name"}), 400
#
#         # Handle "all videos" special case
#         # if video_name == "all":
#         #     print(f"[DEBUG] Processing 'all videos' case")
#         #     if not video_vector_dbs.get("all"):
#         #         print(f"[DEBUG] 'all' vector DB not in memory")
#         #         if not universal_docs:
#         #             print(f"[DEBUG] Error: No universal_docs available")
#         #             return jsonify({"error": "No videos available"}), 404
#         #         print(f"[DEBUG] Creating 'all' vector DB from {len(universal_docs)} universal docs")
#         #         video_vector_dbs["all"] = FAISS.from_documents(universal_docs, embedding_model)
#         #         print(f"[DEBUG] Successfully created 'all' vector DB")
#
#         # Get appropriate vector DB
#         print(f"[DEBUG] Calling _get_vector_db for user_id: {user_id}, video_name: {video_name}, safe_video_name: {safe_video_name}")
#         vector_db = _get_vector_db(user_id, video_name, safe_video_name)
#
#         if vector_db:
#             print(f"[DEBUG] Successfully retrieved vector DB")
#             if hasattr(vector_db, 'index') and hasattr(vector_db.index, 'ntotal'):
#                 print(f"[DEBUG] Vector DB contains {vector_db.index.ntotal} vectors")
#         else:
#             print(f"[DEBUG] Error: Vector DB not found for {video_name}")
#             return jsonify({"error": "Video data not available"}), 404
#
#         # Perform similarity search
#         search_k = 1 if video_name == "all" else 3
#         print(f"[DEBUG] Performing similarity search with k={search_k}")
#
#         try:
#             docs = vector_db.similarity_search(query, k=search_k)
#             print(f"[DEBUG] Similarity search returned {len(docs)} documents")
#             for i, doc in enumerate(docs):
#                 print(f"[DEBUG] Doc {i+1}: Source={doc.metadata.get('source')}, Timestamp={doc.metadata.get('timestamp')}")
#                 print(f"[DEBUG] Content preview: {doc.page_content[:50]}...")
#         except Exception as search_error:
#             print(f"[DEBUG] ERROR in similarity search: {str(search_error)}")
#             raise
#
#         # Prepare results in uniform format
#         print(f"[DEBUG] Calling _prepare_results with {len(docs)} docs, group_by_source={video_name == 'all'}")
#         filtered_results, majority_source = _prepare_results(
#             docs=docs,
#             group_by_source=(video_name == "all"),
#             get_majority_source=True
#         )
#         print(f"[DEBUG] _prepare_results returned {len(filtered_results)} results. Majority source: {majority_source}")
#
#         # Determine final video source for Gemini
#         final_source = majority_source if video_name == "all" else video_name
#         print(f"[DEBUG] Final source for Gemini: {final_source}")
#
#         # Get transcript
#         print(f"[DEBUG] Fetching transcript for {video_name}")
#         transcript_record = Video.query.filter_by(user_id=user_id, title=final_source).first()
#
#         if transcript_record and transcript_record.transcript:
#             transcript = transcript_record.transcript
#             print(f"[DEBUG] Found transcript of length: {len(transcript)}")
#             print(f"[DEBUG] Transcript preview: {transcript[:100]}...")
#         else:
#             print(f"[DEBUG] WARNING: No transcript found for {video_name}")
#             transcript = ""
#
#         # Generate response using the determined source
#         print(f"[DEBUG] Calling _gemini_fallback with query and transcript")
#         content = _gemini_fallback(query, transcript)
#         print(f"[DEBUG] _gemini_fallback returned response of length: {len(content)}")
#
#         if not filtered_results:
#             print(f"[DEBUG] WARNING: filtered_results is empty, cannot access timestamp")
#             timestamp = None
#         else:
#             timestamp = filtered_results[0]["timestamp"]
#             print(f"[DEBUG] Using timestamp: {timestamp}")
#
#         results = {
#             "content": content,
#             "timestamp": timestamp,
#             "source": final_source
#         }
#         print(f"[DEBUG] Final results prepared: {results}")
#
#         return jsonify({"results": [results]})
#
#     except KeyError as e:
#         print(f"[DEBUG] KeyError: Missing required field: {str(e)}")
#         return jsonify({"error": f"Missing required field: {str(e)}"}), 400
#     except Exception as e:
#         error_msg = f"Query error: {str(e)}"
#         print(f"[DEBUG] CRITICAL ERROR: {error_msg}")
#         import traceback
#         print(f"[DEBUG] Traceback: {traceback.format_exc()}")
#         app.logger.error(error_msg)
#         return jsonify({"error": "Processing failed"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=port, debug=True)
