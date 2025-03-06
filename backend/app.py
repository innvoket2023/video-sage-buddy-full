from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import time
import os
import tempfile
import requests
import cloudinary
import cloudinary.uploader
import cloudinary.api
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Access environment variables
openai_api = os.getenv("OPENAI_API_KEY")
gemini_api = os.getenv("GEMINI_API_KEY")
cloudinary_cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME")
cloudinary_api_key = os.getenv("CLOUDINARY_API_KEY")
cloudinary_api_secret = os.getenv("CLOUDINARY_API_SECRET")

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
# Configure CORS to allow requests from your frontend
CORS(app, resources={r"/*": {"origins": "http://localhost:8080"}})

# Initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api)

vector_db = None

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

def transcribe_video_from_url(video_url):
    """
    Downloads and transcribes a video from Cloudinary.
    """
    # Download the video to a temporary file
    temp_file_path = download_video_from_cloudinary(video_url)
    
    if not temp_file_path:
        return None
    
    try:
        # Use gemini to analyze the video
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Open the file in binary mode
        with open(temp_file_path, 'rb') as f:
            video_bytes = f.read()
        
        # Generate content with the video bytes and prompt
        response = model.generate_content(
            contents=[
                {"mime_type": "video/mp4", "data": video_bytes},
                {"text": "Transcribe the video with timestamps."}
            ]
        )
        
        # Clean up the temporary file
        os.unlink(temp_file_path)
        
        return response.text
    except Exception as e:
        # Clean up the temporary file in case of error
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
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

@app.route('/process', methods=['POST'])
def process_video():
    data = request.get_json()
    
    if not data or 'video_url' not in data:
        return jsonify({"error": "Missing video URL"}), 400
        
    video_url = data['video_url']
    
    # Transcribe the video from the Cloudinary URL
    transcript = transcribe_video_from_url(video_url)
    
    if not transcript:
        return jsonify({"error": "Video processing failed"}), 500
        
    return jsonify({"transcript": transcript})

@app.route('/store', methods=['POST'])
def store_transcript():
    global vector_db
    data = request.get_json()
    
    if not data or 'transcript' not in data or 'video_name' not in data:
        return jsonify({"error": "Missing required data"}), 400
        
    docs = create_documents(data['transcript'], data['video_name'])
    
    # Store video metadata along with transcript
    for doc in docs:
        doc.metadata["video_url"] = data.get('video_url', '')
        doc.metadata["description"] = data.get('description', '')
        doc.metadata["public_id"] = data.get('public_id', '')
    
    vector_db = FAISS.from_documents(docs, embedding_model)
    vector_db.save_local("faiss_index")
    return jsonify({"message": "Transcript stored successfully"})

@app.route('/query', methods=['POST'])
def query_video():
    global vector_db
    
    if not vector_db:
        # Try to load from disk if it exists
        try:
            vector_db = FAISS.load_local("faiss_index", embedding_model)
        except:
            return jsonify({"error": "No stored transcripts found"}), 400
    
    data = request.json
    if not data or 'query' not in data:
        return jsonify({"error": "Missing query parameter"}), 400
        
    query = data['query']
    similar_docs = vector_db.similarity_search(query, k=3)
    results = [{"content": doc.page_content, 
                "timestamp": doc.metadata.get("timestamp", ""), 
                "source": doc.metadata.get("source", ""),
                "video_url": doc.metadata.get("video_url", "")} for doc in similar_docs]
                
    return jsonify({"results": results})

@app.route('/videos', methods=['GET'])
def get_videos():
    """Retrieve a list of uploaded videos from Cloudinary."""
    try:
        # Fetch video resources from Cloudinary
        response = cloudinary.api.resources(
            type="upload", resource_type="video", max_results=10
        )

        videos = [
            {
                "video_name": item["public_id"],  # Use public_id as name
                "video_url": item["secure_url"],  # Secure URL for playback
                "public_id": item["public_id"],  # Cloudinary public_id
            }
            for item in response["resources"]
        ]

        return jsonify({"videos": videos})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)