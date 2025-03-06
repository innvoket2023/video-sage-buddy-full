from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import time
import os
import tempfile
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

# Initialize the Google Generative AI correctly
genai.configure(api_key=gemini_api)

app = Flask(__name__)
# Configure CORS to allow requests from your frontend
CORS(app, resources={r"/*": {"origins": ["http://localhost:8080", "http://localhost:3000"]}})

# Initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api)

# Dictionary to store vector databases for each video
video_vector_dbs = {}

def transcribe_video(video_file):
    """
    Uploads and transcribes a video.
    video_file: Flask file object from request.files
    """
    # Save the uploaded file to a temporary location
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    video_file.save(temp_file.name)
    temp_file.close()
    
    try:
        # Use gemini to analyze the video
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Open the file in binary mode
        with open(temp_file.name, 'rb') as f:
            video_bytes = f.read()
        
        # Generate content with the video bytes and prompt
        response = model.generate_content(
            contents=[
                {"mime_type": "video/mp4", "data": video_bytes},
                {"text": "Transcribe the video with timestamps."}
            ]
        )
        
        # Clean up the temporary file
        os.unlink(temp_file.name)
        
        return response.text
    except Exception as e:
        # Clean up the temporary file in case of error
        os.unlink(temp_file.name)
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

# Global dictionary to store video metadata
video_metadata = {}

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
        
    # Pass the actual file object, not just the filename
    transcript = transcribe_video(file)
    
    if not transcript:
        return jsonify({"error": "Video processing failed"}), 500

    # Store the video name and transcript
    video_name = file.filename
    
    # Add video to metadata
    if video_name not in video_metadata:
        video_metadata[video_name] = {
            "transcript": transcript,
            "processed": False
        }
    
    return jsonify({
        "message": "Video uploaded successfully", 
        "video_name": video_name,
        "transcript": transcript
    })

@app.route('/store', methods=['POST'])
def store_transcript():
    data = request.json
    
    if not data or 'transcript' not in data:
        return jsonify({"error": "Missing transcript data"}), 400
    
    # Use provided video name or generate one
    video_name = data.get('video_name', f"Video_{int(time.time())}")
    transcript = data['transcript']
    description = data.get('description', '')
    
    # Store video metadata
    video_metadata[video_name] = {
        "transcript": transcript,
        "description": description,
        "processed": True
    }
    
    # Process and create vector DB
    docs = create_documents(transcript, video_name)
    video_vector_dbs[video_name] = FAISS.from_documents(docs, embedding_model)
    
    # Save the directory structure if needed
    os.makedirs("faiss_indexes", exist_ok=True)
    video_vector_dbs[video_name].save_local(f"faiss_indexes/{video_name.replace('.', '_')}")
    
    return jsonify({"message": "Transcript stored successfully", "video_name": video_name})

@app.route('/videos', methods=['GET'])
def get_videos():
    return jsonify({"videos": list(video_metadata.keys())})

@app.route('/query', methods=['POST'])
def query_video():
    data = request.json
    if not data or 'query' not in data or 'video_name' not in data:
        return jsonify({"error": "Missing query or video_name parameter"}), 400
    
    query = data['query']
    video_name = data['video_name']
    
    # Check if we have this video
    if video_name not in video_metadata:
        return jsonify({"error": f"Video '{video_name}' not found"}), 404
    
    # Check if we have a vector db for this video
    if video_name not in video_vector_dbs:
        # Try to load from disk
        try:
            vector_db = FAISS.load_local(f"faiss_indexes/{video_name.replace('.', '_')}", embedding_model)
            video_vector_dbs[video_name] = vector_db
        except Exception as e:
            print(f"Error loading vector DB: {e}")
            # If we have the transcript, we can recreate the vector DB
            if video_name in video_metadata and video_metadata[video_name].get("transcript"):
                docs = create_documents(video_metadata[video_name]["transcript"], video_name)
                video_vector_dbs[video_name] = FAISS.from_documents(docs, embedding_model)
            else:
                return jsonify({"error": "Could not load or create vector database for this video"}), 500
    
    try:
        # Get similar docs from the correct vector DB
        vector_db = video_vector_dbs[video_name]
        similar_docs = vector_db.similarity_search(query, k=3)
        
        # If no results, try to use Gemini to generate a response based on the transcript
        if not similar_docs and video_name in video_metadata:
            model = genai.GenerativeModel('gemini-1.5-pro')
            prompt = f"Based on this transcript of a video, answer the following question: {query}\n\nTranscript:\n{video_metadata[video_name].get('transcript', '')}"
            
            response = model.generate_content(prompt)
            
            return jsonify({"results": [{
                "content": response.text,
                "timestamp": "",
                "source": video_name
            }]})
        
        # Process the results
        results = [
            {
                "content": doc.page_content, 
                "timestamp": doc.metadata.get("timestamp", ""), 
                "source": doc.metadata.get("source", "")
            } 
            for doc in similar_docs
        ]
        
        return jsonify({"results": results})
    
    except Exception as e:
        print(f"Error during query: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)