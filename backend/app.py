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
from collections import defaultdict
import cloudinary
import cloudinary.uploader
import cloudinary.api
import requests

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
CORS(app, resources={r"/*": {"origins": ["http://localhost:8080", "http://localhost:3000"]}})

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

def _get_vector_db(video_name: str) -> FAISS:
    """Get or create the appropriate vector database."""
    if video_name == "all":
        if "all" not in video_vector_dbs and universal_docs:
            video_vector_dbs["all"] = FAISS.from_documents(universal_docs, embedding_model)
        return video_vector_dbs.get("all")
    
    if video_name not in video_vector_dbs:
        if video_name in video_metadata and video_metadata[video_name].get("transcript"):
            docs = create_documents(video_metadata[video_name]["transcript"], video_name)
            video_vector_dbs[video_name] = FAISS.from_documents(docs, embedding_model)
        else:
            raise ValueError(f"Vector DB for {video_name} not found and couldn't be created")
    
    return video_vector_dbs[video_name]

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



def _gemini_fallback(query: str, video_name: str) -> dict:
    """Generate fallback response using Gemini."""
    model = genai.GenerativeModel('gemini-1.5-pro')
    context = "\n\n".join([v["transcript"] for v in video_metadata.values() if v.get("transcript")]) if video_name == "all" \
        else video_metadata.get(video_name, {}).get("transcript", "")
    
    if not context:
        return {
            "content": "No relevant information found in videos",
            "timestamp": "",
            "source": "System"
        }

    prompt = f"Based on this context: {context[:10000]}\n\nAnswer concisely: {query}"
    response = model.generate_content(prompt)
    # return {
    #     "content": response.text,
    #     # "timestamp": "AI-generated response",
    #     # "source": "Gemini"
    # }
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
def upload_and_store():
    data = request.get_json()
    print(data)
    # Get metadata from form data
    title = request.form.get('title', data.get("public_id"))  # Use title or filename
    description = request.form.get('description', "")
    
    video_name = title  
    # Process video transcription
    video = download_video_from_cloudinary(data.get("video_url"))
    transcript = transcribe_video(video)
    if not transcript:
        return jsonify({"error": "Transcription failed"}), 500

    # Store directly with metadata
    
    # Check if video already exists
    if video_name in video_metadata:
        return jsonify({"error": "Video with this title already exists"}), 409

    # Create and store vector DB
    docs = create_documents(transcript, video_name)
    
    # Store to universal docs and individual index
    video_vector_dbs[video_name] = FAISS.from_documents(docs, embedding_model)
    global universal_docs
    universal_docs += docs
    
    # Create/update the combined index
    try:
        if os.path.exists("faiss_indexes/all"):
            combined_db = FAISS.load_local("faiss_indexes/all", embedding_model)
            combined_db.merge_from(video_vector_dbs[video_name])
        else:
            combined_db = FAISS.from_documents(universal_docs, embedding_model)
        
        combined_db.save_local("faiss_indexes/all")
        video_vector_dbs["all"] = combined_db
    except Exception as e:
        print(f"Error updating combined index: {e}")
        # Fallback to creating fresh combined index
        video_vector_dbs["all"] = FAISS.from_documents(universal_docs, embedding_model)
        video_vector_dbs["all"].save_local("faiss_indexes/all")

    # Save individual index to disk
    os.makedirs("faiss_indexes/individual", exist_ok=True)
    safe_video_name = re.sub(r'[^a-zA-Z0-9_-]', '_', video_name)
    video_vector_dbs[video_name].save_local(f"faiss_indexes/individual/{safe_video_name}")

    # Store metadata last to ensure indexes are valid
    video_metadata[video_name] = {
        "publicID" : video_name,
        "transcript": transcript,
        "description": description,
        "processed": True,
        "index_path": f"faiss_indexes/individual/{safe_video_name}",
        "video_url": data.get("video_url") 
    }

    return jsonify({
        "message": "Video uploaded and stored successfully",
        "video_name": video_name,
        "transcript": transcript
    })

@app.route('/videos', methods=['GET'])
def get_videos():
    print(list(video_metadata.values()))
    return jsonify({"videos": list(video_metadata.keys())})


@app.route('/preview', methods= ['GET'])
def get_preview():
    """Retrieve a list of uploaded videos from Cloudinary."""
    try:
        return jsonify({"videos": list(video_metadata.values())})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/query', methods=['POST'])
def query_video():
    try:
        data = request.get_json()
        query = data['query']
        video_name = data['video_name']

        # Validate inputs
        if not query or not video_name:
            return jsonify({"error": "Missing query or video_name"}), 400
        
        # Handle "all videos" special case
        if video_name == "all" and not video_vector_dbs.get("all"):
            if not universal_docs:
                return jsonify({"error": "No videos available"}), 404
            video_vector_dbs["all"] = FAISS.from_documents(universal_docs, embedding_model)

        # Get appropriate vector DB
        vector_db = _get_vector_db(video_name)
        if not vector_db:
            return jsonify({"error": "Video data not available"}), 404

        # Perform similarity search
        search_k = 3 if video_name == "all" else 3
        docs = vector_db.similarity_search(query, k=search_k)
        
        # Prepare results in uniform format
        filtered_results, majority_source = _prepare_results(
            docs=docs,
            group_by_source=(video_name == "all"),
            get_majority_source=True
        )

        # Determine final video source for Gemini
        final_source = majority_source if video_name == "all" else video_name
        
        # Generate response using the determined source
        content = _gemini_fallback(query, final_source)
        results = {
            "content": content,
            "timestamp": filtered_results[0]["timestamp"],
            "source": final_source
        }

        return jsonify({"results": [results]})

    except Exception as e:
        app.logger.error(f"Query error: {str(e)}")
        return jsonify({"error": "Processing failed"}), 500

if __name__ == '__main__':
    app.run(debug=True)