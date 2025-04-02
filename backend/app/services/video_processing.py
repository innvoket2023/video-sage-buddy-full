import os
import re
import tempfile
import requests
import google.generativeai as genai
from collections import defaultdict
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=os.getenv("OPENAI_API_KEY"))

# Dictionary to store vector databases for each video
video_vector_dbs = {}

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

def transcribe_video(video_path):
    """
    Transcribes a video from a file path
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Open the file from the provided path
        with open(video_path, 'rb') as f:
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

def get_vector_db(user_id, video_name, safe_video_name):
    """Get or create the appropriate vector database."""
    if video_name == "all":
        if not os.path.exists(f"faiss_indexes/{user_id}/all"):
            print("There are no videos")
            return None
        return FAISS.load_local(f"faiss_indexes/{user_id}/all", embedding_model, allow_dangerous_deserialization=True)
    
    individual_vector_db_path = f"faiss_indexes/{user_id}/individual/{safe_video_name}" 

    if os.path.exists(individual_vector_db_path):
        return FAISS.load_local(individual_vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    else:
        raise ValueError(f"The video: {video_name} doesn't exist at {individual_vector_db_path}")

def prepare_results(docs, group_by_source=False, get_majority_source=False):
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

def gemini_fallback(query, transcript):
    """Generate fallback response using Gemini."""
    model = genai.GenerativeModel('gemini-1.5-pro')
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

def create_vector_database(user_id, video_name, docs):
    """Create vector databases for a video"""
    try:
        # Create directories if they don't exist
        os.makedirs(f"faiss_indexes/{user_id}/individual", exist_ok=True)
        
        # Create safe name for filesystem
        safe_video_name = re.sub(r'[^a-zA-Z0-9_-]', '_', video_name)
        
        # Create vector database from documents
        vector_db = FAISS.from_documents(docs, embedding_model)
        
        # Save individual index
        vector_db.save_local(f"faiss_indexes/{user_id}/individual/{safe_video_name}")
        
        # Add to combined index
        try:
            if os.path.exists(f"faiss_indexes/{user_id}/all"):
                combined_db = FAISS.load_local(f"faiss_indexes/{user_id}/all", embedding_model, allow_dangerous_deserialization=True)
                combined_db.merge_from(vector_db)
            else:
                os.makedirs(f"faiss_indexes/{user_id}/all", exist_ok=True)
                combined_db = FAISS.from_documents(docs, embedding_model)
                
            combined_db.save_local(f"faiss_indexes/{user_id}/all")
        except Exception as e:
            print(f"Error updating combined index: {e}")
            
        return safe_video_name
    except Exception as e:
        print(f"Error creating vector database: {e}")
        return None
