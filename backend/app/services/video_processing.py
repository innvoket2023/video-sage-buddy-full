from app.admin.gemini import Gemini
from app.admin.llmusage import LLMUsage
import os
import re
import shutil
import tempfile
import requests
import google.generativeai as genai
from collections import defaultdict
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from flask import current_app

load_dotenv()

# Initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=os.getenv("OPENAI_API_KEY"))

# Dictionary to store vector databases for each video
video_vector_dbs = {}
# Get the usage tracker from app context
def get_usage_tracker():
    # If usage_tracker doesn't exist yet in app context, initialize it
    if not hasattr(current_app, 'usage_tracker'):
        # Initialize with app configuration
        current_app.usage_tracker = LLMUsage(
            token_quota=current_app.config.get('LLM_TOKEN_QUOTA'),
            cost_budget=current_app.config.get('LLM_COST_BUDGET'),
            storage_path=current_app.config.get('LLM_USAGE_STORAGE_PATH')
        )
    return current_app.usage_tracker

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
        all_vector_path = f"faiss_indexes/{user_id}/all"
        return FAISS.load_local(f"faiss_indexes/{user_id}/all", embedding_model, allow_dangerous_deserialization=True), all_vector_path
    
    individual_vector_db_path = f"faiss_indexes/{user_id}/individual/{safe_video_name}" 

    if os.path.exists(individual_vector_db_path):
        return FAISS.load_local(individual_vector_db_path, embedding_model, allow_dangerous_deserialization=True), individual_vector_db_path
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

def gemini_fallback(user_id, query, transcript):
    """Generate fallback response using Gemini."""
    print(f"[DEBUG] gemini_fallback called with user_id: {user_id}")
    print(f"[DEBUG] query: {query}")
    print(f"[DEBUG] transcript length: {len(transcript) if transcript else 0}")


    usage_tracker = get_usage_tracker()
    print(f"[DEBUG] LLMUsage initialized with token_quota: {usage_tracker.token_quota}, cost_budget: {usage_tracker.cost_budget}")
    if usage_tracker:
        print(f"[DEBUG] Got the access to the local usage_tracker")
    gemini_client: Gemini = Gemini(
        model_name="gemini-2.0-flash",
        api_key=os.environ.get("GEMINI_API_KEY"),
        usage_tracker=usage_tracker
    )
    print(f"[DEBUG] Gemini client initialized with model_name: {gemini_client._model_name}")

    context = transcript
    
    if not context:
        print("[DEBUG] No context found, returning default response")
        return {
            "content": "No relevant information found in videos",
            "timestamp": "",
            "source": "System"
        }

    prompt = f"Based on this context: {context[:10000]}\\n\\nAnswer concisely: {query}\\n\\nImportant guidelines:\\n- Complete your response in a friendly, helpful tone\\n- If the question relates to video content, include up to 3 most important timestamps in strictly [HH:MM:SS] format at the end\\n- Do not repeat any timestamp\\n- Only provide timestamps for video-related questions\\n- If the question is completely unrelated to the video, do not include any timestamps\\n- Place timestamps at the very end of your response without any additional commentary\\n\\nExample good response with timestamps:\\n[Your concise answer to the query]\\n[HH:MM:SS]\\n[HH:MM:SS]\\n[HH:MM:SS]\\n\\nExample good response without timestamps:\\n[Your concise answer to the unrelated query]"
    print(f"[DEBUG] Prompt created, length: {len(prompt)}")

    print("[DEBUG] Calling Gemini generate method")
    response = gemini_client.generate(user_id=user_id, prompt=prompt)
    print(f"[DEBUG] Gemini response received, length: {len(response) if response else 0}")
    
    return response

def create_vector_database(user_id, video_name, docs):
    """Create vector databases for a video"""
    try:
        # Create directories if they don't exist
        os.makedirs(f"faiss_indexes/{user_id}/individual", exist_ok=True)
        
        # Create safe name for filesystem
        safe_video_name = re.sub(r'[^a-zA-Z0-9_-]', '_', video_name)
        
        safe_video_name_as_ids = [f"{safe_video_name}_{i}" for i in range(len(docs))]
        # Create vector database from documents
        vector_db = FAISS.from_documents(docs, embedding_model, ids = safe_video_name_as_ids)
        
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

def remove_video_documents(vector_db, safe_video_name):
    """
    Remove all documents from a FAISS vector database that have IDs containing a specific video name.
    
    Args:
        vector_db: The FAISS vector database
        safe_video_name: The safe video name to filter documents by
    
    Returns:
        tuple: (updated_vector_db, num_deleted) - the updated database and count of deleted docs
    """
    
    # Find all docstore IDs that start with or match the video name pattern
    ids_to_delete = []
    
    for doc_id in vector_db.docstore._dict.keys():
        # Check if ID starts with the video name pattern followed by underscore
        if isinstance(doc_id, str) and doc_id.startswith(f"{safe_video_name}_"):
            ids_to_delete.append(doc_id)
    
    # Delete all identified documents at once
    if ids_to_delete:
        print(f"Deleting {len(ids_to_delete)} documents for video: {safe_video_name}")
        vector_db.delete(ids=ids_to_delete)
        return vector_db, len(ids_to_delete)
    else:
        print(f"No documents found for video: {safe_video_name}")
        return vector_db, 0


def delete_video_from_vector_database(user_id, video_name, safe_video_name):
    """
    Deletes vector databases for a specific video name or all videos for a user.

    Args:
        user_id: The user ID associated with the vector database
        video_name: The specific video name or "all" for all videos
        safe_video_name: The sanitized video name used for ID matching

    Returns:
        int: Number of documents removed from the combined vector database
    """
    try:
        individual_deleted = False
        
        if video_name != "all":
            # Handle individual vector database
            try:
                individual_path = f"faiss_indexes/{user_id}/individual/{safe_video_name}"
                if os.path.exists(individual_path):
                    shutil.rmtree(individual_path)  # Use shutil.rmtree instead of os.remove for directories
                    individual_deleted = True
                    print(f"Deleted individual vector database for {safe_video_name}")
                else:
                    print(f"Warning: Individual vector database not found at {individual_path}")
            except Exception as e:
                print(f"Error deleting individual database: {e}")
        
        # Switch to "all" vector database for deletion
        all_vector_db, _ = get_vector_db(user_id, "all", safe_video_name)
        
        # Update the combined vector database
        updated_vector_db, num_deleted = remove_video_documents(all_vector_db, safe_video_name)
        
        # Only save if documents were deleted
        if num_deleted > 0 or individual_deleted:
            updated_vector_db.save_local(f"faiss_indexes/{user_id}/all")
            print(f"Updated combined vector database, removed {num_deleted} documents")
            
        return num_deleted

    except Exception as e:
        print(f"Error during vector database deletion: {e}")
        return 0
