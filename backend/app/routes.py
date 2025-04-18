from flask import Blueprint, request, jsonify, current_app
from app.models import User, Video
from app.extensions import db
from app.services.audio_segmentation import create_audio_segments
from app.services.auth_service import decode_token
from app.middleware import jwt_required
from app.services.elevenlabsIO import create_voice_clones, text_to_speech
from app.services.video_processing import (delete_video_from_vector_database, transcribe_video, create_documents ,get_vector_db, prepare_results, gemini_fallback, create_vector_database)
from app.services.cloudinary import download_video_from_cloudinary, remove_video_from_cloudinary 
from app.services import utils
import re
import os
from sqlalchemy.exc import IntegrityError
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from flask import send_file
from cloudinary.exceptions import Error as CloudinaryError
import datetime

# Create blueprint
app_bp = Blueprint('app', __name__)

# Dictionary to store video metadata
video_metadata = {}
universal_docs = []

# Initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=os.getenv("OPENAI_API_KEY"))

@app_bp.route('/upload-and-store', methods=['POST'])
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

    # Process video transcription
    video = download_video_from_cloudinary(cloudinary_url)
    transcript = transcribe_video(video)
    if not transcript:
        return jsonify({"error": "Transcription failed"}), 500

    # Check if video already exists
    if video_name in video_metadata:
        return jsonify({"error": "Video with this title already exists"}), 409

    # Create and store vector DB
    docs = create_documents(transcript, video_name)
    safe_video_name = create_vector_database(user_id, video_name, docs)

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

    # Store metadata for search functionality
    video_metadata[video_name] = {
        "publicID": cloudinary_public_id,
        "transcript": transcript,
        "description": description,
        "processed": True,
        "index_path": f"faiss_indexes/{user_id}/individual/{safe_video_name}",
        "video_url": cloudinary_url,
        "video_id": str(new_video.video_id),
    }

    db.session.commit()

    return jsonify({
        "message": "Video uploaded and stored successfully",
        "video_name": video_name,
        "video_id": str(new_video.video_id),
        "transcript": transcript
    })

@app_bp.route('/videos', methods=['GET'])
@jwt_required
def get_videos():
    return jsonify({"videos": list(video_metadata.keys())})

@app_bp.route('/preview', methods=['GET'])
@jwt_required
def get_preview():
    """Retrieve a list of uploaded videos from Cloudinary."""
    user_videos = Video.query.filter_by(user_id=request.user_id).all()  
    user_videos_list = [
        {
            "publicID": video.cloudinary_public_id, 
            "description": video.description, 
            "video_url": video.cloudinary_url
        } 
        for video in user_videos
    ]
    
    try:
        return jsonify({"videos": user_videos_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app_bp.route('/query', methods=['POST'])
@jwt_required
def query_video():
    user_id = request.user_id 
    print(f"[DEBUG] user_id: {user_id}")
    try:
        data = request.get_json()
        query = data['query']
        video_name = data['video_name']
        print(f"[DEBUG] query: {query}, video_name: {video_name}")
        
        # Get safe_video_name if not querying all videos
        if video_name != "all":
            video = Video.query.filter_by(user_id=user_id, title=video_name).first()
            if not video:
                print(f"[DEBUG] Video '{video_name}' not found")
                return jsonify({"error": f"Video '{video_name}' not found"}), 404
            safe_video_name = video.safe_name_for_vectordb
            print(f"[DEBUG] safe_video_name: {safe_video_name}")
        else:
            safe_video_name = None
            print(f"[DEBUG] video_name is 'all', safe_video_name: {safe_video_name}")

        # Validate inputs
        if not query or not video_name:
            print(f"[DEBUG] Missing query or video_name")
            return jsonify({"error": "Missing query or video_name"}), 400

        # Get appropriate vector DB
        print(f"[DEBUG] Calling get_vector_db with user_id: {user_id}, video_name: {video_name}, safe_video_name: {safe_video_name}")
        vector_db, path = get_vector_db(user_id, video_name, safe_video_name)
        print(f"[DEBUG] vector_db: {vector_db}, path: {path}")
        if not vector_db:
            print(f"[DEBUG] Video data not available")
            return jsonify({"error": "Video data not available"}), 404

        # Perform similarity search
        search_k = 1 if video_name == "all" else 3
        print(f"[DEBUG] search_k: {search_k}")
        docs = vector_db.similarity_search(query, k=search_k)
        print(f"[DEBUG] docs: {docs}")

        # Prepare results in uniform format
        print(f"[DEBUG] Calling prepare_results with group_by_source: {video_name == 'all'}")
        filtered_results, majority_source = prepare_results(
            docs=docs,
            group_by_source=(video_name == "all"),
            get_majority_source=True
        )
        print(f"[DEBUG] filtered_results: {filtered_results}, majority_source: {majority_source}")

        # Determine final video source for Gemini
        final_source = majority_source if video_name == "all" else video_name
        print(f"[DEBUG] final_source: {final_source}")

        # Get transcript for the selected video
        transcript_video = Video.query.filter_by(user_id=user_id, title=final_source).first()
        print(f"[DEBUG] transcript_video: {transcript_video}")
        transcript = transcript_video.transcript
        print(f"[DEBUG] transcript length: {len(transcript) if transcript else 0}")
        
        # Generate response using the determined source
        print(f"[DEBUG] Calling gemini_fallback")
        content = gemini_fallback(user_id, query, transcript)
        print(f"[DEBUG] content: {content[:100]}...")  # Print first 100 chars of content
        
        results = {
            "content": content,
            "timestamp": filtered_results[0]["timestamp"] if filtered_results else "",
            "source": final_source
        }
        print(f"[DEBUG] results: {results}")

        return jsonify({"results": [results]})

    except KeyError as e:
        print(f"[DEBUG] KeyError: {str(e)}")
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400
    except Exception as e:
        print(f"[DEBUG] Exception: {str(e)}")
        current_app.logger.error(f"Query error: {str(e)}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app_bp.route('/create_clone', methods=["POST"])
@jwt_required
def create_clone():
    try:
        user_id = request.user_id
        print("Attempting to create clone...")

        data = request.get_json()
        print(f"Received data: {data}")

        cloudinary_url = data.get("cloudinary_url")

        if not cloudinary_url:
            print("Error: Missing cloudinary_url in request data")
            return jsonify({"error": "Missing cloudinary_url in request data"}), 400

        video = Video.query.filter_by(user_id = user_id, cloudinary_url=cloudinary_url).first()

        if not video:
            print(f"Error: Video with URL '{cloudinary_url}' not found")
            return jsonify({"error": f"Video with URL '{cloudinary_url}' not found"}), 404

        print(f"Found video: {video.video_id}, checking for voice_id...")

        if video.voice_id is not None:
            print("Video already has voice_id, returning conflict...")
            return jsonify({
                "message": "Video already has a voice_id. Cannot assign a new one.",
                "status": "conflict",
                "code": "video_voice_id_exists"
            }), 409

        print(f"Downloading video from Cloudinary: {video.cloudinary_url}")
        downloaded_video = download_video_from_cloudinary(video.cloudinary_url)

        if not downloaded_video:
            print("Error: Failed to download video from Cloudinary")
            return jsonify({"error": "Failed to download video from Cloudinary"}), 500

        path_to_audio_processing = os.path.join(os.getcwd(), str(video.video_id))
        print(f"Creating path: {path_to_audio_processing}")

        try:
            os.makedirs(path_to_audio_processing, exist_ok=True)
        except OSError as e:
            print(f"Error: Failed to create directories: {e}")
            return jsonify({"error": f"Failed to create directories: {e}"}), 500

        try:
            print(f"Calling create_audio_segments with {downloaded_video} and {path_to_audio_processing}")

            create_audio_segments(downloaded_video, path_to_audio_processing)
            audio_segments_dir = os.path.join(path_to_audio_processing, "segments")
            audio_files = utils.get_audio_segment_files_from_dir(audio_segments_dir)
            audio_files_with_duration = utils.get_sorted_audio_with_duration(audio_files)

            print(f"Creating voice clone with file: {audio_files_with_duration[0][0]} and title: {video.title}")
            print(audio_files_with_duration[0][0])
            voice_id = create_voice_clones([audio_files_with_duration[0][0]], video.title).voice_id

            if not voice_id:
                print("Error: Failed to create voice clone")
                return jsonify({"error": "Failed to create voice clone"}), 500

            video.voice_id = voice_id
            db.session.commit()

            print(f"Voice ID '{voice_id}' assigned to video {video.video_id}, returning success")

            return jsonify({
                "message": "Voice ID created and assigned to the video successfully",
                "video_id": video.video_id
            }), 200
        except Exception as e:
            print(f"Error: Audio processing failed: {e}")
            return jsonify({"error": f"Audio processing failed: {e}"}), 500

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"An error occurred: {e}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

@app_bp.route('/speak_message', methods=["POST"])
@jwt_required
def speak_message():
    try:
        print("Attempting to generate speech...")

        data = request.get_json()
        print(f"Received data: {data}")

        cloudinary_url = data.get("cloudinary_url")
        message = data.get("message")

        if not cloudinary_url or not message:
            print("Error: Missing cloudinary_url or message in request data")
            return jsonify({"message": "Missing cloudinary_url or message in request data", "status": "missing_params"}), 400

        video = Video.query.filter_by(cloudinary_url=cloudinary_url).first()

        if not video:
            print(f"Error: Video with URL '{cloudinary_url}' not found")
            return jsonify({"message": f"Video with URL '{cloudinary_url}' not found", "status": "video_not_found"}), 404

        voice_id = video.voice_id

        if not voice_id:
            print("Error: Video does not have an assigned voice_id")
            return jsonify({
                "message": "Video does not have an assigned voice_id",
                "status": "no_voice_id",
                "code": "video_missing_voice_id"
            }), 400

        path_to_audio_processing = os.path.join(os.getcwd(), str(video.video_id), str(voice_id), "generated_output")

        try:
            os.makedirs(path_to_audio_processing, exist_ok=True)
        except OSError as e:
            print(f"Error: Failed to create directories: {e}")
            return jsonify({"message": f"Failed to create directories: {e}", "status": "directory_creation_failed"}), 500

        try:
            print(f"Calling text_to_speech with voice_id: {voice_id} and message: {message}")
            output_file_path = text_to_speech(
                voice_id=voice_id,
                text=message,
                output_path=path_to_audio_processing
            )

            if not output_file_path:
                print("Error: Text-to-speech processing failed to produce output")
                return jsonify({"message": "Text-to-speech processing failed to produce output", "status": "tts_failed"}), 500

            print(f"Sending file: {output_file_path}")
            return send_file(
                output_file_path,
                mimetype='audio/mpeg'  # Adjust mimetype as needed ('audio/wav', etc.)
            )
        except Exception as e:
            print(f"Error: Text-to-speech processing failed: {e}")
            return jsonify({"message": f"Text-to-speech processing failed: {e}", "status": "tts_failed"}), 500

    except Exception as e:
        db.session.rollback()
        current_app.logger.error(f"An unexpected error occurred: {e}")
        return jsonify({"message": "An unexpected error occurred", "status": "unexpected_error"}), 500

@app_bp.route('/delete/video', methods=["DELETE"])
@jwt_required
def remove_video():
    try:
        data = request.get_json()
        user_id = request.user_id
        video_url = data.get('video_url')

        if not video_url:
            return jsonify({"error": "Video URL is required"}), 400

        video = Video.query.filter_by(user_id=user_id, cloudinary_url=video_url).first()

        if not video:
            return jsonify({"error": "Video not found"}), 404

        try:
            response = remove_video_from_cloudinary(str(video.video_id))
        except CloudinaryError as ce:
            return jsonify({"error": f"Cloudinary error: {str(ce)}"}), 500

        try:
            delete_video_from_vector_database(user_id, video_name=video.title, safe_video_name=video.safe_video_name)
        except Exception as e:
            return jsonify({"error": f"VectorDB deletion error: {str(e)}"}), 500

        db.session.delete(video)
        db.session.commit()

        return jsonify({"message": "Video deleted successfully"}), 200

    except Exception as e:
        db.session.rollback()
        print("possible error : ",e)
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app_bp.route('/api/mock', methods=["GET"])
def mock():
    # Get token from Authorization header
    token = request.headers.get('Authorization', '').replace('Bearer ', '')
    user_id = decode_token(token)
    return jsonify({"user_id": user_id}), 200
