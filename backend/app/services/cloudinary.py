import requests
import tempfile
import os
import cloudinary
import cloudinary.uploader
import cloudinary.api
from dotenv import load_dotenv

load_dotenv()

# Configure Cloudinary with your credentials
cloudinary.config(
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key = os.getenv("CLOUDINARY_API_KEY"),
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
    secure = True
)

def cloudinary_usage():
    try:
        # Call the usage API method
        usage_data = cloudinary.api.usage()

        # Print the usage data
        return usage_data

    except Exception as e:
        print(f"Error fetching usage data: {e}")

def download_video_from_cloudinary(video_url):
    """Downloads a video from Cloudinary URL to a temporary file"""
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

def remove_video_from_cloudinary(video_public_id):
    result = cloudinary.uploader.destroy(video_public_id, resource_type="video")
    return result


