import requests
import tempfile
import os
import cloudinary
import cloudinary.uploader

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
