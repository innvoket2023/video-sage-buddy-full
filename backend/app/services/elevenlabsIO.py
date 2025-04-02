import contextlib
from dotenv import load_dotenv
from elevenlabs import ElevenLabs, save
import os
import random

load_dotenv()

api = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(api_key=api)

def create_voice_clones(file_paths, title, description = None):
    #Make sure to pass the file_paths as a list
    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(open(file_path, "rb")) for file_path in file_paths]
        
        response = client.voices.add(
            name=title,
            files=files,  # Pass the list of open file objects
            description=description
        )
    
    return response

def text_to_speech(voice_id, text, model_id = None, output_path = os.getcwd()):
    audio = client.text_to_speech.convert(
    voice_id=voice_id,
    output_format="mp3_44100_128",
    text="The first move is what sets everything in motion.",
    model_id="eleven_multilingual_v2",
)
    dir= os.path.join(output_path, voice_id)
    os.makedirs(dir, exist_ok=True)
    full_path_to_audio_file = os.path.join(dir, "test.mp3")
    save(audio, full_path_to_audio_file) 

def delete_voice(voice_id):
    client.voices.delete(
        voice_id = voice_id
    )
    return client

def list_similar_voices(file, similarity_threshold = random.uniform(0.1, 0.5), top_k = None):
    client.voices.get_similar_library_voices(
        audio_file = file,
        similarity_threshold = similarity_threshold,
        top_k = top_k
    )
    return client

if __name__ == "__main__":
    # print(create_voice_clones([r"C:\Users\Ansh\Desktop\coding\video-sage-buddy-full\backend\segments\chunk-000.wav"], title = "test"))
    #4w9uhKrfh0MSfaC8YE0Q
    text_to_speech(text = "The first move is what sets everything in motion.", voice_id = "4w9uhKrfh0MSfaC8YE0Q")

