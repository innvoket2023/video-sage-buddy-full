from pydub import AudioSegment
import os

def get_sorted_audio_with_duration(files, min_duration = 0, max_duration = 30):
    """Files: should be list of file paths
    min_duration: 0 (default in seconds)
    max_duration: 30 (default in seconds)

    filter files less than equal to 30 seconds"""

    sorted_files_by_duration = []
    for file in files:
        audio = AudioSegment.from_file(file)
        duration = audio.duration_seconds
        if duration > min_duration and duration <= max_duration:
            sorted_files_by_duration.append((file, duration))
        else:
            continue
    return sorted(sorted_files_by_duration, key=lambda x: x[1])

def get_audio_segment_files_from_dir(dir_path):
    files = []
    for file in os.scandir(dir_path):
        if file.is_file() and file.name.lower().endswith('.wav') and file.name != "audio.wav":
            files.append(file.path)
    return files
            

if __name__ == "__main__":
    list_dir = get_audio_segment_files_from_dir(r"C:\Users\Ansh\Desktop\coding\video-sage-buddy-full\backend\segments")
    print(get_sorted_audio_with_duration(list_dir))
