from moviepy import VideoFileClip, AudioArrayClip
import numpy as np
import wave
import collections
import webrtcvad
import collections
import contextlib
import sys
import os
from pydub import AudioSegment


def extract_audio(input_video_path, output_audio_path, output_audio_name):
    video_file = VideoFileClip(input_video_path) 
    full_path = os.path.join(output_audio_path, output_audio_name)
    audio_file = video_file.audio
    audio_file.write_audiofile(full_path, fps = 16000, nbytes = 2, codec = "pcm_s16le", bitrate = "256k")
    audio_file.close()
    video_file.close()
    convert_to_mono(full_path)

def convert_to_mono(path):
    sound = AudioSegment.from_wav(path)
    sound = sound.set_channels(1)
    sound.export(path, format="wav")

def convert_bit_depth(path):
    pass 

def convert_sample_rate(path):
    pass

def read_wave(path):
    """Reads a .wav file.

    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        if not num_channels == 1:
            pass
            # convert_to_mono(path)
            # print(f"Issue in number of channels: {num_channels}")
        sample_width = wf.getsampwidth()
        if not sample_width == 2:
            convert_bit_depth(path)
            # print(f"Issue in bit depth AKA sample_width: {sample_width}")
        sample_rate = wf.getframerate()
        if not sample_rate in (8000, 16000, 32000, 48000):
            convert_sample_rate(path)
            # print(f"Issue in sample_rate: {sample_rate}")
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def write_wave(filename, output_dir, audio, sample_rate):
    """Writes a .wav file to 'segments' subdirectory."""
    segments_dir = os.path.join(output_dir, "segments")
    
    # Create only the directory (not including the filename)
    os.makedirs(segments_dir, exist_ok=True)  
    
    full_path = os.path.join(segments_dir, filename)
    
    with contextlib.closing(wave.open(full_path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

def create_audio_segments(input_audio_path, output_audio_path=os.getcwd(), output_audio_name="audio.wav", aggressiveness=0):
    extract_audio(input_audio_path, output_audio_path, output_audio_name)
    full_path = os.path.join(output_audio_path, output_audio_name)
    audio, sample_rate = read_wave(full_path)
    vad = webrtcvad.Vad(aggressiveness)
    frames = list(frame_generator(30, audio, sample_rate))
    segments = vad_collector(sample_rate, 30, 300, vad, frames)
    
    for i, segment in enumerate(segments):
        segment_name = f'chunk-{i:03d}.wav'  # Fixed filename format
        print(f'Writing {segment_name}')
        write_wave(segment_name, output_audio_path, segment, sample_rate)

if __name__ == '__main__':
    input_audio_path = r"C:\Users\Ansh\Downloads\Tcs_coding_test.mp4" 
    # output_audio_path = os.getcwd()
    # output_audio_name = "audio.wav"
    create_audio_segments(input_audio_path)
