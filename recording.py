import pyaudio
import numpy as np
import whisper
import torch
import threading
import spacy
from groq import Groq
from tavily import TavilyClient
import time

# Initialize Groq and Tavily clients
groq_client = Groq(api_key="")
tavily_client = TavilyClient(api_key="")

# Load NLP and Whisper models
nlp = spacy.load("en_core_web_sm")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = whisper.load_model("tiny", device=device)

# PyAudio configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1  
RATE = 16000 

# Initialize PyAudio instance
audio = pyaudio.PyAudio()
stream = None

# Shared buffer and flag for recording control
buffer = []
recording_active = False
recording_thread = None  # To keep track of the recording thread

def llm(transcription, web_info):
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": '''You are a fact-checker being provided facts to verify with additional data from the web. 
                Respond with either 'true' or 'false' to indicate whether the fact is true or false, along with any clarifying/additional information you may have. 
                Cite sources and provide the correct answer where possible.''',
            },
            {
                "role": "user",
                "content": "Fact: " + transcription + " " + "Supplementary Info: " + web_info,
            }
        ],
        model="llama3-8b-8192",
        max_tokens=100,
        stop=None,
        stream=False,
    )
    return chat_completion.choices[0].message.content

def transcribe_audio(audio_data):
    result = model.transcribe(audio_data, fp16=False)
    transcription = result['text']
    print("Transcription:", transcription)
    doc = nlp(transcription)

    entities_of_interest = [
        'CARDINAL', 'PERCENT', 'MONEY', 'QUANTITY', 'DATE', 'TIME', 'ORDINAL',
    ]
    
    detected_entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in entities_of_interest]

    if detected_entities:
        fact_check(transcription)

def fact_check(transcription):
    print("Fact-checking the following entities:")
    web_info = str(tavily_client.search(transcription, max_results=5))
    response = llm(transcription, web_info)
    print(response)

def start_recording():
    global stream, buffer, recording_active, recording_thread
    if recording_active:
        print("Recording is already active.")
        return
    if stream is None:
        try:
            stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=1024)
        except Exception as e:
            print(f"Failed to open audio stream: {e}")
            return
    recording_active = True
    buffer = []
    print("Recording started.")

    # Start recording in a separate thread
    recording_thread = threading.Thread(target=record_audio)
    recording_thread.start()

def stop_recording():
    global recording_active
    if not recording_active:
        print("Recording is not active.")
        return
    recording_active = False
    print("Stopping recording...")

    # Wait for the recording thread to finish
    if recording_thread is not None:
        recording_thread.join()
    
    # Concatenate the entire buffer and process it
    if buffer:
        audio_data = np.concatenate(buffer)
        transcribe_audio(audio_data)
    else:
        print("No audio data captured.")

def record_audio():
    global buffer, stream
    try:
        while recording_active:
            audio_chunk = stream.read(1024, exception_on_overflow=False)
            audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            buffer.append(audio_np)
    except Exception as e:
        print(f"An error occurred during recording: {e}")
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
            stream = None  # Reset stream to allow reopening
        print("Exiting recording loop.")

def user_input_control():
    while True:
        command = input("Enter 'start' to begin recording, 'stop' to end recording, or 'exit' to quit: ").strip().lower()
        if command == 'start':
            start_recording()
        elif command == 'stop':
            stop_recording()
        elif command == 'exit':
            if recording_active:
                stop_recording()
            break
        else:
            print("Invalid command. Please enter 'start', 'stop', or 'exit'.")

if __name__ == "__main__":
    try:
        user_input_control()
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected. Exiting...")
    finally:
        if recording_active:
            stop_recording()
        if audio is not None:
            audio.terminate()
        print("Audio resources terminated. Goodbye!")
