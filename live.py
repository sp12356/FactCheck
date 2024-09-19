import pyaudio
import numpy as np
import whisper
import torch
import threading
import spacy
from groq import Groq
from tavily import TavilyClient

groq_client = Groq(api_key="")
tavily_client = TavilyClient(api_key="")

nlp = spacy.load("en_core_web_sm")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = whisper.load_model("tiny", device=device)

FORMAT = pyaudio.paInt16
CHANNELS = 1  
RATE = 16000 
CHUNK = 1024 

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)

print("Recording and transcribing...")

buffer = []

def llm(transcription, web_info):
    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": '''You are a fact-checker being provided facts to verify with additional data from the web. 
                Respond with 'true' or 'false' to indicate whether the fact is true or false, along with any clarifying/additional information you may have. 
                Cite sources and provide the correct answer where possible.''',
            },
            {
                "role": "user",
                "content": "Fact: " + transcription+" " + "Supplementary Info: "+ web_info,
            }
        ],
        model="llama3-8b-8192",
        max_tokens=100,
        stop=None,
        stream=False,
    )
    return (chat_completion.choices[0].message.content)


def transcribe_audio(audio_data):
    result = model.transcribe(audio_data, fp16=False)
    transcription = result['text']
    print("Transcription:", transcription)
    doc = nlp(transcription)

    entities_of_interest = [
    'CARDINAL', 'PERCENT', 'MONEY', 'QUANTITY', 'DATE', 'TIME', 'ORDINAL',
    ]
    '''
    entities_of_interest = [
    'CARDINAL', 'PERCENT', 'MONEY', 'QUANTITY', 'DATE', 'TIME', 'ORDINAL',
    'PERSON', 'ORG', 'GPE', 'LOC', 'EVENT', 'WORK_OF_ART', 'LAW', 'LANGUAGE',
    'PRODUCT', 'NORP', 'FAC'
    ]'''
    detected_entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in entities_of_interest]

    if detected_entities:
        fact_check(transcription)

def fact_check(transcription):
    print("Fact-checking the following entities:")
    web_info = str(tavily_client.search(transcription, max_results=5))
    response = llm(transcription, web_info)
    print(response)

try:
    while True:
        audio_chunk = stream.read(CHUNK, exception_on_overflow=False)
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        buffer.append(audio_np)
        if len(buffer) >= int(RATE / CHUNK * 5):
            audio_data = np.concatenate(buffer)
            buffer = []
            threading.Thread(target=transcribe_audio, args=(audio_data,)).start()

except (KeyboardInterrupt, Exception) as e:
    print(f"An error occurred: {e}")

finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
