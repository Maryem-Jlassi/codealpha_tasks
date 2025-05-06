from transformers import pipeline
from gtts import gTTS
import speech_recognition as sr
import os
from io import BytesIO

def translate_text(source_lang, target_lang, text):
    if source_lang == target_lang:
        raise ValueError("Source and target languages must be different.")
        
    translator = pipeline("translation", model=f"Helsinki-NLP/opus-mt-{source_lang[:2]}-{target_lang[:2]}")
    translated_text = translator(text)[0]["translation_text"]
    
    tts = gTTS(translated_text, lang=target_lang.split("-")[0])
    audio_path = f"static/audio/{source_lang}-{target_lang}.mp3"
    tts.save(audio_path)
    return translated_text, f"/{audio_path}"

def process_voice_input(voice_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(voice_file) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)