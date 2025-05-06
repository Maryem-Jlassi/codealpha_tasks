import os
import logging
import tempfile
import uuid
from datetime import datetime
from gtts import gTTS
import speech_recognition as sr

logger = logging.getLogger(__name__)

def text_to_speech(text, language="en-GB"):
    """
    Convert text to speech using Google Text-to-Speech.
    Returns the path to the generated audio file.
    
    Args:
        text (str): Text to convert to speech
        language (str): Language code (e.g., 'en-GB', 'fr-FR', 'ar-SA')
    Returns:
        str: URL path to the audio file or None if error
    """
    try:
        # Create audio directory if it doesn't exist
        os.makedirs("static/audio", exist_ok=True)
        
        # Create a unique filename
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.mp3"
        audio_path = os.path.join("static/audio", filename)
        
        # Extract language code (e.g., 'en' from 'en-GB')
        lang_code = language.split('-')[0]
        
        # Generate TTS
        tts = gTTS(text=text, lang=lang_code)
        tts.save(audio_path)
        
        logger.info(f"Generated audio file: {filename}")
        return f"/static/audio/{filename}"
    
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        return None

def process_speech_to_text(audio_data, language="en-GB"):
    """
    Process speech to text from audio data.
    
    Args:
        audio_data (bytes): Audio data in bytes
        language (str): Language code (e.g., 'en-GB', 'fr-FR', 'ar-SA')
    Returns:
        dict: Contains recognized text and detected language
    """
    try:
        recognizer = sr.Recognizer()
        
        # Configure recognition settings
        recognizer.energy_threshold = 4000
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8
        
        # Save audio data to temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            temp_audio.write(audio_data)
            temp_audio_path = temp_audio.name
        
        try:
            # Load audio file and recognize speech
            with sr.AudioFile(temp_audio_path) as source:
                # Adjust for ambient noise
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.record(source)
                
                # Attempt to detect language if set to auto
                if language.lower() == 'auto':
                    try:
                        result = recognizer.recognize_google(audio, show_all=True)
                        detected_language = result['alternative'][0].get('language', 'en-GB')
                        text = result['alternative'][0]['transcript']
                    except:
                        # Fallback to specified language or English
                        detected_language = language
                        text = recognizer.recognize_google(audio, language=language)
                else:
                    detected_language = language
                    text = recognizer.recognize_google(audio, language=language)
                
                return {
                    'text': text,
                    'detected_language': detected_language,
                    'success': True
                }
                
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_audio_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {e}")
            
    except sr.UnknownValueError:
        logger.warning("Speech recognition could not understand audio")
        return {
            'text': None,
            'error': "Could not understand audio",
            'success': False
        }
    except sr.RequestError as e:
        logger.error(f"Speech recognition service error: {e}")
        return {
            'text': None,
            'error': "Speech recognition service error",
            'success': False
        }
    except Exception as e:
        logger.error(f"STT error: {str(e)}")
        return {
            'text': None,
            'error': str(e),
            'success': False
        }

def detect_speech_language(audio_data):
    """
    Detect the language of speech in audio data.
    
    Args:
        audio_data (bytes): Audio data in bytes
    Returns:
        str: Detected language code or 'en-GB' if detection fails
    """
    try:
        return process_speech_to_text(audio_data, language='auto')['detected_language']
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        return 'en-GB'