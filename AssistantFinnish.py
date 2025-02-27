"""

Features:
- Speech recognition with Voice Activity Detection (VAD)
- Microsoft Speech SDK for high-quality Finnish TTS
- OpenAI Assistants API for conversation management
- Wake word detection
- Speech caching for better performance
- Enhanced error handling and graceful degradation
"""

import os
import sys
import time
import json
import wave
import logging
import tempfile
import hashlib
from typing import Optional

# Third-party imports
import pyaudio
import webrtcvad
import winsound
from openai import OpenAI
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('assistant.log')
    ]
)
logger = logging.getLogger()

# Configuration
API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")
# Microsoft Speech API keys
SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION", "westeurope")

if not API_KEY or not ASSISTANT_ID:
    logger.error("OPENAI_API_KEY and OPENAI_ASSISTANT_ID must be set in the environment.")
    sys.exit(1)

# Audio settings
RATE = 16000
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
VAD_MODE = 3
WAKE_WORD = "hei avustaja"
CONVERSATION_FILE = "conversation_data.json"
SPEECH_CACHE_DIR = "speech_cache"

# OpenAI client
client = OpenAI(api_key=API_KEY)

# ----------------------------------------------------------------
# Enhanced Microsoft TTS
# ----------------------------------------------------------------
class EnhancedMicrosoftTTS:
    """Enhanced Text-to-Speech using Microsoft Cognitive Services."""
    
    def __init__(self, speech_key=SPEECH_KEY, speech_region=SPEECH_REGION, 
                 language="fi-FI", voice="fi-FI-SelmaNeural",
                 use_ssml=True, cache_dir=SPEECH_CACHE_DIR):
        self.speech_key = speech_key
        self.speech_region = speech_region
        self.language = language
        self.voice = voice
        self.use_ssml = use_ssml
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Setup speech SDK if keys are provided
        self.speech_config = None
        self.speech_synthesizer = None
        
        if speech_key and speech_region:
            self._setup_speech_sdk()
        else:
            logger.warning("Microsoft Speech keys not provided. Will use fallback beep sounds.")
    
    def _setup_speech_sdk(self):
        """Setup the Microsoft Speech SDK."""
        try:
            # Setup the configuration
            self.speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
            self.speech_config.speech_synthesis_language = self.language
            self.speech_config.speech_synthesis_voice_name = self.voice
            
            # Setup the synthesizer
            self.speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
            
            # Test the synthesizer with a quick silent test
            logger.info("Testing Microsoft Speech SDK...")
            test_ssml = '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="fi-FI"><voice name="fi-FI-SelmaNeural"><prosody volume="silent">Test</prosody></voice></speak>'
            test_result = self.speech_synthesizer.speak_ssml_async(test_ssml).get()
            
            if test_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info("Microsoft Speech SDK initialized and tested successfully")
            else:
                logger.warning(f"Speech test returned: {test_result.reason}")
                
        except ImportError:
            logger.error("Microsoft Speech SDK not installed. Run: pip install azure-cognitiveservices-speech")
            self.speech_synthesizer = None
        except Exception as e:
            logger.error(f"Error setting up Microsoft Speech SDK: {e}")
            self.speech_synthesizer = None
    
    def _get_cache_filename(self, text):
        """Generate a cache filename based on the text and voice."""
        # Create a unique hash based on text and voice parameters
        hash_input = f"{self.voice}:{self.language}:{text}"
        hash_value = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_value}.wav")
    
    def _create_ssml(self, text):
        """Create SSML markup for more expressive speech."""
        # Escape XML special characters
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;")
        
        # Basic SSML template with natural pauses for punctuation
        ssml = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{self.language}">
            <voice name="{self.voice}">
                <prosody rate="0%" pitch="0%">
                    {text}
                </prosody>
            </voice>
        </speak>
        """
        return ssml
    
    def speak(self, text):
        """Speak text using Microsoft TTS with caching or fallback to beep sounds."""
        try:
            print(f"Assistant: {text}")
            
            # Check if we have a valid synthesizer
            if self.speech_synthesizer:
                # Check cache first
                cache_file = self._get_cache_filename(text)
                
                if os.path.exists(cache_file):
                    logger.info(f"Using cached speech file: {cache_file}")
                    # Play the cached file
                    winsound.PlaySound(cache_file, winsound.SND_FILENAME)
                    return True
                
                # Generate speech
                logger.info(f"Generating TTS for: {text[:30]}...")
                
                try:
                    # Use SSML if enabled
                    if self.use_ssml:
                        ssml = self._create_ssml(text)
                        # Generate speech from SSML and save to file
                        audio_config = speechsdk.audio.AudioConfig(filename=cache_file)
                        file_synthesizer = speechsdk.SpeechSynthesizer(
                            speech_config=self.speech_config, 
                            audio_config=audio_config
                        )
                        result = file_synthesizer.speak_ssml_async(ssml).get()
                    else:
                        # Generate speech from plain text and save to file
                        audio_config = speechsdk.audio.AudioConfig(filename=cache_file)
                        file_synthesizer = speechsdk.SpeechSynthesizer(
                            speech_config=self.speech_config, 
                            audio_config=audio_config
                        )
                        result = file_synthesizer.speak_text_async(text).get()
                    
                    # Check result of file generation
                    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                        logger.info(f"Speech synthesis saved to file: {cache_file}")
                        # Now play the file
                        winsound.PlaySound(cache_file, winsound.SND_FILENAME)
                        return True
                    else:
                        raise Exception(f"Speech synthesis failed: {result.reason}")
                        
                except Exception as file_error:
                    # If file creation fails, try direct synthesis
                    logger.error(f"File-based synthesis failed: {file_error}")
                    logger.info("Trying direct speech synthesis...")
                    
                    if self.use_ssml:
                        result = self.speech_synthesizer.speak_ssml_async(self._create_ssml(text)).get()
                    else:
                        result = self.speech_synthesizer.speak_text_async(text).get()
                    
                    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                        logger.info("Direct speech synthesis successful")
                        return True
                    else:
                        # Fall back to beep sounds
                        raise Exception(f"Direct speech synthesis failed: {result.reason}")
            else:
                raise Exception("Speech synthesizer not available")
                
        except Exception as e:
            logger.error(f"TTS error: {e}")
            # Fallback to a simple beep sequence
            try:
                self._play_beep_sequence()
            except:
                pass
            return False
    
    def _play_beep_sequence(self):
        """Play a sequence of beeps to indicate assistant speaking."""
        try:
            # Play beeps to indicate assistant is speaking
            winsound.Beep(880, 200)  # Higher tone
            winsound.Beep(660, 300)  # Middle tone
            winsound.Beep(440, 400)  # Lower tone
        except Exception as e:
            logger.error(f"Error playing beep sequence: {e}")

# ----------------------------------------------------------------
# Speech Recognition with VAD
# ----------------------------------------------------------------
class SpeechRecognizer:
    """Speech recognition with Voice Activity Detection."""
    
    def __init__(self):
        self.vad = webrtcvad.Vad(mode=VAD_MODE)

    def record_until_silence(self) -> Optional[bytes]:
        """Record from microphone until silence is detected."""
        logger.info("Initializing audio recording...")
        p = pyaudio.PyAudio()
        
        try:
            stream = p.open(format=pyaudio.paInt16,
                          channels=1,
                          rate=RATE,
                          input=True,
                          frames_per_buffer=CHUNK_SIZE)
        except Exception as e:
            logger.error(f"Error opening audio stream: {e}")
            p.terminate()
            return None

        logger.info("Listening for voice activity...")

        # Flush audio buffer
        try:
            for _ in range(5):
                stream.read(CHUNK_SIZE, exception_on_overflow=False)
        except Exception as e:
            logger.error(f"Error flushing audio buffer: {e}")
            stream.stop_stream()
            stream.close()
            p.terminate()
            return None

        frames = []
        num_silent_chunks = 0
        recording_started = False
        max_recording_time = 10
        max_chunks = int(max_recording_time * 1000 / CHUNK_DURATION_MS)
        chunk_count = 0

        try:
            while chunk_count < max_chunks:
                chunk_count += 1
                
                # Non-blocking read with a short timeout
                raw_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                
                # Check for keyboard interrupt periodically
                if chunk_count % 10 == 0:
                    time.sleep(0.001)
                
                # Determine if chunk has speech
                if self.is_speech(raw_data):
                    frames.append(raw_data)
                    recording_started = True
                    num_silent_chunks = 0
                else:
                    if recording_started:
                        num_silent_chunks += 1
                        frames.append(raw_data)
                        
                        # if enough silent chunks, break
                        if num_silent_chunks > 50:
                            break
        except KeyboardInterrupt:
            logger.info("Recording interrupted by user")
        except Exception as e:
            logger.error(f"Error while recording: {e}")
        finally:
            try:
                stream.stop_stream()
                stream.close()
            except:
                pass
            try:
                p.terminate()
            except:
                pass

        if not frames or not recording_started:
            logger.info("No speech detected")
            return None
            
        logger.info(f"Recording complete: {len(frames)} chunks")
        return b''.join(frames)

    def is_speech(self, chunk: bytes) -> bool:
        """Check if audio chunk contains speech."""
        try:
            return self.vad.is_speech(chunk, sample_rate=RATE)
        except Exception:
            return self._calculate_energy(chunk) > 300

    def _calculate_energy(self, chunk: bytes) -> float:
        """Calculate audio energy."""
        import array
        import math
        
        data = array.array('h', chunk)
        return math.sqrt(sum(x*x for x in data) / len(data)) if data else 0

    def save_wav(self, raw_data: bytes, filename: str):
        """Save raw audio data as a WAV file."""
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)
            wf.writeframes(raw_data)

    def transcribe(self, audio_bytes: bytes, language="fi") -> str:
        """Transcribe audio using OpenAI Whisper."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_wav = temp_file.name
        temp_file.close()
        
        try:
            self.save_wav(audio_bytes, temp_wav)
            logger.info(f"Saved audio to temporary file: {temp_wav}")
            
            logger.info("Transcribing audio...")
            with open(temp_wav, "rb") as f:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language=language
                )
                text = result.text.strip()
                logger.info(f"Transcription result: {text}")
                return text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
        finally:
            try:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
            except:
                pass

# ----------------------------------------------------------------
# Microsoft Speech Recognizer (Alternative to Whisper)
# ----------------------------------------------------------------
class MicrosoftSpeechRecognizer:
    """Speech recognition using Microsoft Cognitive Services."""
    
    def __init__(self, speech_key=SPEECH_KEY, speech_region=SPEECH_REGION, 
                 language="fi-FI"):
        self.speech_key = speech_key
        self.speech_region = speech_region
        self.language = language
        self.speech_recognizer = None
        
        if speech_key and speech_region:
            self._setup_speech_sdk()
        else:
            logger.warning("Microsoft Speech keys not provided. Speech recognition unavailable.")
    
    def _setup_speech_sdk(self):
        """Setup the Microsoft Speech SDK for recognition."""
        try:
            # Setup the configuration
            self.speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
            self.speech_config.speech_recognition_language = self.language
            
            # Create audio config for microphone
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            
            # Create recognizer
            self.speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config, 
                audio_config=audio_config
            )
            
            logger.info("Microsoft Speech Recognizer initialized")
        except ImportError:
            logger.error("Microsoft Speech SDK not installed")
            self.speech_recognizer = None
        except Exception as e:
            logger.error(f"Error setting up Microsoft Speech Recognizer: {e}")
            self.speech_recognizer = None
    
    def recognize(self, timeout=10):
        """
        Recognize speech using Microsoft Speech SDK.
        Returns the recognized text or None if recognition failed.
        """
        if not self.speech_recognizer:
            logger.error("Speech recognizer not available")
            return None
            
        try:
            logger.info("Listening with Microsoft Speech Recognizer...")
            result = self.speech_recognizer.recognize_once_async().get()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                logger.info(f"RECOGNIZED: {result.text}")
                return result.text
            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.warning("NOMATCH: Speech could not be recognized.")
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                logger.warning(f"CANCELED: Reason={cancellation.reason}")
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    logger.error(f"CANCELED: ErrorCode={cancellation.error_code}")
                    logger.error(f"CANCELED: ErrorDetails={cancellation.error_details}")
            
            return None
            
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            return None

# ----------------------------------------------------------------
# Conversation Manager
# ----------------------------------------------------------------
class ConversationManager:
    """Manages conversation with OpenAI Assistant."""
    
    def __init__(self, assistant_id: str, data_file: str = CONVERSATION_FILE):
        self.assistant_id = assistant_id
        self.data_file = data_file
        self.thread_id = None

        self._load_conversation_data()
        if self.thread_id is None:
            self._create_thread()
            # Initialize with instructions
            self.add_user_message(
                "Olet ääniavustaja, joka ymmärtää ja vastaa suomeksi. Keskustele kanssani vain suomeksi."
            )

    def _create_thread(self):
        """Create a new thread."""
        try:
            logger.info("Creating new conversation thread...")
            chat_thread = client.beta.threads.create()
            self.thread_id = chat_thread.id
            self._save_conversation_data()
            logger.info(f"Created new conversation thread: {self.thread_id}")
        except Exception as e:
            logger.error(f"Failed to create thread: {e}")
            raise

    def _load_conversation_data(self):
        """Load conversation data from file."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.thread_id = data.get("thread_id", None)
                logger.info(f"Loaded conversation thread: {self.thread_id}")
            except Exception as e:
                logger.warning(f"Could not load conversation data: {e}. Starting fresh.")

    def _save_conversation_data(self):
        """Save conversation data to file."""
        data = {
            "thread_id": self.thread_id
        }
        try:
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save conversation data: {e}")

    def add_user_message(self, message: str):
        """Add user message to the conversation."""
        if not self.thread_id:
            self._create_thread()
            
        try:
            logger.info(f"Adding user message: {message}")
            client.beta.threads.messages.create(
                thread_id=self.thread_id,
                role="user",
                content=message
            )
        except Exception as e:
            logger.error(f"Error adding user message: {e}")
            if "Thread not found" in str(e):
                logger.info("Thread not found. Creating a new one.")
                self._create_thread()
                self.add_user_message(message)

    def run_assistant(self) -> Optional[str]:
        """Run the assistant and get response."""
        if not self.thread_id:
            return None

        try:
            logger.info("Running assistant...")
            run = client.beta.threads.runs.create(
                thread_id=self.thread_id,
                assistant_id=self.assistant_id
            )

            # Poll for completion
            logger.info("Waiting for assistant response...")
            wait_time = 1
            max_attempts = 10
            attempts = 0

            while attempts < max_attempts:
                run = client.beta.threads.runs.retrieve(
                    thread_id=self.thread_id, 
                    run_id=run.id
                )
                
                if run.status in ["completed", "failed", "cancelled"]:
                    break
                    
                logger.info(f"Run status: {run.status}, waiting {wait_time}s...")
                time.sleep(wait_time)
                wait_time = min(wait_time * 1.5, 5)
                attempts += 1

            if run.status == "failed":
                logger.error(f"Assistant run failed: {run.last_error}")
                return None
            elif run.status == "completed":
                logger.info("Assistant run completed successfully")
            else:
                logger.warning(f"Assistant run did not complete: {run.status}")
                return None

            # Get the latest assistant message
            logger.info("Retrieving assistant message...")
            messages = client.beta.threads.messages.list(
                thread_id=self.thread_id
            ).data
            
            messages_sorted = sorted(messages, key=lambda x: x.created_at, reverse=True)
            for msg in messages_sorted:
                if msg.role == "assistant":
                    response = self.extract_text_from_msg(msg)
                    logger.info(f"Got assistant response: {response}")
                    return response

            logger.warning("No assistant message found")
            return None
            
        except Exception as e:
            logger.error(f"Error running assistant: {e}")
            return None

    @staticmethod
    def extract_text_from_msg(msg):
        """Extract text from message object."""
        content = msg.content
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if hasattr(block, "text"):
                    block_text = getattr(block, "text")
                    if hasattr(block_text, "value"):
                        text_parts.append(str(block_text.value))
                    else:
                        text_parts.append(str(block_text))
                elif isinstance(block, dict) and "text" in block:
                    block_text = block["text"]
                    if isinstance(block_text, dict):
                        text_parts.append(str(block_text.get("value", block_text)))
                    else:
                        text_parts.append(str(block_text))
                else:
                    text_parts.append(str(block))
            return " ".join(text_parts).strip()
        elif isinstance(content, str):
            return content.strip()
        return str(content)

# ----------------------------------------------------------------
# Main Application
# ----------------------------------------------------------------
def main():
    """Run the application."""
    print("Finnish Voice Assistant with Microsoft TTS")
    print("=====================================")
    print("Say 'hei avustaja' to start or press Enter to type.")
    print("Say 'lopeta' to exit.")
    print()
    
    try:
        # Initialize components
        tts = EnhancedMicrosoftTTS(
            speech_key=SPEECH_KEY,
            speech_region=SPEECH_REGION,
            language="fi-FI",
            voice="fi-FI-SelmaNeural",  # Alternative voices: HarriNeural (male), NooraNeural (female)
            use_ssml=True
        )
        
        recognizer = SpeechRecognizer()
        conversation = ConversationManager(assistant_id=ASSISTANT_ID)
        
        # Initial greeting
        print("Starting assistant...")
        tts.speak("Hei, kuinka voin auttaa sinua tänään?")
        
        # Main loop
        running = True
        while running:
            try:
                # Check for keyboard input
                print("\nListening for speech or press Enter to type...")
                
                # Non-blocking input check
                import msvcrt
                if msvcrt.kbhit():
                    # Clear the key buffer
                    while msvcrt.kbhit():
                        msvcrt.getch()
                    # Get typed input
                    user_input = input("Type your message: ").strip()
                    if user_input:
                        # Process command
                        print(f"You typed: {user_input}")
                        
                        # Check for exit
                        if user_input.lower() in ["lopeta", "exit", "quit", "poistu"]:
                            tts.speak("Suljetaan nyt, näkemiin!")
                            running = False
                            break
                            
                        # Send to assistant
                        conversation.add_user_message(user_input)
                        response = conversation.run_assistant()
                        
                        if response:
                            tts.speak(response)
                        else:
                            tts.speak("Valitettavasti en saanut vastausta.")
                    
                    continue
                
                # Listen for speech
                audio_data = recognizer.record_until_silence()
                
                if not audio_data:
                    continue
                
                # Transcribe
                speech_text = recognizer.transcribe(audio_data)
                
                if not speech_text:
                    continue
                
                print(f"You said: {speech_text}")
                
                # Check for wake word
                if WAKE_WORD.lower() in speech_text.lower():
                    print("Wake word detected!")
                    tts.speak("Kyllä, miten voin auttaa?")
                    
                    # Listen for command
                    print("Listening for command...")
                    command_audio = recognizer.record_until_silence()
                    
                    if not command_audio:
                        print("No command detected.")
                        continue
                        
                    command_text = recognizer.transcribe(command_audio)
                    
                    if not command_text:
                        print("Could not transcribe command.")
                        continue
                        
                    print(f"Command: {command_text}")
                    
                    # Check for exit
                    if any(cmd in command_text.lower() for cmd in ["lopeta", "hyvästi", "pois", "exit"]):
                        tts.speak("Suljetaan nyt, näkemiin!")
                        running = False
                        break
                        
                    # Send to assistant
                    conversation.add_user_message(command_text)
                    response = conversation.run_assistant()
                    
                    if response:
                        tts.speak(response)
                    else:
                        tts.speak("Valitettavasti en saanut vastausta.")
                        
                else:
                    # Treat as direct command
                    # Check for exit
                    if any(cmd in speech_text.lower() for cmd in ["lopeta", "hyvästi", "pois", "exit"]):
                        tts.speak("Suljetaan nyt, näkemiin!")
                        running = False
                        break
                        
                    # Send to assistant
                    conversation.add_user_message(speech_text)
                    response = conversation.run_assistant()
                    
                    if response:
                        tts.speak(response)
                    else:
                        tts.speak("Valitettavasti en saanut vastausta.")
                
            except KeyboardInterrupt:
                print("\nInterrupted. Exiting...")
                tts.speak("Suljetaan nyt, näkemiin!")
                running = False
                break
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print(f"Error: {e}")
                # Continue running despite errors
                
    except KeyboardInterrupt:
        print("\nExiting assistant. Näkemiin!")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"Fatal error: {e}")
        
    print("Voice assistant terminated.")

if __name__ == "__main__":
    main()
