"""
==========================================================
 main.py - Example Voice Assistant with VAD & Persistence
==========================================================

Requirements:
    pip install python-dotenv pyaudio webrtcvad openai gTTS playsound
    # On some systems, you may also need: sudo apt-get install portaudio19-dev
    # For webrtcvad:
    # pip install webrtcvad
    # pip install playsound==1.2.2 if voice wont work

Description:
    - Demonstrates:
      1) Voice Activity Detection with webrtcvad
      2) Persistent conversation management
      3) Modular code structure
      4) Enhanced TTS with caching
      5) UX improvements (wake word, fallback input)
"""

import os
import sys
import time
import json
import logging
import wave
import hashlib
from typing import Optional

import pyaudio
import webrtcvad
import playsound
from dotenv import load_dotenv
from gtts import gTTS
import openai
from openai import OpenAI

load_dotenv()

logging.basicConfig(level=logging.INFO)

# ----------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------
API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")
if not API_KEY or not ASSISTANT_ID:
    logging.error("OPENAI_API_KEY and OPENAI_ASSISTANT_ID must be set in the environment.")
    sys.exit(1)

# Tweak these parameters based on your environment
RATE = 16000
CHUNK_DURATION_MS = 30   # must be 10, 20 or 30 ms
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
MAX_SILENCE_CHUNKS = 100  # how many chunks of silence to detect before stopping
VAD_MODE = 3             # aggressiveness of the VAD: 0-3

# A wake word for user experience
WAKE_WORD = "hei avustaja"  # speak in Finnish to wake up the assistant

# File to store conversation data
CONVERSATION_FILE = "conversation_data.json"

# Instantiate the new OpenAI client
client = OpenAI(api_key=API_KEY)

# ----------------------------------------------------------------
# TTS with simple caching
# ----------------------------------------------------------------
class TextToSpeech:
    """
    Handles text-to-speech using gTTS with simple caching of audio files.
    """
    def __init__(self, lang="fi", cache_dir="tts_cache"):
        self.lang = lang
        self.cache_dir = cache_dir
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def speak(self, text: str):
        """Generate TTS audio and play it, using cache if available."""
        # Create a hash key based on the text to use cached file if possible
        hash_key = hashlib.md5(text.encode('utf-8')).hexdigest()
        mp3_file = os.path.join(self.cache_dir, f"{hash_key}.mp3")

        if not os.path.exists(mp3_file):
            try:
                tts = gTTS(text=text, lang=self.lang)
                tts.save(mp3_file)
            except Exception as e:
                logging.error(f"Failed to create TTS: {e}")
                return

        # Play audio
        try:
            playsound.playsound(mp3_file, True)
        except Exception as e:
            logging.error(f"Error playing TTS file: {e}")


# ----------------------------------------------------------------
# Speech Recognition with VAD
# ----------------------------------------------------------------
class SpeechRecognizer:
    """
    Uses PyAudio + webrtcvad to do voice-activated recording.
    Then sends recorded audio to OpenAI Whisper for transcription.
    """
    def __init__(self):
        self.vad = webrtcvad.Vad(mode=VAD_MODE)

    def record_until_silence(self) -> Optional[bytes]:
        """
        Record from microphone until we detect a period of silence.
        Returns raw audio data (16kHz, 1 channel, 16-bit).
        """
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK_SIZE)

        logging.info("Listening for voice activity...")

        # Optional: flush any stale audio buffer before capturing
        for _ in range(50):
            stream.read(CHUNK_SIZE, exception_on_overflow=False)

        frames = []
        num_silent_chunks = 0
        recording_started = False

        try:
            while True:
                raw_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
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
                        if num_silent_chunks > MAX_SILENCE_CHUNKS:
                            break

        except KeyboardInterrupt:
            logging.info("User interrupted recording.")
            pass
        except Exception as e:
            logging.error(f"Error while recording: {e}")
            pass
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

        if not frames:
            return None
        # Join all recorded frames
        return b''.join(frames)

    def is_speech(self, chunk: bytes) -> bool:
        """
        Use webrtcvad to check if chunk of audio contains speech.
        """
        return self.vad.is_speech(chunk, sample_rate=RATE)

    def save_wav(self, raw_data: bytes, filename: str):
        """
        Save raw audio data as a WAV file.
        """
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(pyaudio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)
            wf.writeframes(raw_data)

    def transcribe(self, audio_bytes: bytes, language="fi") -> str:
        """
        Send audio bytes to OpenAI's Whisper for transcription.
        """
        # Save to temp WAV
        temp_wav = "temp_audio.wav"
        self.save_wav(audio_bytes, temp_wav)
        text = ""

        try:
            with open(temp_wav, "rb") as f:
                result = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    language=language
                )
                text = result.text.strip()
        except Exception as e:
            logging.error(f"Transcription error: {e}")
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)

        return text


# ----------------------------------------------------------------
# Conversation Manager (Persistent)
# ----------------------------------------------------------------
class ConversationManager:
    """
    Manages the conversation state with the OpenAI "threads" API
    and stores the thread_id and messages to a local JSON file.
    """
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
        """
        Create a new thread and save it.
        """
        chat_thread = client.beta.threads.create()
        self.thread_id = chat_thread.id
        self._save_conversation_data()
        logging.info(f"Created new conversation thread: {self.thread_id}")

    def _load_conversation_data(self):
        """Load conversation data from file if it exists."""
        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.thread_id = data.get("thread_id", None)
            except Exception:
                logging.warning("Could not load conversation data. Starting fresh.")

    def _save_conversation_data(self):
        """Save conversation data (thread_id, etc.) to file."""
        data = {
            "thread_id": self.thread_id
        }
        try:
            with open(self.data_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Failed to save conversation data: {e}")

    def add_user_message(self, message: str):
        """
        Add user message to the conversation.
        """
        if not self.thread_id:
            self._create_thread()
        client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role="user",
            content=message
        )

    def run_assistant(self) -> Optional[str]:
        """
        Execute the assistant run and return the assistant's latest text response.
        """
        if not self.thread_id:
            return None

        run = client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id
        )

        # Poll for completion
        wait_time = 1
        max_attempts = 5
        attempts = 0

        while run.status not in ["completed", "failed", "cancelled"] and attempts < max_attempts:
            time.sleep(wait_time)
            run = client.beta.threads.runs.retrieve(thread_id=self.thread_id, run_id=run.id)
            logging.debug(f"Run status: {run.status}")
            wait_time *= 2
            attempts += 1

        if run.status == "failed":
            logging.error("Assistant run failed.")
            return None

        # Fetch messages and return the latest assistant message
        messages = client.beta.threads.messages.list(thread_id=self.thread_id).data
        messages_sorted = sorted(messages, key=lambda x: x.created_at, reverse=True)
        for msg in messages_sorted:
            if msg.role == "assistant":
                return self.extract_text_from_msg(msg)

        return None

    @staticmethod
    def extract_text_from_msg(msg):
        """
        Extract plain text from a message's content.
        """
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
    # Initialize components
    tts = TextToSpeech()
    recognizer = SpeechRecognizer()
    conv_manager = ConversationManager(assistant_id=ASSISTANT_ID)

    tts.speak("Hei, kuinka voin auttaa?")

    while True:
        # Optional: Listen for wake word first
        logging.info("Say the wake word or press ENTER to type a query. (CTRL+C to exit)")
        audio_data = recognizer.record_until_silence()

        # If nothing was recorded (i.e., no speech or mic not working), fallback:
        if not audio_data:
            user_input = input("Could not detect speech. Type your query: ").strip()
            if not user_input:
                continue
        else:
            # Transcribe first
            user_input = recognizer.transcribe(audio_data)
            logging.info(f"Transcribed speech: {user_input}")

        # If user said the wake word, keep going and record the *actual* command
        if WAKE_WORD.lower() in user_input.lower():
            tts.speak("Kyllä, miten voin auttaa?")
            # Now capture the actual command
            audio_data = recognizer.record_until_silence()
            if not audio_data:
                # fallback to text input
                user_command = input("Sano tai kirjoita pyyntösi: ").strip()
            else:
                user_command = recognizer.transcribe(audio_data)
        else:
            # Possibly the user directly said a command without the wake word
            # or typed a command
            user_command = user_input

        user_command = user_command.strip()
        if not user_command:
            continue

        logging.info(f"Käyttäjä sanoi: {user_command}")

        # Check for exit commands
        exit_commands = ["lopeta","hyvästi", "pois", "exit", "sulje"]
        if any(cmd in user_command.lower() for cmd in exit_commands):
            tts.speak("Suljetaan nyt, näkemiin!")
            sys.exit(0)

        # Send message to assistant
        conv_manager.add_user_message(user_command)
        response = conv_manager.run_assistant()
        if not response:
            tts.speak("Valitettavasti en saanut vastausta.")
            continue

        logging.info(f"Avustajan vastaus: {response}")
        tts.speak(response)
        time.sleep(1.5)  # short delay so TTS finishes before capturing


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOhjelma keskeytetty käyttäjän toimesta.")
        sys.exit(0)
