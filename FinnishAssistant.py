from dotenv import load_dotenv
load_dotenv()

import os
import sys
import time
import logging
import json
import wave

import pyaudio
from gtts import gTTS
import playsound
from openai import OpenAI
import openai  # for backwards compatibility if needed

logging.basicConfig(level=logging.DEBUG)

API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = os.getenv("OPENAI_ASSISTANT_ID")
if not API_KEY or not ASSISTANT_ID:
    logging.error("OPENAI_API_KEY and OPENAI_ASSISTANT_ID must be set in the environment.")
    sys.exit(1)

# Instantiate the OpenAI client (new API interface)
client = OpenAI(api_key=API_KEY)

# --------------------------
# TEXT-TO-SPEECH USING gTTS
# --------------------------
def speak(text: str):
    """Speak the given text aloud using gTTS for a more natural Finnish voice."""
    try:
        tts = gTTS(text, lang="fi")
        filename = "temp_tts.mp3"
        tts.save(filename)
        playsound.playsound(filename, True)
        os.remove(filename)
    except Exception as e:
        logging.error(f"Error during TTS: {e}")

# --------------------------
# SPEECH RECOGNITION USING WHISPER API
# --------------------------
def get_speech_input(timeout=5):
    """
    Record audio from the microphone for 'timeout' seconds,
    save it as a temporary WAV file, and use OpenAI's Whisper API
    to transcribe the audio into text (in Finnish).
    """
    audio_interface = pyaudio.PyAudio()
    stream = audio_interface.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024
    )
    frames = []
    num_frames = int(16000 / 1024 * timeout)
    for _ in range(num_frames):
        data = stream.read(1024)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    sample_width = audio_interface.get_sample_size(pyaudio.paInt16)
    audio_interface.terminate()

    filename = "temp_audio.wav"
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()

    try:
        with open(filename, "rb") as audio_file:
            # Use the new API call for transcriptions
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="fi"
            )
            return result.text.strip()
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return ""

# --------------------------
# HELPER: Extract Plain Text from Message Blocks
# --------------------------
def extract_text_from_msg(msg):
    """
    Extract plain text from a message's content.
    If the content is a list of blocks, join them into a single string.
    """
    content = msg.content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            # If the block has a 'text' attribute, use it
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

# --------------------------
# CONVERSATION THREAD SETUP
# --------------------------
thread_id = None

def create_thread_with_instructions():
    """
    Create a new conversation thread with instructions in Finnish.
    """
    global thread_id
    chat_thread = client.beta.threads.create()
    thread_id = chat_thread.id
    logging.debug(f"Created new thread with id: {thread_id}")

    instructions = (
        "Olet ääniavustaja, joka ymmärtää ja vastaa suomeksi. Keskustele kanssani vain suomeksi."
    )
    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=instructions
    )

def get_assistant_response(user_message: str):
    """
    Send the user message to the assistant and return the assistant's text reply.
    """
    global thread_id
    if thread_id is None:
        create_thread_with_instructions()

    client.beta.threads.messages.create(
        thread_id=thread_id,
        role="user",
        content=user_message
    )

    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID
    )

    wait_time = 1
    max_attempts = 5
    attempts = 0
    while run.status not in ["completed", "failed", "cancelled"] and attempts < max_attempts:
        time.sleep(wait_time)
        run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        logging.debug(f"Run status: {run.status}")
        wait_time *= 2
        attempts += 1

    if run.status == "failed":
        logging.error("Assistant run failed.")
        return None

    message_response = client.beta.threads.messages.list(thread_id=thread_id)
    messages_sorted = sorted(message_response.data, key=lambda x: x.created_at, reverse=True)
    for msg in messages_sorted:
        if msg.role == "assistant":
            return extract_text_from_msg(msg)
    return None

# --------------------------
# MAIN LOOP
# --------------------------
def main():
    speak("Hei, kuinka voin auttaa?")
    while True:
        user_text = get_speech_input(timeout=5)
        if not user_text:
            continue

        logging.info(f"Käyttäjä sanoi: {user_text}")

        # Allow exit command in Finnish
        exit_commands = ["lopeta", "pois", "exit", "sulje"]
        if any(cmd in user_text.lower() for cmd in exit_commands):
            speak("Suljetaan nyt, näkemiin!")
            sys.exit(0)

        response = get_assistant_response(user_text)
        if response is None:
            speak("Valitettavasti en saanut vastausta.")
            continue

        logging.info(f"Avustajan vastaus: {response}")
        speak(response)

if __name__ == "__main__":
    main()
