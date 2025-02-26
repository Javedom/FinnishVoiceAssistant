# Voice Assistant using OpenAI Whisper and gTTS

This is a Finnish-language voice assistant that uses OpenAI's Whisper API for speech recognition and gTTS (Google Text-to-Speech) for spoken responses. The assistant listens to user input, processes it with OpenAI's API, and responds with generated speech.

## Features
- **Speech Recognition**: Uses OpenAI's Whisper API to transcribe spoken Finnish into text.
- **Text-to-Speech**: Uses Google Text-to-Speech (gTTS) to convert text responses into spoken Finnish.
- **Conversation Memory**: Maintains a conversation thread with OpenAI's API.
- **Exit Commands**: Recognizes Finnish exit commands to terminate the program.

## Installation

### Prerequisites
- Python 3.8+
- `ffmpeg` (required for audio processing with `pydub`)

### Setup
1. Clone this repository:
   ```sh
   git clone https://github.com/javedom/voice-assistant.git
   cd voice-assistant
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Create a `.env` file and add your OpenAI API credentials:
   ```ini
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_ASSISTANT_ID=your_assistant_id
   ```
4. Ensure `ffmpeg` is installed and available in your system's PATH.

## Usage
Run the program with:
```sh
python main.py
```

The assistant will greet you and start listening for voice commands. Speak into your microphone, and it will process your speech, respond accordingly, and continue the conversation.

### Exit Commands
To stop the program, say any of the following Finnish words:
- "lopeta"
- "pois"
- "exit"
- "sulje"

## Dependencies
- `openai` - Communicates with OpenAI API
- `gtts` - Converts text to speech
- `playsound` - Plays audio files
- `pydub` - Handles audio conversion (MP3 to WAV)
- `pyaudio` - Captures microphone input
- `wave` - Handles WAV file operations
- `python-dotenv` - Loads environment variables

## License
MIT License
pydub
pyaudio
wave
```

