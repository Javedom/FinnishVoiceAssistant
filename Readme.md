# Voice Assistant with VAD & Persistent Conversations

This is a voice assistant project using **webrtcvad** for **voice activity detection (VAD)**, **OpenAI Whisper** for **speech-to-text**, and **gTTS** for **text-to-speech (TTS)**. It maintains conversation context using OpenAI's **threads API** and stores conversation history for persistence.

## Features
- **Voice Activity Detection (VAD)**: Uses `webrtcvad` to detect speech and avoid unnecessary recording.
- **Persistent Conversations**: Stores conversation history using OpenAI's **Threads API**.
- **Modular Structure**: Clean and extendable object-oriented design.
- **Enhanced TTS with Caching**: Stores generated TTS files to avoid redundant processing.
- **User Experience Improvements**:
  - Wake word detection (`"hei avustaja"`)
  - Automatic listening for speech
  - Graceful exit commands

---

## Installation
### 1. Clone the Repository
```sh
git clone https://github.com/javedom/FinnishVoiceAssistant.git
cd voice-assistant
```

### 2. Create and Activate a Virtual Environment (Recommended)
```sh
python -m venv venv  # Create virtual environment
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the project directory and add your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
OPENAI_ASSISTANT_ID=your-assistant-id-here
```

---

## Usage
Run the voice assistant:
```sh
python main.py
```

The assistant will start listening for speech. To stop the assistant, say an **exit command** (e.g., *"lopeta", "pois", "exit"*).

---

## Requirements
- **Python 3.8+**
- **Dependencies (installed via `requirements.txt`)**:
  - `python-dotenv`
  - `pyaudio`
  - `webrtcvad`
  - `openai`
  - `gTTS`
  - `playsound==1.2.2` *(for TTS playback)*
  
#### Additional Requirements for Some Systems:
- **Linux Users**: Install `portaudio` package for PyAudio support:
  ```sh
  sudo apt-get install portaudio19-dev
  ```
- **Windows Users**: If facing PyAudio issues, install it manually:
  ```sh
  pip install pipwin
  pipwin install pyaudio
  ```

---

## Troubleshooting
### 1. No sound output from TTS
- Try installing `playsound==1.2.2` instead of newer versions:
  ```sh
  pip install playsound==1.2.2
  ```

### 2. Microphone not detected or no speech input
- Check system audio settings to ensure the correct microphone is selected.
- If using Windows, disable *Stereo Mix* in audio settings.

### 3. OpenAI API errors
- Ensure your `.env` file contains the correct **API key** and **assistant ID**.
- Check your OpenAI account for rate limits or API key issues.






