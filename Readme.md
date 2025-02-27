# Finnish Voice Assistant

## Overview

**FinnishVoiceAssistant** is a fully functional voice assistant designed for Finnish speakers. It leverages the power of OpenAI's Assistants API and Microsoft's Speech SDK for high-quality speech synthesis and recognition. The assistant supports wake-word detection, real-time voice recognition, and a conversation history cache for improved user experience.

## Features

- **Speech recognition** with Voice Activity Detection (VAD) using `webrtcvad`
- **Microsoft Speech SDK** for high-quality Finnish Text-to-Speech (TTS)
- **OpenAI Assistants API** for intelligent conversations
- **Wake word detection** (e.g., "hei avustaja")
- **Speech caching** for improved performance
- **Error handling** and graceful fallback to beep sounds if TTS fails

## Installation

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- `pip`

### Install Dependencies

Clone the repository and install dependencies:

```sh
# Clone repository
git clone https://github.com/Javedom/FinnishVoiceAssistant.git
cd FinnishVoiceAssistant

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory and set the following credentials:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_ASSISTANT_ID=your_openai_assistant_id
SPEECH_KEY=your_microsoft_speech_key
SPEECH_REGION=westeurope
```

## Usage

Run the assistant with:

```sh
python AssistantFinnish.py
```

### Interaction
- Say **"hei avustaja"** to activate the assistant.
- Speak naturally in Finnish, and the assistant will respond.
- Type a message instead if preferred.
- Say **"lopeta"** or press `Ctrl+C` to exit.

## Technical Details

### Speech Recognition
- Uses `webrtcvad` for **Voice Activity Detection (VAD)** to recognize speech.
- Supports **Microsoft Speech SDK** as an alternative speech recognizer.
- Integrates OpenAI **Whisper** for high-accuracy transcription.

### Text-to-Speech (TTS)
- Utilizes **Microsoft Azure Cognitive Services** for high-quality Finnish speech synthesis.
- Supports **SSML markup** for natural-sounding speech.
- Implements **caching** to avoid redundant API calls.

### Conversational AI
- OpenAI's **Assistants API** manages responses.
- Stores conversation history in `conversation_data.json`.
- Ensures **error handling** and fallback mechanisms.

## Troubleshooting

### Audio Issues
- Ensure your microphone is working.
- Adjust **microphone sensitivity** in system settings.
- Run the script with **administrator privileges** if permission errors occur.

### Speech SDK Issues
- Verify your **Azure Speech API key and region**.
- Install the required SDK:

```sh
pip install azure-cognitiveservices-speech
```

### OpenAI API Issues
- Ensure your **API key** is correct.
- Check **OpenAI API rate limits**.

   ```


