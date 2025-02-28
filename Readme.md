# Finnish Voice Assistant

A versatile voice-controlled assistant for Windows that understands natural Finnish commands, with advanced intent recognition and system control capabilities. Also can just chat with user.

## Features

- **Voice Recognition with VAD** - Detects voice activity and transcribes speech using OpenAI Whisper
- **Finnish Language Support** - Optimized for Finnish language comprehension and speech
- **High-Quality Text-to-Speech** - Uses Microsoft Cognitive Services for natural Finnish speech
- **Wake Word Detection** - Trigger with "hei avustaja" to start interactions
- **OpenAI Integration** - Uses OpenAI Assistants API for conversation management
- **Advanced Intent Classification** - Understands various ways to phrase commands in Finnish
- **System Controls:**
  - Open applications and websites
  - Search the web
  - Control volume and screen brightness
  - Power management (shutdown, restart, sleep)
  - Create and edit text files
  - Chat using ChatGPT

## Requirements

- Windows operating system
- Python 3.7 or higher
- OpenAI API key
- Microsoft Speech Services API key
- Internet connection

## Installation

1. Clone or download this repository:
```
git clone https://github.com/Javedom/finnish-voice-assistant.git
cd finnish-voice-assistant
```

2. Install required dependencies:
```
pip install -r requirements.txt
```

3. Create a `.env` file in the project directory with the following content:
```
OPENAI_API_KEY='Your_openai_api_key_here'
OPENAI_ASSISTANT_ID='Your_openai_assistant_id_here'
SPEECH_KEY='Your_Azure_Speech_Key_Here'
SPEECH_REGION='Azure_service_region_here'
```

## Configuration

### OpenAI Assistant Setup

1. Visit the [OpenAI platform](https://platform.openai.com/)
2. Create a new Assistant
3. Configure it with Finnish language capability and your desired instructions
4. Copy the Assistant ID to your `.env` file

### Microsoft Speech Services

1. Sign up for [Microsoft Azure](https://azure.microsoft.com/)
2. Create a Speech Services resource
3. Get your API key and region
4. Add them to your `.env` file

## Usage

Run the assistant:

```
python finnish-voice-assistant.py
```

- Say commands out loud
- Press Enter to type commands directly
- Say "lopeta" to exit the application

## Command Examples

### Basic Controls
- "Avaa ohjelma notepad" - Open Notepad
- "Avaa sivu google.com" - Open Google website
- "Hae googlesta sää Helsinki" - Search Google for weather in Helsinki
- "Näytä komennot" - Show available commands

### System Controls
- "Lisää äänenvoimakkuutta" - Increase volume
- "Laske äänenvoimakkuutta" - Decrease volume
- "Mykistä äänet" - Mute sound
- "Aseta äänenvoimakkuus 70 prosenttiin" - Set volume to 70%

### Display Controls
- "Nosta kirkkautta" - Increase brightness
- "Laske näytön kirkkautta" - Decrease brightness
- "Aseta kirkkaus 50 prosenttiin" - Set brightness to 50%

### Power Management
- "Sammuta tietokone" - Shutdown computer (with 60-second delay)
- "Käynnistä uudelleen" - Restart computer
- "Peruuta sammutus" - Cancel shutdown or restart
- "Laita tietokone lepotilaan" - Put computer to sleep
- "Kirjaudu ulos" - Log out

### File Management
- "Luo tiedosto nimeltä kauppalista" - Create a text file named "kauppalista"
- "Lue tiedosto muistiinpanot" - Read the contents of a file
- "Lisää tekstiä tiedostoon kauppalista" - Append text to a file
- "Korvaa tiedoston sisältö" - Replace file contents
- "Poista tiedoston rivi numero 3" - Delete a specific line from a file
- "Listaa tekstitiedostot" - List all text files

## Voice Assistant Prompt Configuration

For best results, configure your OpenAI Assistant with instructions to:
1. Respond only in Finnish
2. Be conversational and helpful
3. Provide concise, informative answers

## Troubleshooting

### Audio Issues
- Ensure your microphone is connected and working
- Check Windows privacy settings to allow microphone access
- Verify there are no other applications using the microphone

### API Connection Problems
- Verify your internet connection
- Check that your API keys are correct in the `.env` file
- Ensure you have sufficient credits in your OpenAI account

### System Command Failures
- Run the program with administrator privileges for power management commands
- Some laptops may not support brightness control via WMI

## License

This project is licensed under the MIT License - see the LICENSE file for details.
