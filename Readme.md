# Finnish Voice Assistant (Suomalainen Ääniavustaja)

A specialized voice assistant optimized for Finnish language with advanced speech recognition, natural language processing, and system control capabilities.

## Features

- **Advanced Finnish Language Processing**
  - Morphological analysis with spaCy (if available)
  - Fallback rule-based Finnish lemmatizer
  - Entity extraction and intent classification optimized for Finnish

- **High-Quality Speech Recognition**
  - Finnish-optimized Azure Speech SDK for accurate recognition
  - Graceful fallback to simplified recognition when needed
  - Keyboard input fallback for reliability

- **Natural Language Understanding**
  - Finnish-specific intent classification 
  - Flexible command recognition with multiple phrasings
  - Wake word detection optimized for Finnish ("hei avustaja")

- **Finnish Service Integrations**
  - Weather (simulated Finnish Meteorological Institute)
  - News in Finnish
  - Finnish holidays
  - Public transport information

- **System Control Capabilities**
  - Open programs and applications
  - Open websites and search the web
  - Volume and brightness control
  - Power management (shutdown, restart, sleep)
  - Text file creation and editing

- **Enhanced User Experience**
  - High-quality Finnish text-to-speech with SSML optimization
  - Speech caching for better performance
  - Proper Finnish pronunciation of loanwords and numbers
  - Error handling and graceful degradation

## Requirements

- Python 3.7 or higher
- Windows OS (for full functionality with winsound)
- Azure Speech Services subscription (for speech recognition and TTS)
- OpenAI API key (for the assistant integration)

## Installation

1. Clone this repository or download the source code

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Install spaCy with Finnish language model for enhanced language processing:
   ```
   pip install spacy
   python -m spacy download fi_core_news_sm
   ```

4. Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   OPENAI_ASSISTANT_ID=your_openai_assistant_id
   SPEECH_KEY=your_azure_speech_key
   SPEECH_REGION=your_azure_speech_region
   ```

5. Run the assistant:
   ```
   python FinnishVoiceAssistant.py
   ```

## Configuration

The application uses environment variables for configuration. You need to set these in a `.env` file:

- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENAI_ASSISTANT_ID`: ID of your OpenAI Assistant configured for Finnish language
- `SPEECH_KEY`: Your Azure Cognitive Services Speech API key
- `SPEECH_REGION`: Azure region for your Speech resource (e.g., "westeurope")

## Usage

1. **Starting the Assistant**:
   - Run the script: `python FinnishVoiceAssistant.py`
   - The assistant will initialize and wait for the wake word

2. **Wake Word Activation**:
   - Say "hei avustaja" to activate the assistant
   - Or press Enter to type your commands

3. **Basic Commands**:
   - Open programs: "avaa ohjelma notepad"
   - Open websites: "avaa sivu yle.fi"
   - Search the web: "hae googlesta sää Helsinki"
   - Get help: "näytä komennot"

4. **Finnish Services**:
   - Weather: "mikä on sää Helsingissä"
   - News: "kerro uutiset" or "uutisia aiheesta urheilu"
   - Holidays: "milloin on seuraava pyhäpäivä"
   - Transport: "miten pääsen Helsingistä Tampereelle"

5. **System Control**:
   - Volume: "lisää äänenvoimakkuutta" or "äänenvoimakkuus 50 prosenttiin"
   - Brightness: "laske kirkkautta" or "näytön kirkkaus 70"
   - Power: "sammuta tietokone" or "mene lepotilaan"

6. **File Operations**:
   - Create files: "luo tiedosto nimeltä muistiinpanot.txt"
   - Edit files: "avaa tiedosto muistiinpanot.txt"
   - List files: "listaa tiedostot"

7. **Exiting**:
   - Say "lopeta" or type "exit" to quit the application

## Speech Recognition Modes

The application uses a tiered approach to speech recognition:

1. **Azure Speech SDK with Finnish Optimization** (default)
   - Highest quality Finnish speech recognition
   - Uses continuous recognition mode

2. **Simplified Speech Recognition** (fallback)
   - Uses English recognition model but attempts to recognize Finnish commands
   - The assistant will inform you if this mode is active

3. **Keyboard-Only Mode** (last resort)
   - If speech recognition is unavailable
   - Press Enter to type your commands

## Troubleshooting

- **No speech recognition**: Check your microphone and Azure Speech API credentials
- **Recognition quality issues**: Try speaking more clearly or typing commands
- **"spaCy Finnish language model not found"**: Run `python -m spacy download fi_core_news_sm`
- **Sound not working**: Verify your speakers and Azure credentials
- **Error on startup**: Check your `.env` file has all required variables

## Credits

This Finnish Voice Assistant combines several Azure and OpenAI technologies along with Finnish language optimizations to create a natural, responsive assistant for Finnish speakers.

## License

MIT License
