"""
FinnishVoiceAssistantTasks.py - Enhanced Voice Assistant with Intent Classification

Features:
- Speech recognition with Voice Activity Detection (VAD)
- Microsoft Speech SDK for high-quality Finnish TTS
- OpenAI Assistants API for conversation management
- Wake word detection
- Speech caching for better performance
- Enhanced error handling and graceful degradation
- System command capabilities (launch programs, open websites, search web)
- Advanced intent classification for flexible command phrasing
- NEW: Volume and brightness control
- NEW: Power management (shutdown, restart, sleep)
- NEW: Text file creation and editing capabilities
"""

import os
import sys
import time
import json
import wave
import logging
import tempfile
import hashlib
import subprocess
import webbrowser
import re
from difflib import SequenceMatcher
from typing import Optional, Dict, List, Callable, Any, Tuple
from urllib.parse import quote_plus

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
# Intent Classification
# ----------------------------------------------------------------
class IntentClassifier:
    """
    Advanced intent classifier with sample-based matching.
    Identifies user intent from various phrasings.
    """
    
    def __init__(self):
        # Define intent examples with different ways users might phrase each command
        self.intent_examples = {
            "open_program": [
                "avaa ohjelma {program}",
                "käynnistä ohjelma {program}",
                "avaa sovellus {program}",
                "käynnistä sovellus {program}",
                "avaa {program}",
                "käynnistä {program}",
                "voisitko avata {program}",
                "haluan käyttää {program}",
                "pääsenkö käyttämään {program} ohjelmaa",
                "voisitko avata minulle {program} sovelluksen",
                "tarvitsen {program}",
                "voitko avata {program}",
                "suorita {program}",
                "käynnistä {program} ohjelma",
                "haluan avata sovelluksen {program}"
            ],
            
            "open_website": [
                "avaa sivu {url}",
                "avaa verkkosivu {url}",
                "avaa sivusto {url}",
                "mene sivulle {url}",
                "avaa selain sivulle {url}",
                "näytä sivu {url}",
                "näytä verkkosivu {url}",
                "haluan käydä sivulla {url}",
                "vieraile sivustolla {url}",
                "haluaisin nähdä sivun {url}",
                "voisitko näyttää verkkosivun {url}",
                "mene osoitteeseen {url}",
                "pääsenkö sivulle {url}",
                "näytä minulle sivu {url}",
                "avaa verkkoselain ja mene sivulle {url}"
            ],
            
            "search_google": [
                "hae googlesta {query}",
                "etsi googlesta {query}",
                "google {query}",
                "hae netistä {query}",
                "etsi netistä {query}",
                "googleta {query}",
                "tee google-haku {query}",
                "hae tietoa aiheesta {query}",
                "etsi hakukoneella {query}",
                "voisitko hakea tietoa {query}",
                "haluan tietää {query}",
                "etsi tietoa {query}",
                "hae googlella {query}",
                "hakusana {query}",
                "hakusana {query}",
                "googlaa {query}"
            ],
            
            "show_help": [
                "näytä komennot",
                "näytä ohjeet",
                "mitä komentoja on",
                "apua",
                "ohje",
                "miten käytän assistenttia",
                "mitä osaat tehdä",
                "mitä komentoja voin käyttää",
                "millaisia komentoja on",
                "kerro mitä komentoja on",
                "näytä mitä osaat",
                "tarvitsen apua",
                "en tiedä mitä tehdä",
                "miten tämä toimii",
                "neuvot"
            ],
            
            # NEW: Volume control intents
            "control_volume": [
                "lisää äänenvoimakkuutta",
                "nosta ääntä",
                "kovempaa ääntä",
                "äänenvoimakkuus ylös",
                "laske äänenvoimakkuutta",
                "pienennä ääntä",
                "hiljaisempaa ääntä",
                "äänenvoimakkuus alas",
                "mykistä äänet",
                "mykistä kaiuttimet",
                "poista mykistys",
                "mute off",
                "palauta äänet",
                "aseta äänenvoimakkuus {level} prosenttiin",
                "säädä äänenvoimakkuus tasolle {level}",
                "äänenvoimakkuus {level}",
                "säädä ääni {level} prosenttiin",
                "kuinka kovalla äänet ovat",
                "paljonko äänenvoimakkuus on"
            ],
            
            # NEW: Brightness control intents
            "control_brightness": [
                "lisää kirkkautta",
                "nosta kirkkautta",
                "näyttö kirkkaammaksi",
                "kirkkaus ylös",
                "laske kirkkautta",
                "vähennä kirkkautta",
                "näyttö himmeämmäksi",
                "kirkkaus alas",
                "aseta kirkkaus {level} prosenttiin",
                "säädä kirkkaus tasolle {level}",
                "näytön kirkkaus {level}",
                "säädä näytön kirkkaus {level} prosenttiin",
                "kuinka kirkas näyttö on",
                "paljonko kirkkaus on"
            ],
            
            # NEW: Power management intents
            "power_management": [
                "sammuta tietokone",
                "sulje tietokone",
                "sammuta kone",
                "sulje järjestelmä",
                "sammuta järjestelmä",
                "käynnistä uudelleen",
                "uudelleenkäynnistä tietokone",
                "uudelleenkäynnistä kone",
                "käynnistä tietokone uudelleen",
                "reboot",
                "peruuta sammutus",
                "peruuta uudelleenkäynnistys",
                "keskeytä sammutus",
                "mene lepotilaan",
                "laita lepotilaan",
                "laita kone nukkumaan",
                "horrostila",
                "laita horrostilaan",
                "kirjaudu ulos",
                "kirjaa minut ulos",
                "logout"
            ],
            
            # NEW: File creation intents
            "create_text_file": [
                "luo tiedosto nimeltä {filename}",
                "luo tekstitiedosto {filename}",
                "tee uusi tiedosto {filename}",
                "kirjoita tiedosto {filename}",
                "tallenna tiedosto nimellä {filename}",
                "tee muistiinpano nimellä {filename}",
                "luo muistiinpano {filename}",
                "tee tekstitiedosto {filename}",
                "tallenna teksti tiedostoon {filename}",
                "luo uusi tekstitiedosto {filename}",
                "kirjoita uusi tiedosto {filename}",
                "tallenna dokumentti nimellä {filename}"
            ],
            
            # NEW: File editing intents
            "edit_text_file": [
                "avaa tiedosto {filename}",
                "lue tiedosto {filename}",
                "näytä tiedosto {filename}",
                "näytä tiedoston {filename} sisältö",
                "lisää tekstiä tiedostoon {filename}",
                "lisää tiedostoon {filename}",
                "kirjoita tiedostoon {filename}",
                "muokkaa tiedostoa {filename}",
                "päivitä tiedostoa {filename}",
                "poista rivi tiedostosta {filename}",
                "poista tiedoston {filename} rivi numero {line_number}",
                "korvaa tiedoston {filename} sisältö",
                "tyhjennä tiedosto {filename} ja kirjoita uudelleen"
            ],
            
            # NEW: List files intent
            "list_text_files": [
                "listaa tiedostot",
                "näytä tekstitiedostot",
                "mitä tiedostoja minulla on",
                "listaa tekstitiedostot",
                "näytä tiedostolista",
                "mitä tiedostoja on tallennettuna",
                "listaa dokumentit",
                "näytä muistiinpanot",
                "näytä tallentamani tiedostot",
                "luettele tekstitiedostot"
            ]
        }
        
        # Compile regular expressions for faster matching
        self.intent_patterns = {}
        for intent, examples in self.intent_examples.items():
            self.intent_patterns[intent] = [
                self._example_to_regex(example) for example in examples
            ]
    
    def _example_to_regex(self, example: str) -> re.Pattern:
        """Convert an example with placeholders to a regex pattern."""
        # Replace placeholders with regex capture groups
        pattern = example.replace("{", "(?P<").replace("}", ">.+)")
        
        # Make the pattern case-insensitive and match the whole string
        return re.compile(f"^{pattern}$", re.IGNORECASE)
    
    def _calculate_similarity(self, a: str, b: str) -> float:
        """Calculate string similarity between 0.0 and 1.0."""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()
    
    def _find_best_example(self, text: str, intent: str) -> Tuple[float, Dict]:
        """
        Find the best matching example for an intent.
        Returns the similarity score and extracted parameters.
        """
        max_score = 0.0
        best_match = {}
        
        # Check each example for this intent
        for example, pattern in zip(self.intent_examples[intent], self.intent_patterns[intent]):
            # First check if the text matches the pattern exactly
            match = pattern.match(text)
            if match:
                # Found an exact pattern match
                params = match.groupdict()
                # Give a high score for exact pattern matches
                return 0.95, params
            
            # If no exact match, try to calculate similarity
            similarity = self._calculate_similarity(
                # Remove placeholders for similarity calculation
                example.replace("{program}", "").replace("{url}", "").replace("{query}", "")
                       .replace("{level}", "").replace("{filename}", "").replace("{line_number}", ""),
                text
            )
            
            if similarity > max_score:
                max_score = similarity
                # Try to extract parameters based on context
                params = self._extract_params_from_text(text, intent)
                best_match = params
        
        return max_score, best_match
    
    def _extract_params_from_text(self, text: str, intent: str) -> Dict[str, str]:
        """Extract parameters from text based on intent type."""
        params = {}
        
        # Extract different parameters based on intent
        if intent == "open_program":
            # Try to identify program name
            for keyword in ["ohjelma", "sovellus", "käynnistä", "avaa", "suorita"]:
                if keyword in text.lower():
                    parts = text.lower().split(keyword, 1)
                    if len(parts) > 1 and parts[1].strip():
                        params["program"] = parts[1].strip()
                        break
        
        elif intent == "open_website":
            # Try to identify URL
            for keyword in ["sivu", "sivusto", "verkkosivu", "osoite", "sivulle", "selain"]:
                if keyword in text.lower():
                    parts = text.lower().split(keyword, 1)
                    if len(parts) > 1 and parts[1].strip():
                        params["url"] = parts[1].strip()
                        break
        
        elif intent == "search_google":
            # Try to identify search query
            for keyword in ["hae", "etsi", "google", "googleta", "tietoa", "hakusana"]:
                if keyword in text.lower():
                    parts = text.lower().split(keyword, 1)
                    if len(parts) > 1 and parts[1].strip():
                        params["query"] = parts[1].strip()
                        break
                        
        # NEW: Extract parameters for volume control
        elif intent == "control_volume":
            # Extract level if present
            if "prosenttiin" in text or "tasolle" in text or "%" in text:
                # Look for numbers in the text
                import re
                numbers = re.findall(r'\d+', text)
                if numbers:
                    params["level"] = int(numbers[0])
            
            # Determine action
            if any(keyword in text.lower() for keyword in ["lisää", "nosta", "kovempaa", "ylös"]):
                params["action"] = "up"
            elif any(keyword in text.lower() for keyword in ["laske", "pienennä", "hiljaisempaa", "alas"]):
                params["action"] = "down"
            elif any(keyword in text.lower() for keyword in ["mykistä"]):
                params["action"] = "mute"
            elif any(keyword in text.lower() for keyword in ["poista mykistys", "palauta äänet"]):
                params["action"] = "unmute"
            elif "level" in params:
                params["action"] = "set"
            else:
                # Default to getting current volume
                params["action"] = "get"
                
        # NEW: Extract parameters for brightness control
        elif intent == "control_brightness":
            # Extract level if present
            if "prosenttiin" in text or "tasolle" in text or "%" in text:
                # Look for numbers in the text
                import re
                numbers = re.findall(r'\d+', text)
                if numbers:
                    params["level"] = int(numbers[0])
            
            # Determine action
            if any(keyword in text.lower() for keyword in ["lisää", "nosta", "kirkkaammaksi", "ylös"]):
                params["action"] = "up"
            elif any(keyword in text.lower() for keyword in ["laske", "vähennä", "himmeämmäksi", "alas"]):
                params["action"] = "down"
            elif "level" in params:
                params["action"] = "set"
            else:
                # Default to getting current brightness
                params["action"] = "get"
                
        # NEW: Extract parameters for power management
        elif intent == "power_management":
            # Determine action
            if any(keyword in text.lower() for keyword in ["sammuta", "sulje"]):
                params["action"] = "shutdown"
            elif any(keyword in text.lower() for keyword in ["käynnistä uudelleen", "uudelleenkäynnistä", "reboot"]):
                params["action"] = "restart"
            elif any(keyword in text.lower() for keyword in ["peruuta", "keskeytä"]):
                params["action"] = "cancel"
            elif any(keyword in text.lower() for keyword in ["lepotila", "nukkumaan"]):
                params["action"] = "sleep"
            elif any(keyword in text.lower() for keyword in ["horrostila"]):
                params["action"] = "hibernate"
            elif any(keyword in text.lower() for keyword in ["kirjaudu ulos", "logout"]):
                params["action"] = "logout"
                
        # NEW: Extract parameters for file creation
        elif intent == "create_text_file":
            # Extract filename
            for keyword in ["nimeltä", "tiedosto", "tekstitiedosto", "muistiinpano", "nimellä"]:
                if keyword in text.lower():
                    parts = text.lower().split(keyword, 1)
                    if len(parts) > 1 and parts[1].strip():
                        params["filename"] = parts[1].strip()
                        break
                        
        # NEW: Extract parameters for file editing
        elif intent == "edit_text_file":
            # Extract filename
            for keyword in ["tiedosto", "tiedostoon", "tiedostoa", "tiedoston"]:
                if keyword in text.lower():
                    parts = text.lower().split(keyword, 1)
                    if len(parts) > 1 and parts[1].strip():
                        # Get the first word after the keyword
                        words = parts[1].strip().split()
                        if words:
                            params["filename"] = words[0]
                            break
            
            # Determine operation
            if any(keyword in text.lower() for keyword in ["avaa", "lue", "näytä", "sisältö"]):
                params["operation"] = "read"
            elif any(keyword in text.lower() for keyword in ["lisää"]):
                params["operation"] = "append"
            elif any(keyword in text.lower() for keyword in ["kirjoita", "korvaa", "tyhjennä"]):
                params["operation"] = "write"
            elif any(keyword in text.lower() for keyword in ["poista rivi"]):
                params["operation"] = "delete_line"
                # Look for line number
                import re
                numbers = re.findall(r'\d+', text)
                if numbers and len(numbers) > 0:
                    params["line_number"] = int(numbers[-1])  # Take the last number found
        
        return params
    
    def classify_intent(self, text: str) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Classify the user's intent and extract parameters.
        
        Args:
            text: User's input text
            
        Returns:
            Tuple of (intent_name, parameters) or (None, {}) if no intent was identified
        """
        best_intent = None
        best_score = 0.5  # Minimum threshold to consider it a match
        best_params = {}
        
        # Check each intent
        for intent in self.intent_examples:
            score, params = self._find_best_example(text, intent)
            logger.debug(f"Intent {intent} score: {score}, params: {params}")
            
            if score > best_score:
                best_score = score
                best_intent = intent
                best_params = params
        
        if best_intent == "show_help":
            # Special case - no parameters for help
            return best_intent, {}
            
        # Return the best matching intent and its parameters
        if best_intent and best_params:
            return best_intent, best_params
        
        # No clear intent identified
        return None, {}

# ----------------------------------------------------------------
# Enhanced Command Registry with Intent Classification
# ----------------------------------------------------------------
class EnhancedCommandRegistry:
    """Enhanced command registry with intent classification."""
    
    def __init__(self):
        self.commands = {}
        self.intent_classifier = IntentClassifier()
        
    def register(self, command_name: str, function: callable, description: str):
        """Register a command with its handling function."""
        self.commands[command_name] = {
            "function": function,
            "description": description
        }
        logger.info(f"Registered command: {command_name}")
    
    def find_command(self, text: str) -> Tuple[Optional[callable], Dict[str, Any]]:
        """
        Find the best matching command based on intent classification.
        
        Args:
            text: User input text
            
        Returns:
            Tuple of (function, params) or (None, {}) if no command was matched
        """
        # Use intent classifier to determine the most likely intent
        intent, params = self.intent_classifier.classify_intent(text)
        
        if intent == "show_help":
            # Special case for help
            return self.help, {}
            
        if intent and intent in self.commands:
            return self.commands[intent]["function"], params
            
        return None, {}
    
    def help(self) -> str:
        """Generate help text for available commands."""
        help_text = "Käytettävissä olevat komennot:\n\n"
        
        # Show each command with examples
        for name, command in self.commands.items():
            help_text += f"• {command['description']}\n"
            # Include some example phrasings
            examples = self.intent_classifier.intent_examples.get(name, [])[:3]
            for example in examples:
                # Replace placeholders with examples
                if "{program}" in example:
                    example = example.replace("{program}", "notepad")
                elif "{url}" in example:
                    example = example.replace("{url}", "google.com")
                elif "{query}" in example:
                    example = example.replace("{query}", "sää Helsinki")
                elif "{level}" in example:
                    example = example.replace("{level}", "50")
                elif "{filename}" in example:
                    example = example.replace("{filename}", "muistiinpanot")
                elif "{line_number}" in example:
                    example = example.replace("{line_number}", "3")
                help_text += f"  Esimerkki: \"{example}\"\n"
            help_text += "\n"
            
        return help_text

# ----------------------------------------------------------------
# System Command Functions
# ----------------------------------------------------------------
def open_program(program: str) -> str:
    """
    Open a program based on its name.
    
    Args:
        program_name: Name of the program to open
        
    Returns:
        Response message
    """
    # Common programs with their executable paths
    program_map = {
        "notepad": "notepad.exe",
        "muistio": "notepad.exe",
        "word": "WINWORD.EXE",
        "excel": "EXCEL.EXE",
        "powerpoint": "POWERPNT.EXE",
        "chrome": r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        "firefox": r"C:\Program Files\Mozilla Firefox\firefox.exe",
        "edge": r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        "explorer": "explorer.exe",
        "tiedostonhallinta": "explorer.exe",
        "calc": "calc.exe",
        "laskin": "calc.exe",
        "paint": "mspaint.exe",
        "cmd": "cmd.exe",
        "komentokehote": "cmd.exe",
        "spotify": r"C:\Users\%USERNAME%\AppData\Roaming\Spotify\Spotify.exe"
    }
    
    try:
        # Strip common words that might be in the command
        program = program.strip()
        for prefix in ["nimeltä", "ohjelma", "sovellus"]:
            if program.startswith(prefix):
                program = program[len(prefix):].strip()
                
        # Try to find the program in our map
        program_to_launch = None
        
        # Direct match
        if program.lower() in program_map:
            program_to_launch = program_map[program.lower()]
        else:
            # Try to find a partial match
            for prog_key, prog_path in program_map.items():
                if prog_key in program.lower():
                    program_to_launch = prog_path
                    break
        
        # Launch the program
        if program_to_launch:
            # Replace %USERNAME% with actual username if needed
            if "%USERNAME%" in program_to_launch:
                program_to_launch = program_to_launch.replace("%USERNAME%", os.getenv("USERNAME"))
                
            # Launch the program
            subprocess.Popen(program_to_launch)
            return f"Avaan ohjelman: {program}"
        else:
            # Try to launch directly if not in our map
            subprocess.Popen(program)
            return f"Yritän avata ohjelman: {program}"
    
    except Exception as e:
        logger.error(f"Error opening program '{program}': {e}")
        return f"En voinut avata ohjelmaa {program}. Virhe: {str(e)}"

def open_website(url: str) -> str:
    """
    Open a website in the default browser.
    
    Args:
        url: URL to open
        
    Returns:
        Response message
    """
    try:
        # Clean up the URL
        url = url.strip()
        
        # Remove common Finnish prefixes
        prefixes = ["osoite", "sivu", "sivusto", "verkkosivu", "verkko-osoite", "www", "http"]
        for prefix in prefixes:
            if url.startswith(prefix):
                url = url[len(prefix):].strip()
        
        # Remove any leading dots, slashes or colons
        url = url.lstrip('./:')
        
        # If the URL doesn't start with http:// or https://, add it
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
            
        # Open the URL in the default browser
        webbrowser.open(url)
        return f"Avaan verkkosivun: {url}"
    
    except Exception as e:
        logger.error(f"Error opening website '{url}': {e}")
        return f"En voinut avata verkkosivua {url}. Virhe: {str(e)}"

def search_google(query: str) -> str:
    """
    Search Google for the given query.
    
    Args:
        query: Search query
        
    Returns:
        Response message
    """
    try:
        # Clean up the query
        query = query.strip()
        
        # Create the search URL
        search_url = f"https://www.google.com/search?q={quote_plus(query)}"
        
        # Open the URL in the default browser
        webbrowser.open(search_url)
        return f"Haen Googlesta: {query}"
    
    except Exception as e:
        logger.error(f"Error searching Google '{query}': {e}")
        return f"En voinut hakea Googlesta {query}. Virhe: {str(e)}"

# ----------------------------------------------------------------
# NEW: Advanced System Control Functions
# ----------------------------------------------------------------
def control_volume(action: str, level: Optional[int] = None) -> str:
    """
    Control system volume.
    
    Args:
        action: "up", "down", "mute", "unmute", or "set"
        level: Volume level (0-100) when action is "set"
        
    Returns:
        Response message
    """
    try:
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        from comtypes import CLSCTX_ALL
        from ctypes import cast, POINTER
        
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(
            IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
        
        # Current volume as percentage
        current_vol = round(volume.GetMasterVolumeLevelScalar() * 100)
        
        if action == "up":
            # Increase by 10%
            new_vol = min(current_vol + 10, 100) / 100
            volume.SetMasterVolumeLevelScalar(new_vol, None)
            return f"Äänenvoimakkuus nostettu tasolle {round(new_vol * 100)}%"
        
        elif action == "down":
            # Decrease by 10%
            new_vol = max(current_vol - 10, 0) / 100
            volume.SetMasterVolumeLevelScalar(new_vol, None)
            return f"Äänenvoimakkuus laskettu tasolle {round(new_vol * 100)}%"
        
        elif action == "mute":
            volume.SetMute(1, None)
            return "Äänet mykistetty"
        
        elif action == "unmute":
            volume.SetMute(0, None)
            return "Mykistys poistettu"
        
        elif action == "set" and level is not None:
            # Set to specific level
            level = max(0, min(level, 100))
            volume.SetMasterVolumeLevelScalar(level / 100, None)
            return f"Äänenvoimakkuus asetettu tasolle {level}%"
        
        return f"Nykyinen äänenvoimakkuus on {current_vol}%"
    
    except ImportError:
        logger.error("Volume control requires pycaw. Install with: pip install pycaw")
        return "Äänenvoimakkuuden säätö vaatii pycaw-kirjaston. Asenna se komennolla 'pip install pycaw'."
    except Exception as e:
        logger.error(f"Error controlling volume: {e}")
        return f"Äänenvoimakkuuden säätö epäonnistui. Virhe: {str(e)}"

def control_brightness(action: str, level: Optional[int] = None) -> str:
    """
    Control screen brightness.
    
    Args:
        action: "up", "down", or "set"
        level: Brightness level (0-100) when action is "set"
        
    Returns:
        Response message
    """
    try:
        import wmi
        
        # Connect to WMI namespace
        wmi_service = wmi.WMI(namespace="root\\WMI")
        
        # Get the first brightness controller
        controllers = wmi_service.WmiMonitorBrightnessMethods()
        if not controllers:
            return "Näytön kirkkauden säätöä ei tueta tällä laitteella."
        
        controller = controllers[0]
        
        # Get current brightness
        brightness_info = wmi_service.WmiMonitorBrightness()[0]
        current_brightness = brightness_info.CurrentBrightness
        
        if action == "up":
            # Increase by 10%
            new_brightness = min(current_brightness + 10, 100)
            controller.WmiSetBrightness(new_brightness, 0)
            return f"Näytön kirkkaus nostettu tasolle {new_brightness}%"
        
        elif action == "down":
            # Decrease by 10%
            new_brightness = max(current_brightness - 10, 0)
            controller.WmiSetBrightness(new_brightness, 0)
            return f"Näytön kirkkaus laskettu tasolle {new_brightness}%"
        
        elif action == "set" and level is not None:
            # Set to specific level
            level = max(0, min(level, 100))
            controller.WmiSetBrightness(level, 0)
            return f"Näytön kirkkaus asetettu tasolle {level}%"
        
        return f"Nykyinen näytön kirkkaus on {current_brightness}%"
    
    except ImportError:
        logger.error("Brightness control requires wmi. Install with: pip install wmi")
        return "Näytön kirkkauden säätö vaatii wmi-kirjaston. Asenna se komennolla 'pip install wmi'."
    except Exception as e:
        logger.error(f"Error controlling brightness: {e}")
        return f"Näytön kirkkauden säätö epäonnistui. Virhe: {str(e)}"

def power_management(action: str) -> str:
    """
    Manage system power (shutdown, restart, sleep, etc.).
    
    Args:
        action: "shutdown", "restart", "cancel", "sleep", "hibernate", "logout"
        
    Returns:
        Response message
    """
    try:
        if action == "shutdown":
            # Prompt user to confirm
            subprocess.run(["shutdown", "/s", "/t", "60", "/c", "Tietokone sammutetaan 60 sekunnin kuluttua. Paina Peruuta-painiketta peruuttaaksesi."])
            return "Tietokone sammutetaan 60 sekunnin kuluttua. Voit peruuttaa sen Windowsin ilmoituksesta."
        
        elif action == "restart":
            subprocess.run(["shutdown", "/r", "/t", "60", "/c", "Tietokone käynnistetään uudelleen 60 sekunnin kuluttua. Paina Peruuta-painiketta peruuttaaksesi."])
            return "Tietokone käynnistetään uudelleen 60 sekunnin kuluttua. Voit peruuttaa sen Windowsin ilmoituksesta."
        
        elif action == "cancel":
            # Cancel shutdown or restart
            subprocess.run(["shutdown", "/a"])
            return "Sammutus tai uudelleenkäynnistys peruutettu."
        
        elif action == "sleep":
            subprocess.run(["powercfg", "-setactive", "scheme_max"])
            subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
            return "Tietokone laitetaan lepotilaan."
        
        elif action == "hibernate":
            subprocess.run(["rundll32.exe", "powrprof.dll,SetSuspendState", "Hibernate"])
            return "Tietokone laitetaan horrostilaan."
        
        elif action == "logout":
            subprocess.run(["shutdown", "/l"])
            return "Kirjaudutaan ulos."
        
        return "Tuntematon virransäästökomento. Vaihtoehdot: shutdown, restart, cancel, sleep, hibernate, logout."
    
    except Exception as e:
        logger.error(f"Error in power management: {e}")
        return f"Virransäästötoiminto epäonnistui. Virhe: {str(e)}"

# ----------------------------------------------------------------
# NEW: File Creation & Editing Functions
# ----------------------------------------------------------------
def create_text_file(filename: str, content: Optional[str] = None) -> str:
    """
    Create a new text file with optional content.
    
    Args:
        filename: Name of the file to create
        content: Optional content to write to the file
        
    Returns:
        Response message
    """
    try:
        # Clean up filename
        if not filename.endswith(('.txt', '.md', '.csv', '.log')):
            filename += '.txt'
        
        # Use Documents folder as the base location
        documents_path = os.path.join(os.path.expanduser('~'), 'Documents')
        file_path = os.path.join(documents_path, filename)
        
        # Check if file already exists
        if os.path.exists(file_path):
            return f"Tiedosto {filename} on jo olemassa. Valitse toinen nimi tai käytä muokkauskomentoa."
        
        # Create the file
        with open(file_path, 'w', encoding='utf-8') as f:
            if content:
                f.write(content)
        
        return f"Tiedosto {filename} luotu onnistuneesti kansioon Tiedostot."
    
    except Exception as e:
        logger.error(f"Error creating text file: {e}")
        return f"Tiedoston luonti epäonnistui. Virhe: {str(e)}"

def edit_text_file(filename: str, operation: str = "read", content: Optional[str] = None, line_number: Optional[int] = None) -> str:
    """
    Edit an existing text file.
    
    Args:
        filename: Name of the file to edit
        operation: "read", "append", "write", "delete_line"
        content: Content to write when operation is "append" or "write"
        line_number: Line number for "delete_line" operation
        
    Returns:
        Response message or file content for "read" operation
    """
    try:
        # Clean up filename
        if not filename.endswith(('.txt', '.md', '.csv', '.log')):
            filename += '.txt'
        
        # Use Documents folder as the base location
        documents_path = os.path.join(os.path.expanduser('~'), 'Documents')
        file_path = os.path.join(documents_path, filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return f"Tiedostoa {filename} ei löydy. Voit luoda sen 'luo tiedosto' -komennolla."
        
        if operation == "read":
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # If content is too long, truncate it
            if len(content) > 500:
                content = content[:497] + "..."
                return f"Tiedoston {filename} sisältö (osittainen):\n\n{content}"
            
            return f"Tiedoston {filename} sisältö:\n\n{content}"
        
        elif operation == "append" and content:
            # Append to file
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(content)
            return f"Teksti lisätty tiedostoon {filename}."
        
        elif operation == "write" and content:
            # Overwrite file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"Tiedosto {filename} kirjoitettu uudelleen."
        
        elif operation == "delete_line" and line_number is not None:
            # Read all lines
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Check if line number is valid
            if line_number < 1 or line_number > len(lines):
                return f"Virheellinen rivinumero. Tiedostossa {filename} on {len(lines)} riviä."
            
            # Remove the specified line
            del lines[line_number - 1]
            
            # Write the file back
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            return f"Rivi {line_number} poistettu tiedostosta {filename}."
        
        # If we get here, something went wrong
        return "Virheellinen tiedoston muokkausoperaatio. Vaihtoehdot: read, append, write, delete_line."
    
    except Exception as e:
        logger.error(f"Error editing text file: {e}")
        return f"Tiedoston muokkaus epäonnistui. Virhe: {str(e)}"

def list_text_files() -> str:
    """
    List text files in the Documents folder.
    
    Returns:
        List of text files
    """
    try:
        # Use Documents folder as the base location
        documents_path = os.path.join(os.path.expanduser('~'), 'Documents')
        
        # List files with text extensions
        text_files = []
        for file in os.listdir(documents_path):
            if file.endswith(('.txt', '.md', '.csv', '.log')):
                text_files.append(file)
        
        if not text_files:
            return "Ei tekstitiedostoja Documents-kansiossa."
        
        # Format the list
        file_list = "\n".join(text_files)
        return f"Tekstitiedostot Documents-kansiossa:\n\n{file_list}"
    
    except Exception as e:
        logger.error(f"Error listing text files: {e}")
        return f"Tiedostojen listaus epäonnistui. Virhe: {str(e)}"

def handle_system_command(text: str) -> Optional[str]:
    """
    Parse and handle system commands with intent classification.
    
    Args:
        text: Command text to parse
        
    Returns:
        Response message if command was handled, None otherwise
    """
    # Find and execute command
    function, args = command_registry.find_command(text)
    
    if function and args:
        return function(**args)
    elif function:  # Special case for help function which doesn't need args
        return function()
    
    # No command found
    return None

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
    print("Finnish Voice Assistant with Advanced Intent Classification")
    print("======================================================")
    print("Say 'hei avustaja' to start or press Enter to type.")
    print("Say 'lopeta' to exit.")
    print()
    print("You can now phrase commands naturally, for example:")
    print("- 'Voisitko avata muistio-ohjelman?'")
    print("- 'Haluaisin käydä sivulla google.com'") 
    print("- 'Etsi tietoa säästä Helsingissä'")
    print("- 'Näytä mitä komentoja osaat'")
    print("- 'Nosta äänenvoimakkuutta'")  # New examples for system control
    print("- 'Laske näytön kirkkautta'")
    print("- 'Luo tiedosto nimeltä kauppalista'")
    print("- 'Laita tietokone lepotilaan'")
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
        
        # Initialize command registry
        global command_registry
        command_registry = EnhancedCommandRegistry()
        
        # Register basic commands
        command_registry.register(
            "open_program",
            open_program,
            "Avaa ohjelman tietokoneella"
        )
        
        command_registry.register(
            "open_website",
            open_website,
            "Avaa verkkosivun selaimessa"
        )
        
        command_registry.register(
            "search_google",
            search_google,
            "Hakee tietoa Googlesta"
        )
        
        # Register new system control commands
        command_registry.register(
            "control_volume",
            control_volume,
            "Säädä järjestelmän äänenvoimakkuutta"
        )
        
        command_registry.register(
            "control_brightness",
            control_brightness,
            "Säädä näytön kirkkautta"
        )
        
        command_registry.register(
            "power_management",
            power_management,
            "Hallitse tietokoneen virtatilaa (sammutus, uudelleenkäynnistys, lepotila)"
        )
        
        command_registry.register(
            "create_text_file",
            create_text_file,
            "Luo uusi tekstitiedosto"
        )
        
        command_registry.register(
            "edit_text_file",
            edit_text_file,
            "Lue tai muokkaa olemassa olevaa tekstitiedostoa"
        )
        
        command_registry.register(
            "list_text_files",
            list_text_files,
            "Listaa saatavilla olevat tekstitiedostot"
        )
        
        # Initial greeting
        print("Starting assistant...")
        tts.speak("Hei, kuinka voin auttaa sinua tänään? Voit pyytää minua avaamaan ohjelmia, verkkosivuja tai hakemaan tietoa. Ymmärrän myös monia erilaisia tapoja ilmaista pyyntöjä.")
        
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
                        
                        # Check if it's a system command with intent classification
                        system_response = handle_system_command(user_input)
                        if system_response:
                            print(f"System Command: {system_response}")
                            tts.speak(system_response)
                        else:
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
                    
                    # Check if it's a system command with intent classification
                    system_response = handle_system_command(command_text)
                    if system_response:
                        print(f"System Command: {system_response}")
                        tts.speak(system_response)
                    else:
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
                    
                    # Check if it's a system command with intent classification
                    system_response = handle_system_command(speech_text)
                    if system_response:
                        print(f"System Command: {system_response}")
                        tts.speak(system_response)
                    else:
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