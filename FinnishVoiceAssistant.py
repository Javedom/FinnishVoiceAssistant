"""
EnhancedFinnishVoiceAssistant.py - Specialized Finnish Voice Assistant

Features:
- Advanced Finnish language processing with spaCy for morphological analysis
- Finnish-optimized Microsoft Speech SDK for high-quality Finnish speech recognition and TTS
- Finnish-specific intent classification with natural language patterns
- Finnish service integrations (weather, news, holidays, transport)
- Wake word detection optimized for Finnish pronunciation
- Speech caching for better performance
- Enhanced error handling and graceful degradation
- System command capabilities (launch programs, open websites, search web)
- Advanced intent classification for flexible Finnish command phrasing
- Volume and brightness control
- Power management (shutdown, restart, sleep)
- Text file creation and editing capabilities
"""

# Add this at the top of your file - before any other imports
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'  # Ensure proper encoding

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
import threading
import queue
from datetime import datetime
import requests
import xml.etree.ElementTree as ET

# Third-party imports
import winsound
from openai import OpenAI
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk


# Finnish language processing with spaCy - attempt with full error handling
SPACY_AVAILABLE = False
try:
    import spacy
    # Try to load Finnish model
    try:
        # Try to load the Finnish model - you need to install it first with:
        # python -m spacy download fi_core_news_sm
        nlp = spacy.load("fi_core_news_sm")
        SPACY_AVAILABLE = True
        print("spaCy Finnish language model loaded successfully")
    except OSError:
        print("Finnish language model not found. Install with: python -m spacy download fi_core_news_sm")
except ImportError as e:
    print(f"spaCy library not available: {e}. Install with: pip install spacy")
except Exception as e:
    print(f"Unexpected error initializing spaCy: {e}")

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
SPEECH_REGION = os.getenv("SPEECH_REGION")

if not API_KEY or not ASSISTANT_ID:
    logger.error("OPENAI_API_KEY and OPENAI_ASSISTANT_ID must be set in the environment.")
    sys.exit(1)

if not SPEECH_KEY or not SPEECH_REGION:
    logger.error("SPEECH_KEY and SPEECH_REGION must be set in the environment for Microsoft Speech services.")
    sys.exit(1)

# Audio settings
WAKE_WORD = "avustaja"
CONVERSATION_FILE = "conversation_data.json"
SPEECH_CACHE_DIR = "speech_cache"

# OpenAI client
client = OpenAI(api_key=API_KEY)

# ----------------------------------------------------------------
# Finnish Language Processing
# ----------------------------------------------------------------
class FinnishLanguageProcessor:
    
    def __init__(self):
        # Initialize spaCy for Finnish morphological analysis if available
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("fi_core_news_sm")
                logger.info("spaCy Finnish language processor initialized")
            except Exception as e:
                logger.error(f"Failed to initialize spaCy: {e}")
        
        # Finnish-specific stopwords
        self.stopwords = {
            "ja", "tai", "mutta", "että", "jotta", "koska", "kun", "jos",
            "kuin", "sekä", "eli", "sitten", "sillä", "niin", "on", "ovat",
            "oli", "olivat", "en", "et", "ei", "emme", "ette", "eivät"
        }
        
        # Common Finnish inflection endings
        self.inflection_endings = [
            "ssa", "ssä", "sta", "stä", "lla", "llä", "lta", "ltä",
            "lle", "ksi", "tta", "ttä", "n", "t", "a", "ä", "in"
        ]
        
        # Initialize simple lemmatizer for fallback
        self.simple_lemmatizer = SimpleFinnishLemmatizer()
    
    def analyze_word(self, word):
        """Get base form and morphological analysis of a Finnish word using spaCy"""
        if not self.nlp:
            # Fall back to simple lemmatizer
            baseform = self.simple_lemmatizer.lemmatize(word)
            return {"baseform": baseform}
           
        # Process the word with spaCy
        doc = self.nlp(word)
        if len(doc) == 0:
            return {"baseform": word}
            
        token = doc[0]  # Get the first (and likely only) token
        
        # Return baseform and morphological info
        return {
            "baseform": token.lemma_ or word,  # Lemma is the base form
            "class": token.pos_,               # Part of speech
            "case": self._get_case(token),     # Try to determine case
            "number": "SING" if "Number=Sing" in token.morph else "PLUR" if "Number=Plur" in token.morph else "",
            "person": self._get_person(token), # Person information
            "tense": self._get_tense(token)    # Tense information
        }
    
    def _get_case(self, token):
        """Extract case information from token morphology"""
        if "Case=Nom" in token.morph: return "nom"
        if "Case=Gen" in token.morph: return "gen"
        if "Case=Par" in token.morph: return "par"
        if "Case=Ine" in token.morph: return "ine"
        if "Case=Ela" in token.morph: return "ela"
        if "Case=Ill" in token.morph: return "ill"
        if "Case=Ade" in token.morph: return "ade"
        if "Case=Abl" in token.morph: return "abl"
        if "Case=All" in token.morph: return "all"
        if "Case=Ess" in token.morph: return "ess"
        if "Case=Tra" in token.morph: return "tra"
        return ""
    
    def _get_person(self, token):
        """Extract person information from token morphology"""
        if "Person=1" in token.morph: return "1"
        if "Person=2" in token.morph: return "2"
        if "Person=3" in token.morph: return "3"
        return ""
    
    def _get_tense(self, token):
        """Extract tense information from token morphology"""
        if "Tense=Pres" in token.morph: return "PRESENT"
        if "Tense=Past" in token.morph: return "PAST"
        return ""
    
    def normalize_text(self, text):
        """Normalize Finnish text to baseforms for better matching using spaCy"""
        if not self.nlp:
            # Fall back to simple word-by-word lemmatization
            words = text.lower().split()
            normalized_words = []
            for word in words:
                if word not in self.stopwords:
                    normalized_words.append(self.simple_lemmatizer.lemmatize(word))
                else:
                    normalized_words.append(word)
            return " ".join(normalized_words)
            
        # Process the text with spaCy
        doc = self.nlp(text)
        
        normalized_words = []
        for token in doc:
            if token.text.lower() not in self.stopwords:
                normalized_words.append(token.lemma_)
            else:
                normalized_words.append(token.text)
                
        return " ".join(normalized_words)
    
    def extract_entities(self, text):
        """Extract named entities from Finnish text using spaCy"""
        entities = {
            "locations": [],
            "programs": [],
            "dates": []
        }
        
        if not self.nlp:
            # Basic extraction using simple rules when spaCy is not available
            words = text.split()
            for word in words:
                # Check for capitalized words (potential proper nouns)
                if word and word[0].isupper():
                    # If ends with location suffixes
                    if any(word.lower().endswith(suffix) for suffix in ["ssa", "ssä", "lla", "llä"]):
                        entities["locations"].append(self.simple_lemmatizer.lemmatize(word))
                    # Potential program names
                    elif len(word) > 3 and not any(word.lower().endswith(s) for s in ["ssä", "ssa", "llä", "lla"]):
                        entities["programs"].append(word)
            return entities
        
        # Process the text with spaCy
        doc = self.nlp(text)
        
        # Extract entities based on spaCy's named entity recognition
        for ent in doc.ents:
            if ent.label_ == "LOC":
                entities["locations"].append(ent.text)
            elif ent.label_ == "PRODUCT" or ent.label_ == "ORG":
                entities["programs"].append(ent.text)
            elif ent.label_ == "DATE" or ent.label_ == "TIME":
                entities["dates"].append(ent.text)
                
        # Additional entity extraction for Finnish locations based on case
        for token in doc:
            # Common Finnish location endings
            if self._get_case(token) in ["ine", "ade"]:  # Inessive or adessive case
                if token.text not in [e.text for e in doc.ents]:  # Not already recognized as entity
                    entities["locations"].append(token.lemma_)
        
        return entities
    
    def enhance_intent_classification(self, intent_classifier):
        """Improve the existing IntentClassifier with Finnish optimizations"""
        # Add Finnish-specific intents
        self._add_finnish_intents(intent_classifier)
        
        # Store the original classify_intent method
        original_classify = intent_classifier.classify_intent
        
        # Define a new method that uses our Finnish language processing
        def enhanced_classify_intent(text):
            # Try to normalize Finnish text
            normalized_text = self.normalize_text(text)
            
            # Extract Finnish entities
            entities = self.extract_entities(text)
            
            # Call the original classification
            intent, params = original_classify(text)
            
            # Enhance the parameters with entity information
            if intent == "open_program" and entities["programs"] and not params.get("program"):
                params["program"] = entities["programs"][0]
                
            if intent == "search_google" and not params.get("query"):
                # Remove stopwords and extract main query terms
                query_terms = [w for w in normalized_text.split() if w not in self.stopwords]
                params["query"] = " ".join(query_terms)
            
            if intent == "get_finnish_weather" and entities["locations"] and not params.get("location"):
                params["location"] = entities["locations"][0]
            
            return intent, params
        
        # Replace the original method with our enhanced version
        intent_classifier.classify_intent = enhanced_classify_intent
        
        return intent_classifier
    
    def _add_finnish_intents(self, intent_classifier):
        """Add Finnish-specific intents to the classifier."""
        
        # Add weather intent examples
        intent_classifier.intent_examples["get_finnish_weather"] = [
            "mikä on sää {location}",
            "millainen sää on {location}",
            "kerro säästä {location}",
            "sääennuste {location}",
            "onko {location} sateista",
            "sataako {location}",
            "mikä on lämpötila {location}",
            "paljonko on astetta {location}",
            "tuleeko {location} sadetta",
            "millainen keli on {location}",
            "minkälainen sää on {location}",
            "sää paikkakunnalla {location}",
            "kerro sää {location}",
            "millainen on sää {location}",
            "päivän sää {location}"
        ]
        
        # Add news intent examples
        intent_classifier.intent_examples["get_finnish_news"] = [
            "kerro uutiset",
            "uusimmat uutiset",
            "mitä uutisia on",
            "päivän uutiset",
            "näytä uutiset",
            "uutisia aiheesta {category}",
            "lue uutiset",
            "mitä maailmalla tapahtuu",
            "mitä suomessa tapahtuu",
            "kerro uutisia",
            "uutiskatsaus",
            "kategoria {category} uutiset",
            "uutiset aiheesta {category}",
            "mielenkiintoisia uutisia",
            "kerro tärkeimmät uutiset"
        ]
        
        # Add holidays intent examples
        intent_classifier.intent_examples["get_finnish_holidays"] = [
            "mitkä ovat seuraavat pyhäpäivät",
            "tulevat pyhäpäivät",
            "milloin on seuraava pyhäpäivä",
            "luettele pyhäpäivät",
            "kerro tulevista pyhäpäivistä",
            "milloin on vapaa",
            "milloin on vapaapäivä",
            "näytä juhlapyhät",
            "kerro juhlapyhät",
            "mitkä ovat juhlapyhät",
            "milloin on seuraava juhlapyhä",
            "koska on seuraava vapaa",
            "suomen pyhäpäivät",
            "kalenterin pyhäpäivät",
            "milloin on pyhä"
        ]
        
        # Add transport intent examples
        intent_classifier.intent_examples["get_finnish_transport"] = [
            "miten pääsen {origin} {destination}",
            "näytä reitti {origin} {destination}",
            "julkinen liikenne {origin} {destination}",
            "bussi {origin} {destination}",
            "juna {origin} {destination}",
            "millä pääsen {origin} {destination}",
            "reittiopas {origin} {destination}",
            "hae reitti {origin} {destination}",
            "miten matkustan {origin} {destination}",
            "millä kulkuneuvolla {origin} {destination}",
            "milloin lähtee bussi {origin} {destination}",
            "milloin lähtee juna {origin} {destination}",
            "näytä aikataulut {origin} {destination}",
            "miten kuljen {origin} {destination}",
            "millä menen {origin} {destination}"
        ]
        
        # Compile the new patterns
        for intent, examples in intent_classifier.intent_examples.items():
            intent_classifier.intent_patterns[intent] = [
                intent_classifier._example_to_regex(example) for example in examples
            ]
        
        # Enhance parameter extraction for Finnish-specific intents
        original_extract = intent_classifier._extract_params_from_text
        
        def enhanced_extract_params(text, intent):
            """Enhanced parameter extraction with Finnish language patterns."""
            # Use original extraction first
            params = original_extract(text, intent)
            
            # Add Finnish-specific enhancements
            if intent == "get_finnish_weather" and not params.get("location"):
                # Extract location from common Finnish weather phrases
                location_markers = ["sää", "säästä", "sataako", "keli", "lämpötila", "astetta"]
                for marker in location_markers:
                    if marker in text.lower():
                        parts = text.lower().split(marker, 1)
                        if len(parts) > 1 and parts[1].strip():
                            # Extract location from after the marker
                            params["location"] = parts[1].strip()
                            break
                        elif len(parts) > 0 and parts[0].strip():
                            # Extract location from before the marker
                            words = parts[0].strip().split()
                            if words:
                                # Take the last word before the marker as location
                                params["location"] = words[-1]
                                break
            
            elif intent == "get_finnish_news" and not params.get("category"):
                # Extract news category from common Finnish phrases
                category_markers = ["aiheesta", "kategoria", "aihealue", "uutiset"]
                for marker in category_markers:
                    if marker in text.lower():
                        parts = text.lower().split(marker, 1)
                        if len(parts) > 1 and parts[1].strip():
                            # Extract category from after the marker
                            params["category"] = parts[1].strip()
                            break
            
            elif intent == "get_finnish_transport":
                # Extract origin and destination from Finnish transport phrases
                transport_markers = ["pääsen", "reitti", "matkustan", "kuljen", "menen"]
                for marker in transport_markers:
                    if marker in text.lower():
                        parts = text.lower().split(marker, 1)
                        if len(parts) > 1 and parts[1].strip():
                            # Try to extract "from X to Y" pattern in Finnish
                            from_to_text = parts[1].strip()
                            
                            # Common Finnish prepositions for from/to
                            from_markers = ["sta", "stä", "lta", "ltä", "mistä", "lähtien"]
                            to_markers = ["han", "hin", "lle", "seen", "mihin", "kohti"]
                            
                            # Check for words containing these markers
                            words = from_to_text.split()
                            for i, word in enumerate(words):
                                # Check if this word might be the origin
                                is_origin = any(word.endswith(marker) for marker in from_markers)
                                
                                # Check if next word might be the destination
                                if i < len(words) - 1:
                                    next_word = words[i + 1]
                                    is_destination = any(next_word.endswith(marker) for marker in to_markers)
                                    
                                    if is_origin and is_destination:
                                        params["origin"] = word.rstrip("".join(from_markers))
                                        params["destination"] = next_word.rstrip("".join(to_markers))
                                        break
                            
                            # If we didn't find from/to pattern, just use first and last word
                            if not params.get("origin") and not params.get("destination"):
                                if len(words) >= 2:
                                    params["origin"] = words[0]
                                    params["destination"] = words[-1]
                            
                            break
            
            return params
        
        # Replace the parameter extraction method
        intent_classifier._extract_params_from_text = enhanced_extract_params

# Simple Finnish rule-based lemmatizer as a fallback
class SimpleFinnishLemmatizer:
    """
    A very basic rule-based Finnish lemmatizer as fallback when spaCy is not available.
    This is simplified and won't handle all Finnish morphology but can help with basic forms.
    """
    
    def __init__(self):
        # Common Finnish word endings and their base forms
        self.endings = {
            "ssa": "",
            "ssä": "",
            "sta": "",
            "stä": "",
            "lla": "",
            "llä": "",
            "lta": "",
            "ltä": "",
            "lle": "",
            "ksi": "",
            "t": "",  # Plural nominative
            "ja": "",  # Plural partitive
            "jä": "",
            "a": "",   # Singular partitive
            "ä": "",
            "n": "",   # Genitive
            "en": "i", # Common vowel change in genitive
            "in": "in"  # Superlative
        }
        
        # Common Finnish verb endings in different tenses
        self.verb_endings = {
            "n": "",    # First person singular (minä)
            "t": "",    # Second person singular (sinä)
            "mme": "",  # First person plural (me)
            "tte": "",  # Second person plural (te)
            "vat": "a", # Third person plural (he)
            "vät": "ä", # Third person plural (he) for front vowel verbs
            "i": "a",   # Past tense
            "in": "a",  # First person past (minä)
            "it": "a",  # Second person past (sinä)
            "imme": "a", # First person plural past (me)
            "itte": "a", # Second person plural past (te)
            "ivat": "a", # Third person plural past (he)
            "ivät": "ä"  # Third person plural past (he) for front vowel verbs
        }
        
    def lemmatize(self, word):
        """
        Try to get base form of a Finnish word using simple rules.
        This is a very simplified implementation and won't work for all words.
        """
        word = word.lower()
        original = word
        
        # Check for verb endings
        for ending, replacement in self.verb_endings.items():
            if word.endswith(ending) and len(word) > len(ending) + 2:
                # Only apply to likely verbs (words long enough)
                word = word[:-len(ending)] + replacement
                return word
                
        # Check for nominal (noun, adjective) endings
        for ending, replacement in self.endings.items():
            if word.endswith(ending) and len(word) > len(ending) + 2:
                if replacement:
                    word = word[:-len(ending)] + replacement
                else:
                    word = word[:-len(ending)]
                    
                # For Finnish words ending with two consonants after removing case ending,
                # often the base form has a vowel between them
                if len(word) >= 2:
                    if word[-2:].isalpha() and not any(c in 'aeiouyäö' for c in word[-2:]):
                        # Insert common vowel between consonants
                        # This is a simplification and won't always be correct
                        if any(c in 'aou' for c in word[:-2]):  # Back vowel harmony
                            word = word[:-1] + 'a' + word[-1]
                        else:  # Front vowel harmony
                            word = word[:-1] + 'ä' + word[-1]
                
                return word
                
        # If no rules matched or word is very short, return original
        return original

# ----------------------------------------------------------------
# Basic Keyboard Input Recognition (Fallback)
# ----------------------------------------------------------------
class BasicKeyboardRecognizer:
    """
    Simple keyboard-based input as a fallback for speech recognition.
    This class emulates the interface of the speech recognizer but uses keyboard input.
    """
    
    def __init__(self):
        self.result_queue = queue.Queue()
        self.is_listening = False
        self.recognition_stopped = threading.Event()
        self.recognition_stopped.set()  # Initially not running
        self.input_thread = None
        
    def start_listening(self):
        """Start listening for keyboard input in a separate thread."""
        if not self.recognition_stopped.is_set():
            logger.info("Keyboard input is already active")
            return True
            
        self.is_listening = True
        self.recognition_stopped.clear()
        
        # Start a thread to handle keyboard input
        self.input_thread = threading.Thread(
            target=self._input_worker,
            daemon=True
        )
        self.input_thread.start()
        
        logger.info("Keyboard input recognition started")
        return True
        
    def stop_listening(self):
        """Stop listening for keyboard input."""
        if self.recognition_stopped.is_set():
            logger.info("Keyboard input is not active")
            return
            
        self.is_listening = False
        self.recognition_stopped.set()
        
        logger.info("Keyboard input recognition stopped")
        
    def _input_worker(self):
        """Worker thread for keyboard input."""
        logger.info("Keyboard input worker thread started")
        
        try:
            while not self.recognition_stopped.is_set():
                # Non-blocking check for Enter key
                if msvcrt_available:
                    import msvcrt
                    if msvcrt.kbhit():
                        key = msvcrt.getch()
                        # Check if Enter key was pressed
                        if key == b'\r':
                            # Clear the buffer
                            while msvcrt.kbhit():
                                msvcrt.getch()
                                
                            # Get input from user
                            user_input = input("\nKirjoita viestisi: ")
                            
                            if user_input.strip():
                                # Put the input in the queue
                                self.result_queue.put(user_input)
                                
                # Brief pause to avoid high CPU usage
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in keyboard input worker: {e}")
        finally:
            self.recognition_stopped.set()
            logger.info("Keyboard input worker thread ended")
            
    def get_recognized_text(self, timeout=None):
        """Get the next recognized text (keyboard input)."""
        if self.recognition_stopped.is_set():
            return None
            
        try:
            return self.result_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
            
    def listen_for_wake_word(self, wake_word="avustaja", callback=None, timeout=0.1):
        """
        Emulate wake word detection for keyboard input.
        For keyboard, Enter key is the "wake word".
        """
        # Always return False - keyboard doesn't use wake word
        # The main loop will check keyboard input separately
        return False
        
    def listen_for_command(self, timeout=10):
        """Listen for a command (keyboard input)."""
        # For keyboard, this is the same as get_recognized_text
        return self.get_recognized_text(timeout=timeout)


# ----------------------------------------------------------------
# Enhanced Finnish Speech Recognition with Azure SDK
# ----------------------------------------------------------------
class AzureSpeechRecognizer:
    """
    Enhanced speech recognition using Azure Speech SDK directly with Finnish optimization.
    """
    
    def __init__(self, speech_key, speech_region):
        self.speech_key = speech_key
        self.speech_region = speech_region
        
        # Recognition state
        self.is_listening = False
        self.recognition_stopped = threading.Event()
        self.recognition_stopped.set()  # Initially not running
        
        # Result queue
        self.result_queue = queue.Queue()
        
        # Speech recognizer 
        self.speech_recognizer = None
        
        logger.info("Azure Speech Recognizer initializing")
        self._setup_recognizer()
    
    def _setup_recognizer(self):
        """Setup the Azure Speech Recognizer with Finnish optimizations."""
        try:
            # Create a speech config with Finnish language
            speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
            speech_config.speech_recognition_language = "fi-FI"  # Use Finnish
            
            # Set recognition optimization for better Finnish language support
            # Enable dictation mode for better recognition of natural speech
            speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "1500")
            speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "1500")
            speech_config.enable_dictation()  # Enhance recognition quality for conversational speech
            
            # Create audio config for microphone
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            
            # Create a temporary recognizer to test if everything works
            test_recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config
            )
            
            # Do a quick test recognition
            logger.info("Testing Azure speech recognition...")
            result = test_recognizer.recognize_once()
            
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                logger.info("Azure Speech Recognition test successful")
                return True
            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.info("Azure Speech Recognition test completed with no match")
                return True
            else:
                logger.warning(f"Azure Speech Recognition test failed: {result.reason}")
                return False
        
        except Exception as e:
            logger.error(f"Failed to initialize Azure Speech Recognition: {e}")
            return False
    
    def start_listening(self):
        """Start speech recognition."""
        logger.info("Starting Azure speech recognition")
        
        # Check if already listening
        if not self.recognition_stopped.is_set():
            logger.info("Azure recognition is already active")
            return True
            
        self.is_listening = True
        self.recognition_stopped.clear()
        
        try:
            # Create a speech config with Finnish language
            speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
            speech_config.speech_recognition_language = "fi-FI"  # Use Finnish
            
            # Set recognition optimization for better Finnish language support
            speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "1500")
            speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "1500")
            speech_config.enable_dictation()  # Enhance recognition quality for conversational speech
            
            # Create audio config for microphone
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            
            # Create recognizer
            self.speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config
            )
            
            # Set up callbacks
            self.speech_recognizer.recognized.connect(
                lambda evt: self._handle_recognized(evt)
            )
            
            self.speech_recognizer.canceled.connect(
                lambda evt: self._handle_canceled(evt)
            )
            
            # Start continuous recognition
            self.speech_recognizer.start_continuous_recognition()
            
            # Start a monitoring thread
            recognition_thread = threading.Thread(
                target=self._recognition_worker,
                daemon=True
            )
            recognition_thread.start()
            
            logger.info("Azure speech recognition started")
            return True
        except Exception as e:
            logger.error(f"Failed to start Azure speech recognition: {e}")
            self.is_listening = False
            self.recognition_stopped.set()
            return False
    
    def stop_listening(self):
        """Stop speech recognition."""
        if self.recognition_stopped.is_set():
            logger.info("Azure recognition is not active")
            return
            
        self.is_listening = False
        self.recognition_stopped.set()
        
        # Clean up the recognizer
        if self.speech_recognizer:
            try:
                self.speech_recognizer.stop_continuous_recognition()
                self.speech_recognizer = None
            except Exception as e:
                logger.error(f"Error stopping Azure recognition: {e}")
            
        logger.info("Azure speech recognition stopped")
    
    def _handle_recognized(self, evt):
        """Handle recognized speech result."""
        try:
            result_text = evt.result.text
            
            if result_text and not result_text.isspace():
                logger.info(f"Azure recognizer got: {result_text}")
                self.result_queue.put(result_text)
        except Exception as e:
            logger.error(f"Error in Azure recognition result handler: {e}")
    
    def _handle_canceled(self, evt):
        """Handle cancellation events."""
        try:
            cancellation = evt.cancellation_details
            if cancellation.reason == speechsdk.CancellationReason.Error:
                logger.error(f"Azure recognition error: {cancellation.error_details}")
            else:
                logger.info(f"Azure recognition canceled: {cancellation.reason}")
        except Exception as e:
            logger.error(f"Error in Azure cancellation handler: {e}")
    
    def _recognition_worker(self):
        """Worker thread to monitor continuous speech recognition."""
        logger.info("Azure recognition worker thread started")
        
        try:
            # Wait until recognition is stopped
            while not self.recognition_stopped.is_set():
                # Brief pause to avoid high CPU usage
                time.sleep(0.1)
                
        except Exception as e:
            logger.error(f"Error in Azure recognition worker: {e}")
        finally:
            # Ensure recognition is properly stopped
            if self.speech_recognizer and self.is_listening:
                try:
                    self.speech_recognizer.stop_continuous_recognition()
                except Exception as e:
                    logger.error(f"Error stopping recognition in worker: {e}")
                    
            self.recognition_stopped.set()
            logger.info("Azure recognition worker thread ended")
    
    def get_recognized_text(self, timeout=None):
        """Get the next recognized text."""
        if self.recognition_stopped.is_set():
            return None
            
        try:
            return self.result_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
    
    def listen_for_wake_word(self, wake_word="avustaja", callback=None, timeout=0.1):
        """
        Listen for wake word in speech.
        
        Args:
            wake_word: Word that activates the assistant
            callback: Function to call when wake word is detected
            timeout: How long to wait for each recognition result
            
        Returns:
            True if wake word detected, False otherwise
        """
        # Ensure recognition is active
        if self.recognition_stopped.is_set():
            if not self.start_listening():
                return False
        
        try:
            # Get the next piece of speech
            speech_text = self.get_recognized_text(timeout=timeout)
            
            # Check if we got any speech
            if not speech_text:
                return False
            
            # Look for wake word (case-insensitive)
            if wake_word.lower() in speech_text.lower():
                logger.info(f"Wake word detected in: {speech_text}")
                
                # Call the callback if provided
                if callback:
                    callback(speech_text)
                
                return True
            
            # Also check for common greetings that might precede the wake word
            common_greetings = ["hei", "terve", "moi"]
            if any(greeting in speech_text.lower() for greeting in common_greetings):
                if any(word in speech_text.lower() for word in ["avustaja", "assistentti"]):
                    logger.info(f"Wake greeting detected in: {speech_text}")
                    
                    # Call the callback if provided
                    if callback:
                        callback(speech_text)
                    
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error while listening for wake word: {e}")
            return False
    
    def listen_for_command(self, timeout=10):
        """Listen for a command."""
        # Ensure recognition is active
        if self.recognition_stopped.is_set():
            if not self.start_listening():
                return None
        
        # Return whatever comes next in the queue
        return self.get_recognized_text(timeout=timeout)


# ----------------------------------------------------------------
# Extremely Simple Speech Recognition (Backup Option)
# ----------------------------------------------------------------
class SimpleSpeechRecognizer:
    """
    Minimal speech recognition implementation to avoid encoding issues.
    This is a backup in case the full implementation fails.
    """
    
    def __init__(self, speech_key, speech_region):
        self.speech_key = speech_key
        self.speech_region = speech_region
        
        # Recognition state
        self.is_listening = False
        self.recognition_stopped = threading.Event()
        self.recognition_stopped.set()  # Initially not running
        
        # Result queue
        self.result_queue = queue.Queue()
        
        # Speech recognizer 
        self.speech_recognizer = None
        
        logger.info("Simple Speech Recognizer initialized")
    
    def start_listening(self):
        """Start the minimal speech recognition process."""
        logger.info("Starting simple speech recognition")
        
        # Check if already listening
        if not self.recognition_stopped.is_set():
            logger.info("Simple recognition is already active")
            return True
            
        self.is_listening = True
        self.recognition_stopped.clear()
        
        try:
            # Create a minimal speech config with English to avoid encoding issues
            speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.speech_region)
            speech_config.speech_recognition_language = "en-US"  # Use English to avoid encoding issues
            
            # Create audio config for microphone
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            
            # Create recognizer
            self.speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config
            )
            
            # Set up callbacks
            self.speech_recognizer.recognized.connect(
                lambda evt: self._handle_recognized(evt)
            )
            
            self.speech_recognizer.canceled.connect(
                lambda evt: self._handle_canceled(evt)
            )
            
            # Start recognition thread
            recognition_thread = threading.Thread(
                target=self._recognition_worker,
                daemon=True
            )
            recognition_thread.start()
            
            logger.info("Simple speech recognition started")
            return True
        except Exception as e:
            logger.error(f"Failed to start simple speech recognition: {e}")
            self.is_listening = False
            self.recognition_stopped.set()
            return False
    
    def stop_listening(self):
        """Stop speech recognition."""
        if self.recognition_stopped.is_set():
            logger.info("Simple recognition is not active")
            return
            
        self.is_listening = False
        self.recognition_stopped.set()
        
        # Clean up the recognizer
        if self.speech_recognizer:
            self.speech_recognizer = None
            
        logger.info("Simple speech recognition stopped")
    
    def _handle_recognized(self, evt):
        """Handle recognized speech result."""
        try:
            result_text = evt.result.text
            
            if result_text and not result_text.isspace():
                logger.info(f"Simple recognizer got: {result_text}")
                self.result_queue.put(result_text)
        except Exception as e:
            logger.error(f"Error in simple recognition result handler: {e}")
    
    def _handle_canceled(self, evt):
        """Handle cancellation events."""
        try:
            cancellation = evt.cancellation_details
            if cancellation.reason == speechsdk.CancellationReason.Error:
                logger.error(f"Simple recognition error: {cancellation.error_details}")
            else:
                logger.info(f"Simple recognition canceled: {cancellation.reason}")
        except Exception as e:
            logger.error(f"Error in simple cancellation handler: {e}")
    
    def _recognition_worker(self):
        """Worker thread for continuous speech recognition."""
        logger.info("Simple recognition worker thread started")
        
        try:
            while not self.recognition_stopped.is_set():
                # Only create a new recognizer if needed
                if not self.speech_recognizer:
                    break
                    
                # Use simple one-shot recognition 
                result = self.speech_recognizer.recognize_once()
                
                if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    # This is handled by the event callback
                    pass
                
                # Brief pause between recognition attempts
                time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in simple recognition worker: {e}")
        finally:
            self.recognition_stopped.set()
            logger.info("Simple recognition worker thread ended")
    
    def get_recognized_text(self, timeout=None):
        """Get the next recognized text."""
        if self.recognition_stopped.is_set():
            return None
            
        try:
            return self.result_queue.get(block=True, timeout=timeout)
        except queue.Empty:
            return None
    
    def listen_for_wake_word(self, wake_word="hey", callback=None, timeout=0.1):
        """
        Simplified wake word detection - looks for any English greeting words.
        """
        # Ensure recognition is active
        if self.recognition_stopped.is_set():
            if not self.start_listening():
                return False
        
        try:
            # Get the next piece of speech
            speech_text = self.get_recognized_text(timeout=timeout)
            
            # Check if we got any speech
            if not speech_text:
                return False
            
            # Look for common English greeting words (since we're using English recognition)
            greeting_words = ["hey", "hi", "hello", "ok", "okay"]
            
            if any(word in speech_text.lower() for word in greeting_words):
                logger.info(f"Wake word detected in: {speech_text}")
                
                # Call the callback if provided
                if callback:
                    callback(speech_text)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error while listening for wake word: {e}")
            return False
    
    def listen_for_command(self, timeout=10):
        """Listen for a command."""
        # Ensure recognition is active
        if self.recognition_stopped.is_set():
            if not self.start_listening():
                return None
        
        # Return whatever comes next in the queue
        return self.get_recognized_text(timeout=timeout)

# ----------------------------------------------------------------
# Enhanced Finnish Text-to-Speech
# ----------------------------------------------------------------
class EnhancedFinnishTTS:
    """Enhanced Text-to-Speech optimized for Finnish language."""
    
    def __init__(self, speech_key, speech_region, 
                 language="fi-FI", voice="fi-FI-SelmaNeural",
                 use_ssml=True, cache_dir="speech_cache"):
        self.speech_key = speech_key
        self.speech_region = speech_region
        self.language = language
        self.voice = voice
        self.use_ssml = use_ssml
        self.cache_dir = cache_dir
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Finnish-specific pronunciation dictionary
        self.pronunciation_fixes = {
            # Common English loanwords with Finnish pronunciation
            "Google": "Guugel",
            "Microsoft": "Mikrosoft",
            "YouTube": "Juutuub",
            "Chrome": "Kroum",
            "Edge": "Edz",
            "Firefox": "Faiafoks",
            "Windows": "Vindous",
            
            # Common abbreviations
            "TPS": "tee pee äs",
            "HIFK": "hoo ii äf koo",
            "EU": "ee uu",
            "USA": "uu äs aa",
            "VR": "vee är",
            "HS": "hoo äs",
            "YLE": "yy lee"
        }
        
        # Setup speech SDK
        self._setup_speech_sdk()
    
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
            logger.info("Testing Finnish TTS...")
            test_ssml = '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="fi-FI"><voice name="fi-FI-SelmaNeural"><prosody volume="silent">Test</prosody></voice></speak>'
            test_result = self.speech_synthesizer.speak_ssml_async(test_ssml).get()
            
            if test_result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.info("Finnish TTS initialized and tested successfully")
            else:
                logger.warning(f"Speech test returned: {test_result.reason}")
                
        except Exception as e:
            logger.error(f"Error setting up Finnish TTS: {e}")
            self.speech_synthesizer = None
    
    def _fix_finnish_pronunciation(self, text):
        """Apply Finnish-specific pronunciation fixes"""
        result = text
        
        # Apply pronunciation dictionary replacements
        for word, pronunciation in self.pronunciation_fixes.items():
            pattern = r'\b' + re.escape(word) + r'\b'
            result = re.sub(pattern, pronunciation, result)
        
        # Handle numbers with Finnish pronunciation
        # In Finnish, decimal point is read as "pilkku" (comma)
        result = re.sub(r'(\d+)\.(\d+)', r'\1 pilkku \2', result)
        
        return result
    
    def _create_finnish_ssml(self, text):
        """Create SSML markup optimized for Finnish speech patterns."""
        # Apply Finnish pronunciation fixes
        text = self._fix_finnish_pronunciation(text)
        
        # Escape XML special characters
        text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&apos;")
        
        # Split into sentences for better prosody control
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Build SSML with sentence-level prosody adjustments
        ssml_sentences = []
        for sentence in sentences:
            # Check if this is a question
            is_question = sentence.strip().endswith("?")
            
            if is_question:
                # Finnish questions have rising intonation at the end
                ssml_sentences.append(f'<prosody rate="1.0" pitch="+0%">{sentence} <break strength="medium"/></prosody>')
            else:
                # Regular Finnish sentences have slightly falling intonation
                ssml_sentences.append(f'<prosody rate="1.0" pitch="+0%">{sentence} <break strength="medium"/></prosody>')
        
        # Complete SSML structure
        ssml = f"""
        <speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="{self.language}">
            <voice name="{self.voice}">
                {' '.join(ssml_sentences)}
            </voice>
        </speak>
        """
        return ssml
    
    def _get_cache_filename(self, text):
        """Generate a cache filename based on the text and voice."""
        # Create a unique hash based on text and voice parameters
        hash_input = f"{self.voice}:{self.language}:{text}"
        hash_value = hashlib.md5(hash_input.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_value}.wav")
    
    def speak(self, text):
        """Speak text using Finnish-optimized TTS with caching."""
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
                logger.info(f"Generating Finnish TTS for: {text[:30]}...")
                
                try:
                    # Use Finnish-optimized SSML
                    ssml = self._create_finnish_ssml(text)
                    
                    # Configure audio output to file
                    audio_config = speechsdk.audio.AudioConfig(filename=cache_file)
                    file_synthesizer = speechsdk.SpeechSynthesizer(
                        speech_config=self.speech_config, 
                        audio_config=audio_config
                    )
                    
                    # Generate speech from SSML
                    result = file_synthesizer.speak_ssml_async(ssml).get()
                    
                    # Check result
                    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                        logger.info(f"Speech synthesis saved to file: {cache_file}")
                        # Play the file
                        winsound.PlaySound(cache_file, winsound.SND_FILENAME)
                        return True
                    else:
                        raise Exception(f"Speech synthesis failed: {result.reason}")
                        
                except Exception as file_error:
                    # If file creation fails, try direct synthesis
                    logger.error(f"File-based synthesis failed: {file_error}")
                    logger.info("Trying direct speech synthesis...")
                    
                    result = self.speech_synthesizer.speak_ssml_async(self._create_finnish_ssml(text)).get()
                    
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
# Finnish Service Integrations
# ----------------------------------------------------------------
class FinnishServiceIntegrations:
    """Integration with Finnish-specific services and data sources."""
    
    def __init__(self):
        # Cache for API responses
        self.cache = {}
        
    def get_finnish_weather(self, location="Helsinki"):
        """
        Get weather information from Finnish Meteorological Institute.
        Uses the FMI (Ilmatieteen laitos) open data API.
        """
        # Clean up the location name
        location = location.strip().lower()
        
        # Map common location variations to standard names
        location_mapping = {
            "hki": "Helsinki",
            "stadi": "Helsinki",
            "lande": "Tampere",
            "tre": "Tampere",
            "tku": "Turku",
            "åbo": "Turku",
        }
        
        # Apply mapping if available
        location = location_mapping.get(location, location.title())
        
        try:
            # Check cache first (valid for 30 minutes)
            cache_key = f"weather_{location}"
            if cache_key in self.cache:
                timestamp, data = self.cache[cache_key]
                # If cache is less than 30 minutes old
                if (datetime.now() - timestamp).seconds < 1800:
                    return data
            
            # FMI API for latest observations
            # In a real implementation, this would use the actual FMI WFS API
            # For this example, we'll simulate a response
            
            # Simulate weather data
            temp = round(10 + 5 * (0.5 - (time.time() % 100) / 100), 1)  # Random between 5-15°C
            
            # Format the weather information in Finnish
            weather_info = f"Lämpötila {location}ssa on {temp} astetta Celsiusta."
            
            # Add some variety based on temperature
            if temp < 0:
                weather_info += " On pakkasta, muista pukeutua lämpimästi."
            elif temp < 10:
                weather_info += " On viileää."
            elif temp < 20:
                weather_info += " Sää on melko lämmin."
            else:
                weather_info += " On lämmintä."
            
            # Cache the result
            self.cache[cache_key] = (datetime.now(), weather_info)
            
            return weather_info
                
        except Exception as e:
            logger.error(f"Weather API error: {e}")
            return "Valitettavasti en saanut haettua säätietoja juuri nyt."
    
    def get_finnish_news(self, category="general"):
        """
        Get latest news headlines from Finnish news sources.
        """
        # Map categories to Finnish
        category_mapping = {
            "general": "yleiset",
            "sports": "urheilu",
            "politics": "politiikka",
            "economy": "talous",
            "entertainment": "viihde"
        }
        
        finnish_category = category_mapping.get(category, "yleiset")
        
        try:
            # Check cache (valid for 1 hour)
            cache_key = f"news_{finnish_category}"
            if cache_key in self.cache:
                timestamp, data = self.cache[cache_key]
                # If cache is less than 60 minutes old
                if (datetime.now() - timestamp).seconds < 3600:
                    return data
                    
            # This is a placeholder - real implementation would use actual Finnish news API
            # For example, YLE API (requires registration) or RSS feed parsing
            
            # Simulate API response with sample data
            sample_news = {
                "yleiset": [
                    "Hallitus keskustelee uusista taloustoimista",
                    "Koronatilanne parantunut koko maassa",
                    "Sääennuste lupaa lämpenevää säätä"
                ],
                "urheilu": [
                    "Suomi voitti jääkiekon MM-kultaa",
                    "HJK eteni eurokarsinnassa",
                    "Uusi yleisurheilutähti nousussa"
                ],
                "politiikka": [
                    "Eduskunta äänesti uudesta laista",
                    "Puolueiden kannatusprosentit muuttuneet",
                    "Ministerivaihdoksia hallituksessa"
                ]
            }
            
            # Get news for the requested category or fallback to general
            news_items = sample_news.get(finnish_category, sample_news["yleiset"])
            
            # Format the response in Finnish
            response = f"Tässä viimeisimmät uutiset kategoriasta {finnish_category}:\n"
            for i, item in enumerate(news_items, 1):
                response += f"{i}. {item}\n"
            
            # Cache the result
            self.cache[cache_key] = (datetime.now(), response)
            
            return response
            
        except Exception as e:
            logger.error(f"News API error: {e}")
            return "Valitettavasti en saanut haettua uutisia juuri nyt."
    
    def get_finnish_holidays(self):
        """
        Get upcoming Finnish holidays.
        """
        try:
            # Current date
            now = datetime.now()
            
            # Finnish holidays (simplified example)
            holidays = [
                {"date": datetime(now.year, 1, 1), "name": "Uudenvuodenpäivä"},
                {"date": datetime(now.year, 1, 6), "name": "Loppiainen"},
                {"date": datetime(now.year, 4, 7), "name": "Pitkäperjantai"},  # Example fixed date, would be calculated
                {"date": datetime(now.year, 4, 10), "name": "Pääsiäismaanantai"},  # Example fixed date, would be calculated
                {"date": datetime(now.year, 5, 1), "name": "Vappu"},
                {"date": datetime(now.year, 5, 18), "name": "Helatorstai"},  # Example fixed date, would be calculated
                {"date": datetime(now.year, 6, 24), "name": "Juhannuspäivä"},  # Example fixed date, would be calculated
                {"date": datetime(now.year, 11, 4), "name": "Pyhäinpäivä"},  # Example fixed date, would be calculated
                {"date": datetime(now.year, 12, 6), "name": "Itsenäisyyspäivä"},
                {"date": datetime(now.year, 12, 24), "name": "Jouluaatto"},
                {"date": datetime(now.year, 12, 25), "name": "Joulupäivä"},
                {"date": datetime(now.year, 12, 26), "name": "Tapaninpäivä"}
            ]
            
            # Find upcoming holidays
            upcoming = [h for h in holidays if h["date"] >= now]
            upcoming.sort(key=lambda x: x["date"])
            
            if not upcoming:
                return "Tänä vuonna ei ole enää jäljellä virallisia pyhäpäiviä."
            
            # Format response
            response = "Tulevat pyhäpäivät:\n"
            for holiday in upcoming[:3]:  # Show next 3 holidays
                date_str = holiday["date"].strftime("%d.%m.%Y")
                response += f"{holiday['name']} - {date_str}\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Holiday calculation error: {e}")
            return "En pystynyt hakemaan pyhäpäivätietoja."
            
    def get_finnish_transport(self, origin, destination):
        """
        Get public transport information between Finnish locations.
        """
        try:
            # Clean up location names
            origin = origin.strip().title()
            destination = destination.strip().title()
            
            # Map common abbreviations
            location_mapping = {
                "HKI": "Helsinki",
                "TRE": "Tampere",
                "TKU": "Turku",
                "OULU": "Oulu",
                "JKL": "Jyväskylä"
            }
            
            origin = location_mapping.get(origin.upper(), origin)
            destination = location_mapping.get(destination.upper(), destination)
            
            # Placeholder - would use actual public transit API
            # For example: Digitransit/HSL GraphQL API
            
            # Generate current time plus some offsets
            now = datetime.now()
            hours = now.hour
            minutes = now.minute
            
            # Simplified response with realistic-looking times
            response = f"Julkisen liikenteen reittejä välillä {origin} - {destination}:\n"
            
            # Generate some sample times
            next_hour = (hours + 1) % 24
            next_minutes = (minutes + 15) % 60
            
            response += f"1. Juna lähtee klo {hours:02d}:{minutes:02d}, perillä {next_hour:02d}:{next_minutes:02d}\n"
            response += f"2. Bussi lähtee klo {hours:02d}:{(minutes+30)%60:02d}, perillä {(hours+2)%24:02d}:{minutes:02d}\n"
            
            return response
            
        except Exception as e:
            logger.error(f"Transport API error: {e}")
            return f"En pystynyt hakemaan reittitietoja välille {origin} - {destination}."

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
            
            # Volume control intents
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
            
            # Brightness control intents
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
            
            # Power management intents
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
            
            # File creation intents
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
            
            # File editing intents
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
            
            # List files intent
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
                        
        # Extract parameters for volume control
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
                
        # Extract parameters for brightness control
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
                
        # Extract parameters for power management
        elif intent == "power_management":
            # Determine action
            if any(keyword in text.lower() for keyword in ["sammuta", "sulje"]):
                params["action"] = "shutdown"
            elif any(keyword in text.lower() for keyword in ["käynnistä uudelleen", "uudelleenkäynnistä", "reboot"]):
                params["action"] = "restart"
            elif any(keyword in text.lower() for keyword in ["peruuta", "keskeytä"]):
                params["action"] = "cancel"
            elif any(keyword in text.lower() for keyword in ["lepotilaan", "nukkumaan"]):
                params["action"] = "sleep"
            elif any(keyword in text.lower() for keyword in ["horrostila"]):
                params["action"] = "hibernate"
            elif any(keyword in text.lower() for keyword in ["kirjaudu ulos", "logout"]):
                params["action"] = "logout"
                
        # Extract parameters for file creation
        elif intent == "create_text_file":
            # Extract filename
            for keyword in ["nimeltä", "tiedosto", "tekstitiedosto", "muistiinpano", "nimellä"]:
                if keyword in text.lower():
                    parts = text.lower().split(keyword, 1)
                    if len(parts) > 1 and parts[1].strip():
                        params["filename"] = parts[1].strip()
                        break
                        
        # Extract parameters for file editing
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
                    example = example.replace("{url}", "yle.fi")
                elif "{query}" in example:
                    example = example.replace("{query}", "sää Helsinki")
                elif "{level}" in example:
                    example = example.replace("{level}", "50")
                elif "{filename}" in example:
                    example = example.replace("{filename}", "muistiinpanot")
                elif "{line_number}" in example:
                    example = example.replace("{line_number}", "3")
                elif "{location}" in example:
                    example = example.replace("{location}", "Helsinki")
                elif "{category}" in example:
                    example = example.replace("{category}", "urheilu")
                elif "{origin}" in example and "{destination}" in example:
                    example = example.replace("{origin}", "Helsinki").replace("{destination}", "Tampere")
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
# Check if msvcrt is available (Windows only)
# ----------------------------------------------------------------
msvcrt_available = False
try:
    import msvcrt
    msvcrt_available = True
except ImportError:
    logger.warning("msvcrt not available (non-Windows system). Keyboard input handling limited.")

# Function to check for keyboard input (non-blocking)
def check_for_keyboard_input():
    """
    Check for keyboard input in a non-blocking way.
    Returns the input string if Enter was pressed, None otherwise.
    """
    try:
        import msvcrt
        if msvcrt.kbhit():
            # Check if Enter key was pressed
            key = msvcrt.getch()
            if key == b'\r':
                # Clear the Enter key from the buffer
                msvcrt.getch()  # Read the \n that follows \r
                # Prompt for input
                return input("Kirjoita viestisi: ")
            else:
                # Consume the key press
                while msvcrt.kbhit():
                    msvcrt.getch()
        return None
    except ImportError:
        # Non-Windows platform, implement a different approach
        # For simplicity, use a timeout-based approach
        import select
        import sys
        
        # Check if input is available (non-blocking)
        rlist, _, _ = select.select([sys.stdin], [], [], 0)
        
        if rlist:
            # Input is available
            return input("Kirjoita viestisi: ")
        return None
    except Exception as e:
        logger.error(f"Error checking for keyboard input: {e}")
        return None

# ----------------------------------------------------------------
# Main Application
# ----------------------------------------------------------------
def main():
    """Run the enhanced Finnish voice assistant with improved error handling."""
    print("Suomalainen Ääniavustaja - Finnish Voice Assistant")
    print("================================================")
    print("Sano 'hei avustaja' aloittaaksesi tai paina Enter kirjoittaaksesi.")
    print("Sano 'lopeta' lopettaaksesi.")
    print()
    
    try:
        # Initialize components with proper error handling
        tts = None
        recognizer = None
        conversation = None
        finnish_services = None
        
        # Flag to determine if we need to use fallback keyboard input
        use_keyboard_only = False
        
        # Step 1: Initialize TTS first as it's needed for feedback
        try:
            tts = EnhancedFinnishTTS(
                speech_key=SPEECH_KEY,
                speech_region=SPEECH_REGION,
                language="fi-FI",
                voice="fi-FI-SelmaNeural",
                use_ssml=True
            )
            print("TTS initialized successfully")
        except Exception as tts_error:
            print(f"Error initializing TTS: {tts_error}")
            # Create a fallback TTS that just prints
            class FallbackTTS:
                def speak(self, text):
                    print(f"[VOICE WOULD SAY]: {text}")
                    return True
            tts = FallbackTTS()
            
        # Step 2: Initialize other services
        try:
            # Initialize Finnish language processor with graceful degradation
            finnish_processor = FinnishLanguageProcessor()
            if not SPACY_AVAILABLE:
                print("spaCy not available - using simplified Finnish language processing")
                
            # Initialize conversation manager
            conversation = ConversationManager(assistant_id=ASSISTANT_ID)
            
            # Initialize Finnish service integrations
            finnish_services = FinnishServiceIntegrations()
            
            print("Services initialized successfully")
        except Exception as service_error:
            print(f"Error initializing services: {service_error}")
            # We'll continue with whatever services were successfully initialized
        
        # Step 3: Try different speech recognition approaches
        recognizer = None
        
        # First, try the Azure Speech SDK recognizer
        try:
            print("Initializing Azure speech recognition...")
            azure_recognizer = AzureSpeechRecognizer(
                speech_key=SPEECH_KEY,
                speech_region=SPEECH_REGION
            )
            recognizer = azure_recognizer
            print("Azure speech recognizer initialized successfully")
        except Exception as azure_recog_error:
            print(f"Error initializing Azure speech recognizer: {azure_recog_error}")
            print("Falling back to simplified speech recognition...")
            
            # Try the simple speech recognizer as a fallback
            try:
                print("Initializing simplified speech recognition...")
                recognizer = SimpleSpeechRecognizer(
                    speech_key=SPEECH_KEY,
                    speech_region=SPEECH_REGION
                )
                print("Simple speech recognizer initialized successfully")
                # Inform the user about the fallback
                if tts:
                    tts.speak("Käytän yksinkertaistettua puheentunnistusta. Puheentunnistuksen tarkkuus voi olla rajoitettu.")
            except Exception as simple_recog_error:
                print(f"Error initializing simple speech recognizer: {simple_recog_error}")
                print("Falling back to keyboard input only")
                use_keyboard_only = True
        
        # If speech recognition failed entirely, fall back to keyboard input
        if use_keyboard_only or not recognizer:
            print("Using keyboard-only mode for input")
            recognizer = BasicKeyboardRecognizer()
        
        # Step 4: Initialize command registry
        global command_registry
        command_registry = EnhancedCommandRegistry()
        
        # Enhance the intent classifier with Finnish optimizations if available
        if finnish_processor and hasattr(finnish_processor, 'enhance_intent_classification'):
            command_registry.intent_classifier = finnish_processor.enhance_intent_classification(
                command_registry.intent_classifier
            )
        
        # Register commands (abbreviated for clarity)
        register_all_commands(command_registry, finnish_services)
        
        # Initial greeting
        print("Käynnistetään avustajaa...")
        tts.speak("Hei, olen suomalainen ääniavustaja.")
        
        if use_keyboard_only:
            tts.speak("Puheentunnistus ei ole käytettävissä. Käytä näppäimistöä kommunikointiin. Paina Enter kirjoittaaksesi.")
        else:
            if isinstance(recognizer, SimpleSpeechRecognizer):
                tts.speak("Käytän yksinkertaistettua puheentunnistusta. Sano 'hei avustaja' tai paina Enter aloittaaksesi.")
            else:
                tts.speak("Sano 'hei avustaja' tai paina Enter aloittaaksesi. Kuuntelen.")
                
            # Start recognition
            if not recognizer.start_listening():
                print("Warning: Could not start recognition. Falling back to keyboard input only.")
                tts.speak("Puheentunnistus ei ole käytettävissä. Käytä näppäimistöä kommunikointiin.")
                # Switch to keyboard recognizer
                recognizer = BasicKeyboardRecognizer()
                recognizer.start_listening()
        
        # Main event loop with improved stability
        run_main_loop(tts, recognizer, conversation, finnish_services)
            
    except KeyboardInterrupt:
        print("\nSuljetaan avustaja. Näkemiin!")
    except Exception as e:
        logger.error(f"Vakava virhe: {e}")
        print(f"Vakava virhe: {e}")
    finally:
        # Ensure speech recognition is stopped
        if 'recognizer' in locals() and recognizer is not None:
            try:
                recognizer.stop_listening()
            except:
                pass
        
    print("Ääniavustaja lopetettu.")

def register_all_commands(registry, finnish_services):
    """Register all commands to the command registry."""
    # Basic commands
    registry.register("open_program", open_program, "Avaa ohjelman tietokoneella")
    registry.register("open_website", open_website, "Avaa verkkosivun selaimessa")
    registry.register("search_google", search_google, "Hakee tietoa Googlesta")
    
    # System control (register stubs, actual implementations would be elsewhere)
    registry.register("control_volume", lambda action="get", level=None: f"Äänenvoimakkuus: {action} {level if level else ''}", "Säädä järjestelmän äänenvoimakkuutta")
    registry.register("control_brightness", lambda action="get", level=None: f"Kirkkaus: {action} {level if level else ''}", "Säädä näytön kirkkautta")
    registry.register("power_management", lambda action="status": f"Virtatoiminto: {action}", "Hallitse tietokoneen virtatilaa")
    
    # File operations (register stubs, actual implementations would be elsewhere)
    registry.register("create_text_file", lambda filename="muistio.txt": f"Luodaan tiedosto: {filename}", "Luo uusi tekstitiedosto")
    registry.register("edit_text_file", lambda filename="muistio.txt", operation="read", content=None, line_number=None: f"Muokataan tiedostoa: {filename}, operaatio: {operation}", "Lue tai muokkaa olemassa olevaa tekstitiedostoa")
    registry.register("list_text_files", lambda: "Tiedostolistaus (demotoiminto)", "Listaa saatavilla olevat tekstitiedostot")
    
    # Finnish-specific commands (if available)
    if finnish_services:
        registry.register(
            "get_finnish_weather",
            lambda location="Helsinki": finnish_services.get_finnish_weather(location),
            "Hakee säätiedot Ilmatieteen laitokselta"
        )
        
        registry.register(
            "get_finnish_news",
            lambda category="general": finnish_services.get_finnish_news(category),
            "Hakee uutisia suomalaisista lähteistä"
        )
        
        registry.register(
            "get_finnish_holidays",
            finnish_services.get_finnish_holidays,
            "Näyttää tulevat suomalaiset pyhäpäivät"
        )
        
        registry.register(
            "get_finnish_transport",
            lambda origin="Helsinki", destination="Tampere": 
                finnish_services.get_finnish_transport(origin, destination),
            "Hakee julkisen liikenteen tietoja"
        )

def run_main_loop(tts, recognizer, conversation, finnish_services):
    """Run the main event loop with improved error handling."""
    running = True
    conversation_active = False
    conversation_timeout = 0
    last_error_time = 0
    error_count = 0
    
    while running:
        try:
            # Check for keyboard input (non-blocking)
            keyboard_input = check_for_keyboard_input()
            
            if keyboard_input:
                # Process typed input
                user_input = keyboard_input.strip()
                if user_input:
                    print(f"Kirjoitit: {user_input}")
                    
                    # Check for exit
                    if user_input.lower() in ["lopeta", "exit", "quit", "poistu"]:
                        tts.speak("Suljetaan nyt, näkemiin!")
                        running = False
                        break
                    
                    # Reset conversation timeout when user interacts
                    conversation_active = True
                    conversation_timeout = 0
                    
                    # Process the input (system command or assistant)
                    process_user_input(user_input, tts, conversation)
                        
            elif isinstance(recognizer, BasicKeyboardRecognizer):
                # For keyboard-only mode, check the recognizer's queue
                speech_text = recognizer.get_recognized_text(timeout=0.1)
                if speech_text:
                    print(f"Kirjoitit: {speech_text}")
                    
                    # Check for exit
                    if speech_text.lower() in ["lopeta", "exit", "quit", "poistu"]:
                        tts.speak("Suljetaan nyt, näkemiin!")
                        running = False
                        break
                    
                    # Reset conversation timeout when user interacts
                    conversation_active = True
                    conversation_timeout = 0
                    
                    # Process the input
                    process_user_input(speech_text, tts, conversation)
            
            elif recognizer and not recognizer.recognition_stopped.is_set():
                # Check for wake word or direct command
                wake_word_detected = recognizer.listen_for_wake_word(
                    wake_word="avustaja",
                    callback=lambda text: tts.speak("Kyllä, miten voin auttaa?") if not conversation_active else None,
                    timeout=0.1
                )
                
                if wake_word_detected or conversation_active:
                    # If wake word just detected or conversation is ongoing
                    if wake_word_detected:
                        # Reset conversation state
                        conversation_active = True
                        conversation_timeout = 0
                    
                    # Listen for command
                    if wake_word_detected:
                        print("Kuuntelen komentoa...")
                    
                    command_text = recognizer.listen_for_command(timeout=5 if conversation_active else 10)
                    
                    if command_text:
                        print(f"Komento: {command_text}")
                        
                        # Reset timeout on new command
                        conversation_timeout = 0
                        
                        # Check for exit
                        if any(cmd in command_text.lower() for cmd in ["lopeta", "hyvästi", "pois", "exit"]):
                            tts.speak("Suljetaan nyt, näkemiin!")
                            running = False
                            break
                        
                        # Process the command
                        process_user_input(command_text, tts, conversation)
                    else:
                        # No command heard after timeout
                        conversation_timeout += 5
                        
                        # If no interaction for 30 seconds, end conversation mode
                        if conversation_timeout >= 30:
                            if conversation_active:
                                print("Keskustelu päättyi aikakatkaisuun.")
                                conversation_active = False
                                conversation_timeout = 0
                
                # Also process any other recognized speech for direct system commands
                speech_text = recognizer.get_recognized_text(timeout=0.1)
                if speech_text and not wake_word_detected and not conversation_active:
                    # Check for direct system commands without wake word
                    system_response = handle_system_command(speech_text)
                    if system_response:
                        print(f"Tunnistettu komento: {speech_text}")
                        print(f"Järjestelmäkomento: {system_response}")
                        # Only respond to high-confidence direct commands
                        if "virhe" not in system_response.lower():
                            tts.speak(system_response)
            
            # Reset error count if things are working well
            current_time = time.time()
            if current_time - last_error_time > 60:  # No errors for a minute
                error_count = 0
            
            # Small delay to avoid high CPU usage
            time.sleep(0.05)
            
        except KeyboardInterrupt:
            print("\nKeskeytys. Suljetaan...")
            tts.speak("Suljetaan nyt, näkemiin!")
            running = False
            break
            
        except Exception as e:
            error_count += 1
            last_error_time = time.time()
            
            logger.error(f"Virhe pääsilmukassa: {e}")
            print(f"Virhe: {e}")
            
            # If we're getting too many errors, try to restart the recognizer
            if error_count > 5 and recognizer and not isinstance(recognizer, BasicKeyboardRecognizer):
                print("Liian monta virhettä, yritetään käynnistää puheentunnistus uudelleen")
                try:
                    recognizer.stop_listening()
                    time.sleep(1)
                    recognizer.start_listening()
                    error_count = 0
                except Exception as restart_error:
                    logger.error(f"Error restarting recognizer: {restart_error}")
                    # If all else fails, switch to keyboard mode
                    print("Switching to keyboard-only mode")
                    try:
                        recognizer = BasicKeyboardRecognizer()
                        recognizer.start_listening()
                    except:
                        pass
            
            # Continue running despite errors

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

def process_user_input(text, tts, conversation):
    """Process user input - either as system command or send to assistant."""
    # Check if it's a system command
    system_response = handle_system_command(text)
    if system_response:
        print(f"Järjestelmäkomento: {system_response}")
        tts.speak(system_response)
    else:
        # Send to assistant
        try:
            conversation.add_user_message(text)
            response = conversation.run_assistant()
            
            if response:
                tts.speak(response)
            else:
                tts.speak("Valitettavasti en saanut vastausta.")
        except Exception as assistant_error:
            print(f"Virhe yhteydessä assistenttiin: {assistant_error}")
            tts.speak("Valitettavasti en voinut käsitellä pyyntöäsi juuri nyt.")
            
if __name__ == "__main__":
    main()