#!/usr/bin/env python3
"""
ElevenLabs Conversation Listener for Pi 5
----------------------------------------
This script connects directly to ElevenLabs' conversation API and plays back
audio responses using the Pi 5's audio output.
"""

import os
import asyncio
import time
import json
import base64
import websockets
import pyaudio
import wave
import logging
from typing import Dict, Any
from dotenv import load_dotenv
import urllib.parse
import warnings
import traceback
import subprocess
import tempfile

# Suppress ALSA warnings
# This needs to be done before importing pyaudio
warnings.filterwarnings("ignore", category=UserWarning)
os.environ['ALSA_CARD'] = 'Set'  # Reduces some ALSA warnings
if not os.environ.get('PYTHONWARNINGS'):
    os.environ['PYTHONWARNINGS'] = 'ignore:snd_pcm_hw_params_set_rate_near:UserWarning'

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("elevenlabs_listener.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("elevenlabs_listener")

# Suppress ALSA lib messages if possible
try:
    from ctypes import cdll
    cdll.LoadLibrary('libasound.so.2')
    from ctypes.util import find_library
    from ctypes import CDLL
    alsa_lib = CDLL(find_library('asound'))
    # Try to reduce error output from ALSA
    if hasattr(alsa_lib, 'snd_lib_error_set_handler'):
        # Define error handler
        error_handler = alsa_lib.snd_lib_error_set_handler
        error_handler(None)
except Exception as e:
    logger.debug(f"Failed to suppress ALSA messages: {e}")

# Try to use uvloop for better performance
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logger.info("Using uvloop for improved performance")
except ImportError:
    logger.info("uvloop not available, using standard event loop")

class ElevenLabsStreamingClient:
    def __init__(self, api_key, agent_id=None, voice_id=None, sample_rate=24000, audio_device=None, **kwargs):
        """Initialize the ElevenLabs streaming client."""
        self.api_key = api_key
        self.agent_id = agent_id or "DcCX20Y9uBl1bNWzZhaB"  # Default agent ID
        self.voice_id = voice_id or "21m00Tcm4TlvDq8ikWAM"  # Default voice ID
        self.sample_rate = sample_rate
        self.channels = 1
        self.audio_device = audio_device  # Store the audio device parameter
        
        # Audio stream properties
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.hifiberry_device_index = self._find_hifiberry_device()
        
        # WebSocket properties
        self.websocket = None
        self.running = False
        self.last_ping_time = None
        self.ping_interval = 10  # Send ping every 10 seconds
        
        # Audio queue for received chunks
        self.audio_queue = asyncio.Queue()
        
        # If specific audio device was specified, log it
        device_info = f", audio_device={self.audio_device}" if self.audio_device else ""
        logger.info(f"ElevenLabs Client initialized with agent_id={self.agent_id}{device_info}, HiFiBerry device: {'found' if self.hifiberry_device_index is not None else 'not found'}")

    def _find_hifiberry_device(self):
        """Find the HiFiBerry DAC audio device or user-specified device."""
        logger.info("Searching for audio output device...")
        
        # Check if user specified a device index 
        if self.audio_device is not None:
            try:
                device_index = int(self.audio_device)
                if device_index >= 0 and device_index < self.p.get_device_count():
                    device_info = self.p.get_device_info_by_index(device_index)
                    if device_info['maxOutputChannels'] > 0:
                        logger.info(f"Using user-specified audio device at index {device_index}: {device_info['name']}")
                        return device_index
                    else:
                        logger.warning(f"Specified device {device_index} has no output channels, falling back to default")
                else:
                    logger.warning(f"Specified device index {device_index} is out of range, falling back to default")
            except (ValueError, TypeError):
                # If audio_device is not a valid integer, assume it's a device name
                logger.info(f"Looking for audio device containing name: {self.audio_device}")
                
                # Search for a device with matching name
                for i in range(self.p.get_device_count()):
                    device_info = self.p.get_device_info_by_index(i)
                    logger.info(f"Found device: {device_info['name']} (index {i})")
                    if str(self.audio_device).lower() in device_info['name'].lower() and device_info['maxOutputChannels'] > 0:
                        logger.info(f"Found matching device at index {i}: {device_info['name']}")
                        return i
        
        # First try to find HiFiBerry through PyAudio
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)
            logger.info(f"Found device: {device_info['name']} (index {i})")
            if 'hifiberry' in device_info['name'].lower() and device_info['maxOutputChannels'] > 0:
                logger.info(f"Found HiFiBerry DAC at index {i}")
                return i
        
        # Try running aplay -L to see if HiFiBerry is listed there
        try:
            output = subprocess.check_output(['aplay', '-L']).decode('utf-8')
            if 'sysdefault:CARD=sndrpihifiberry' in output:
                logger.info("HiFiBerry found in aplay -L output, but not in PyAudio devices")
                # We'll use ALSA directly in the _play_audio method
                return -1  # Special marker for ALSA usage
        except Exception as e:
            logger.error(f"Error checking aplay devices: {e}")
        
        logger.warning("No suitable audio device found, will use default")
        return None

    def _setup_audio_stream(self):
        """Set up the PyAudio stream for playback."""
        # First check if the stream is already active
        if self.stream and self.stream.is_active():
            logger.debug("Audio stream already active")
            return True
            
        # Close any existing stream that might not be properly cleaned up
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.warning(f"Error closing existing stream: {e}")
            self.stream = None
            
        try:
            logger.debug("Setting up new audio stream")
            
            # Try specified device first if we have one
            if self.hifiberry_device_index is not None and self.hifiberry_device_index >= 0:
                try:
                    logger.info(f"Setting up audio stream using device index: {self.hifiberry_device_index}")
                    self.stream = self.p.open(
                        format=pyaudio.paInt16,
                        channels=self.channels,
                        rate=self.sample_rate,
                        output=True,
                        output_device_index=self.hifiberry_device_index,
                        frames_per_buffer=4096
                    )
                    logger.info(f"Successfully set up audio stream with device index {self.hifiberry_device_index}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to open stream with specified device index {self.hifiberry_device_index}: {e}")
                    # Continue to try default device
            
            # Try default device
            try:
                logger.info("Setting up audio stream using default output device")
                self.stream = self.p.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    output=True,
                    frames_per_buffer=4096
                )
                logger.info("Successfully set up audio stream with default device")
                return True
            except Exception as e:
                logger.error(f"Failed to open stream with default device: {e}")
                
            # If we got here, we couldn't set up a stream
            return False
                
        except Exception as e:
            logger.error(f"Failed to set up audio stream: {e}")
            traceback.print_exc()
            return False

    async def _play_audio(self, audio_data):
        """Play audio data through the configured output device."""
        # Decode base64 if needed
        if isinstance(audio_data, str):
            try:
                audio_bytes = base64.b64decode(audio_data)
            except Exception as e:
                logger.error(f"Failed to decode base64 audio data: {e}")
                return
        else:
            audio_bytes = audio_data
            
        try:
            # If we're using ALSA directly for HiFiBerry (device index -1)
            if self.hifiberry_device_index == -1:
                try:
                    # Create a temporary WAV file
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_path = temp_file.name
                        
                    # Convert raw audio bytes to WAV file
                    with wave.open(temp_path, 'wb') as wf:
                        wf.setnchannels(self.channels)
                        wf.setsampwidth(2)  # 16-bit audio
                        wf.setframerate(self.sample_rate)
                        wf.writeframes(audio_bytes)
                    
                    # Play using aplay with HiFiBerry device
                    subprocess.run(['aplay', '-D', 'sysdefault:CARD=sndrpihifiberry', temp_path])
                    
                    # Clean up temp file
                    os.unlink(temp_path)
                except Exception as e:
                    logger.error(f"Failed to play audio through ALSA: {e}")
                    # Fall back to PyAudio method
                    self._play_audio_pyaudio(audio_bytes)
            else:
                # Use standard PyAudio method
                self._play_audio_pyaudio(audio_bytes)
                
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
            traceback.print_exc()
            
    def _play_audio_pyaudio(self, audio_bytes):
        """Play audio using PyAudio stream."""
        if not audio_bytes or len(audio_bytes) == 0:
            logger.warning("Received empty audio data, skipping playback")
            return
            
        try:
            # Check if we need to create or restart the stream
            if not self.stream or not self.stream.is_active():
                self._setup_audio_stream()
                if not self.stream:
                    logger.error("Failed to set up audio stream")
                    return
                    
            # Log audio details
            logger.debug(f"Playing audio: {len(audio_bytes)} bytes")
            
            # Write audio to the stream in chunks
            CHUNK_SIZE = 4096
            for i in range(0, len(audio_bytes), CHUNK_SIZE):
                chunk = audio_bytes[i:i + CHUNK_SIZE]
                self.stream.write(chunk)
                
            logger.debug("Audio playback completed successfully")
            
        except Exception as e:
            logger.error(f"Error playing audio through PyAudio: {e}")
            traceback.print_exc()
            
            # Try to reset stream on error
            try:
                if self.stream:
                    self.stream.stop_stream()
                    self.stream.close()
                self.stream = None
                logger.info("Audio stream reset after error")
            except Exception as e2:
                logger.error(f"Error resetting audio stream: {e2}")

    async def connect(self):
        """Connect to the ElevenLabs WebSocket API."""
        if not self.api_key or not self.agent_id:
            logger.error("Missing API key or agent ID - cannot connect to ElevenLabs")
            return False
            
        try:
            # Build the WebSocket URL with agent_id and add the API key as a query parameter
            # This works with older versions of websockets that don't support extra_headers
            ws_url = f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={self.agent_id}&xi-api-key={self.api_key}"
            logger.info(f"Connecting to ElevenLabs at wss://api.elevenlabs.io/v1/convai/conversation?agent_id={self.agent_id}")
            
            # Connect to WebSocket
            self.websocket = await websockets.connect(ws_url)
            logger.info("WebSocket connection established")
            
            # Send initialization message indicating this is a listener only
            init_message = {
                "type": "conversation_initiation_client_data",
                "conversation_config_override": {
                    "prompt": "You are a friendly voice assistant. Keep responses brief and helpful.",
                    "first_message": "Hello, I'm your voice assistant. How can I help you today?",
                    "language": {"type": "autodetect"},
                    "tts": {
                        "voice_id": self.voice_id or "21m00Tcm4TlvDq8ikWAM"
                    }
                },
                "mode": "listening",
                "client_type": "companion_listener",
                "device_name": "pi5_speaker"
            }
            await self.websocket.send(json.dumps(init_message))
            logger.info("Initialization message sent")
            
            # Start a background task to send periodic pings to keep the connection alive
            asyncio.create_task(self._keep_alive())
            
            self.running = True
            self.last_ping_time = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to ElevenLabs WebSocket: {e}")
            traceback.print_exc()
            self.websocket = None
            self.running = False
            return False

    async def _keep_alive(self):
        """Send periodic pings to keep the connection alive."""
        ping_interval = 15  # seconds
        ping_id = 0
        
        while self.running and self.websocket:
            try:
                ping_id += 1
                # Use a simpler ping format since the complex one might not be supported
                await self.websocket.ping()
                logger.debug(f"Sent WebSocket ping #{ping_id}")
                await asyncio.sleep(ping_interval)
            except Exception as e:
                logger.error(f"Error in keep-alive: {e}")
                if not self.running:
                    break
                await asyncio.sleep(5)  # Wait before trying again

    async def listen(self):
        """Listen for responses from ElevenLabs."""
        if not self.websocket or not self.running:
            logger.error("Not connected to ElevenLabs")
            return
            
        try:
            logger.info("Starting to listen for responses")
            print("\n" + "="*60)
            print("🎧 ELEVENLABS LISTENER ACTIVE")
            print(f"🤖 Agent ID: {self.agent_id}")
            print("🔈 Waiting for audio responses...")
            print("="*60 + "\n")
            
            # Track total audio responses received
            audio_responses_count = 0
            last_audio_time = time.time()
            
            while self.running:
                try:
                    # Receive message from WebSocket with a timeout
                    try:
                        message = await asyncio.wait_for(self.websocket.recv(), timeout=30)
                        self.last_ping_time = time.time()
                    except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                        logger.warning("No message received for 30 seconds or connection closed, checking connection")
                        # Send a ping to check if still connected
                        try:
                            pong_waiter = await self.websocket.ping()
                            await asyncio.wait_for(pong_waiter, timeout=5)
                            logger.debug("Ping successful, connection still active")
                            continue
                        except Exception as e:
                            logger.error(f"Ping failed, connection may be down: {e}")
                            self.running = False
                            break
                    
                    # Parse as JSON
                    data = json.loads(message)
                    message_type = data.get("type")
                    
                    # Handle different message types
                    if message_type == "ping":
                        # Respond to pings to keep the connection alive
                        event_id = data.get("ping_event", {}).get("event_id")
                        if event_id:
                            await self.websocket.send(json.dumps({
                                "type": "pong",
                                "event_id": event_id
                            }))
                            self.last_ping_time = time.time()
                            logger.debug(f"Sent pong response to ping event: {event_id}")
                    
                    elif message_type == "audio":
                        # Handle audio response
                        audio_event = data.get("audio_event", {})
                        audio_base64 = audio_event.get("audio_base_64")
                        if audio_base64:
                            audio_data = base64.b64decode(audio_base64)
                            audio_responses_count += 1
                            last_audio_time = time.time()
                            logger.info(f"Received audio response #{audio_responses_count}: {len(audio_data)} bytes")
                            
                            # Play the audio
                            try:
                                await self._play_audio(audio_data)
                                logger.debug("Audio playback completed")
                            except Exception as e:
                                logger.error(f"Error playing audio: {e}")
                                traceback.print_exc()
                    
                    elif message_type == "agent_response":
                        # Log agent text response
                        response = data.get("agent_response_event", {}).get("agent_response")
                        if response:
                            print("\n" + "="*60)
                            print(f"🤖 {response}")
                            print("="*60 + "\n")
                    
                    elif message_type == "conversation_initiation_metadata":
                        # Log conversation ID
                        metadata = data.get("conversation_initiation_metadata_event", {})
                        conversation_id = metadata.get("conversation_id")
                        if conversation_id:
                            print("\n" + "="*60)
                            print(f"🔗 Connected to conversation: {conversation_id}")
                            print(f"🔊 Audio format: {metadata.get('agent_output_audio_format', 'unknown')}")
                            print("="*60 + "\n")
                            
                    elif message_type == "user_transcript":
                        # Log user transcript
                        transcript = data.get("user_transcription_event", {}).get("user_transcript")
                        if transcript:
                            print(f"👤 User: {transcript}")
                    
                    elif message_type == "vad_score":
                        # Just log VAD score in debug mode
                        score = data.get("vad_score_event", {}).get("vad_score")
                        if score:
                            logger.debug(f"VAD score: {score}")
                    
                    else:
                        # Log any other message types we receive
                        logger.debug(f"Received message of type: {message_type}")
                            
                except json.JSONDecodeError:
                    logger.error("Failed to parse WebSocket message")
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("WebSocket connection closed")
                    self.running = False
                    break
                
        except asyncio.CancelledError:
            logger.info("Listening task cancelled")
            self.running = False
        except Exception as e:
            logger.error(f"Error in WebSocket listening loop: {e}")
            traceback.print_exc()
            self.running = False
        
        logger.info(f"Listening stopped. Received {audio_responses_count} audio responses total.")
    
    async def reconnect_loop(self):
        """Keep trying to reconnect if connection is lost."""
        retry_count = 0
        # Use class attributes with defaults if not set
        max_retries = getattr(self, 'max_retries', 10)
        retry_delay = getattr(self, 'retry_delay', 5)
        
        while True:
            if not self.running:
                logger.info("Connection lost, attempting to reconnect...")
                success = await self.connect()
                
                if success:
                    # Reset retry count on successful connection
                    retry_count = 0
                    # Start listening again
                    asyncio.create_task(self.listen())
                else:
                    # Increment retry count
                    retry_count += 1
                    logger.warning(f"Reconnection attempt {retry_count}/{max_retries} failed")
                    
                    # If we've reached the max retries, wait longer
                    if retry_count >= max_retries:
                        logger.error(f"Failed to reconnect after {retry_count} attempts, waiting longer...")
                        await asyncio.sleep(retry_delay * 2)
                        retry_count = 0
                    else:
                        await asyncio.sleep(retry_delay)
            else:
                # Check every 5 seconds when connected
                await asyncio.sleep(5)
    
    async def close(self):
        """Close the connection and cleanup resources."""
        self.running = False
        
        if self.websocket:
            await self.websocket.close()
            logger.info("WebSocket connection closed")
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
        logger.info("Audio resources cleaned up")

async def main():
    # Load configuration 
    config = {
        "api_key": os.getenv("ELEVENLABS_API_KEY"),
        "agent_id": os.getenv("AGENT_ID"),
        "audio_device": os.getenv("AUDIO_DEVICE"),
        "sample_rate": 16000,
        "retry_delay": 5,
        "max_retries": 10
    }
    
    # Log API info (without showing the full key)
    if config["api_key"]:
        masked_key = config["api_key"][:4] + "..." + config["api_key"][-4:]
        logger.info(f"Using API key: {masked_key}")
    else:
        logger.error("No API key found in environment variables")
        
    if config["agent_id"]:
        logger.info(f"Using Agent ID: {config['agent_id']}")
    else:
        logger.error("No Agent ID found in environment variables")
        
    # Validate essential parameters
    if not config["api_key"] or not config["agent_id"]:
        logger.error("Missing required configuration. Please set ELEVENLABS_API_KEY and AGENT_ID environment variables.")
        return
    
    # Create ElevenLabs listener
    listener = ElevenLabsStreamingClient(
        api_key=config["api_key"],
        agent_id=config["agent_id"],
        audio_device=config["audio_device"],
        sample_rate=config["sample_rate"]
    )
    
    # Store retry parameters that weren't passed to the constructor
    listener.retry_delay = config["retry_delay"]
    listener.max_retries = config["max_retries"]
    
    try:
        # Display startup message
        print("\n" + "="*80)
        print("🔊 ELEVENLABS PI5 COMPANION LISTENER")
        print("="*80)
        print("This script is designed to run on a Raspberry Pi 5 and acts as a companion")
        print("listener for the voice assistant running on the Raspberry Pi 3.")
        print("")
        print("HOW IT WORKS:")
        print("1. The Raspberry Pi 3 handles wake word detection and sends audio to ElevenLabs")
        print("2. This companion listener connects to the same ElevenLabs conversation")
        print("3. Audio responses from ElevenLabs will play through this device's speakers")
        print("")
        print("IMPORTANT:")
        print("- Both devices must use the same ElevenLabs API key and Agent ID")
        print("- Start this listener BEFORE using the voice assistant on the Pi 3")
        print("- Ensure your audio output device is properly connected and configured")
        print("="*80 + "\n")
        
        # Main connection and operation loop
        while True:
            try:
                # Connect to ElevenLabs
                connected = await listener.connect()
                if connected:
                    # Start listening task
                    listen_task = asyncio.create_task(listener.listen())
                    # Start reconnection loop - works in the background
                    reconnect_task = asyncio.create_task(listener.reconnect_loop())
                    
                    # Wait for the listening task to complete (or be cancelled)
                    await listen_task
                    
                    # Cancel the reconnect task if the listen task exits
                    reconnect_task.cancel()
                    try:
                        await reconnect_task
                    except asyncio.CancelledError:
                        pass
                    
                # If we got here, either connection failed or the listening task exited
                # Wait a bit before trying again
                logger.info("Connection closed or failed, will retry in 5 seconds...")
                await asyncio.sleep(5)
                
            except asyncio.CancelledError:
                # Pass through CancelledError for clean program exit
                raise
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                traceback.print_exc()
                await asyncio.sleep(5)  # Wait before retrying
                
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except asyncio.CancelledError:
        logger.info("Program cancelled")
    finally:
        # Clean up resources
        await listener.close()
        logger.info("Program exited cleanly")

if __name__ == "__main__":
    # Set up asyncio loop
    asyncio.run(main())
