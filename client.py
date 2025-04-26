#!/usr/bin/env python3
"""
ElevenLabs Client Integration
Real implementation using the official ElevenLabs SDK with WebSocket fallback
"""

import os
import asyncio
import time
import json
import base64
import uuid
import websockets
import numpy as np
from typing import Dict, Any, Optional, List, Callable
import sys
import wave

from utils.logging import get_logger

# Import ElevenLabs SDK components with better error handling
ELEVENLABS_SDK_AVAILABLE = False
try:
    # First try a reduced import that just tests API access without the full type system
    from elevenlabs import generate, save, voices, Models, Voice

    # Now try the conversational components separately
    try:
        # Wrap this in its own try block to keep the base functionality
        from elevenlabs.client import ElevenLabs
        from elevenlabs.conversational_ai.conversation import Conversation, AudioInterface
        ELEVENLABS_SDK_AVAILABLE = True
    except (ImportError, NameError) as e:
        logger = get_logger("elevenlabs_client")
        logger.error(f"Failed to import ElevenLabs Conversational AI components: {e}")
        logger.info("Basic ElevenLabs functionality is available but not the Conversational AI")
        ELEVENLABS_SDK_AVAILABLE = False
except (ImportError, NameError) as e:
    logger = get_logger("elevenlabs_client")
    logger.error(f"Failed to import ElevenLabs SDK: {e}")
    ELEVENLABS_SDK_AVAILABLE = False
    
logger = get_logger("elevenlabs_client")

class MockAudioInterface:
    """Fallback implementation when full AudioInterface is not available."""
    
    def __init__(self):
        self.input_queue = asyncio.Queue()
        self.output_callbacks = []
        
    def register_output_callback(self, callback):
        self.output_callbacks.append(callback)
        
    async def get_audio(self):
        try:
            return await self.input_queue.get()
        except Exception as e:
            logger.error(f"Error getting audio: {e}")
            return None
    
    async def add_audio(self, audio_data):
        await self.input_queue.put(audio_data)
        
    async def play_audio(self, audio_data):
        for callback in self.output_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(audio_data)
                else:
                    callback(audio_data)
            except Exception as e:
                logger.error(f"Error in audio output callback: {e}")

# Use conditional imports to define the class
if ELEVENLABS_SDK_AVAILABLE:
    class CustomAudioInterface(AudioInterface):
        """Custom AudioInterface implementation that works with our speech buffer.
        Instead of using microphone input directly, it receives audio from our speech buffer.
        """
        
        def __init__(self):
            self.input_queue = asyncio.Queue()
            self.output_callbacks = []
            
        def register_output_callback(self, callback):
            """Register a callback to receive audio output from ElevenLabs."""
            self.output_callbacks.append(callback)
            
        async def get_audio(self) -> Optional[bytes]:
            """Get audio from the input queue (filled by speech buffer)."""
            try:
                audio_data = await self.input_queue.get()
                return audio_data
            except Exception as e:
                logger.error(f"Error getting audio: {e}")
                return None
        
        async def add_audio(self, audio_data: bytes):
            """Add audio to the input queue for processing by ElevenLabs."""
            await self.input_queue.put(audio_data)
            
        async def play_audio(self, audio_data: bytes):
            """Play audio through registered callbacks instead of directly."""
            for callback in self.output_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(audio_data)
                    else:
                        callback(audio_data)
                except Exception as e:
                    logger.error(f"Error in audio output callback: {e}")
else:
    # Fallback if the SDK isn't available
    CustomAudioInterface = MockAudioInterface

class WebSocketConversation:
    """Direct implementation of ElevenLabs conversation using WebSockets.
    
    This class provides a fallback when the SDK has issues, by directly implementing
    the WebSocket protocol described in the ElevenLabs documentation.
    """
    
    def __init__(self, api_key, agent_id, config=None):
        self.api_key = api_key
        self.agent_id = agent_id
        self.config = config or {}
        self.websocket = None
        self.conversation_id = None
        self.running = False
        self.audio_queue = asyncio.Queue()
        self.response_callbacks = []
        self.audio_callbacks = []
        self.transcript_callbacks = []
        
    def register_response_callback(self, callback):
        """Register a callback for agent responses."""
        self.response_callbacks.append(callback)
        
    def register_audio_callback(self, callback):
        """Register a callback for audio responses."""
        self.audio_callbacks.append(callback)
        
    def register_transcript_callback(self, callback):
        """Register a callback for user transcripts."""
        self.transcript_callbacks.append(callback)
        
    async def start_session(self):
        """Start a WebSocket conversation session."""
        try:
            # Connect to the ElevenLabs WebSocket API
            headers = {}
            if self.api_key:
                headers["xi-api-key"] = self.api_key
                
            # Build the URL with the agent ID
            url = f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={self.agent_id}"
            
            logger.info(f"Connecting to ElevenLabs WebSocket: {url}")
            self.websocket = await websockets.connect(url, extra_headers=headers)
            self.running = True
            
            # Send initial configuration
            await self._send_initialization()
            
            # Start the WebSocket receive loop
            asyncio.create_task(self._receive_loop())
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to ElevenLabs WebSocket: {e}")
            return False
    
    async def _send_initialization(self):
        """Send initialization data to the WebSocket."""
        if not self.websocket:
            return
        # Create initialization message with silence timeout override
        init_message = {
            "type": "conversation_initiation_client_data",
            "conversation_config_override": {
                "silence_timeout_ms": 5000  # 5 seconds in milliseconds
            }
        }
        
        # Send the initialization message
        await self.websocket.send(json.dumps(init_message))
       
    async def _receive_loop(self):
        """Continuously receive and process WebSocket messages."""
        if not self.websocket:
            return
            
        # Create a transcript file for debugging
        transcript_path = os.path.join("logs", f"conversation_{time.strftime('%Y%m%d_%H%M%S')}.txt")
        os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
        
        with open(transcript_path, "w") as transcript_file:
            transcript_file.write(f"=== ElevenLabs Conversation Transcript ===\n")
            transcript_file.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        try:
            while self.running:
                # Receive the next message
                message = await self.websocket.recv()
                
                # Log all received messages for debugging
                logger.debug(f"RECV: {message[:200]}{'...' if len(message) > 200 else ''}")
                
                # Parse as JSON
                try:
                    data = json.loads(message)
                    message_type = data.get("type")
                    
                    # Log all non-ping messages at INFO level
                    if message_type != "ping":
                        logger.info(f"Received message type: {message_type}")
                    
                    # Append to transcript file for debugging
                    with open(transcript_path, "a") as transcript_file:
                        if message_type == "conversation_initiation_metadata":
                            metadata = data.get("conversation_initiation_metadata_event", {})
                            transcript_file.write(f"CONNECTION ESTABLISHED - ID: {metadata.get('conversation_id')}\n\n")
                        elif message_type == "user_transcript":
                            transcript = data.get("user_transcription_event", {}).get("user_transcript", "")
                            transcript_file.write(f"USER: {transcript}\n\n")
                        elif message_type == "agent_response":
                            response = data.get("agent_response_event", {}).get("agent_response", "")
                            transcript_file.write(f"AGENT: {response}\n\n")
                        elif message_type == "audio":
                            transcript_file.write(f"AUDIO: [Received {len(data.get('audio_event', {}).get('audio_base_64', ''))} chars of base64 audio]\n\n")
                    
                    # Handle different message types
                    if message_type == "conversation_initiation_metadata":
                        metadata = data.get("conversation_initiation_metadata_event", {})
                        self.conversation_id = metadata.get("conversation_id")
                        logger.info(f"Conversation started with ID: {self.conversation_id}")
                        
                    elif message_type == "user_transcript":
                        transcript = data.get("user_transcription_event", {}).get("user_transcript")
                        if transcript:
                            logger.info(f"User transcript: {transcript}")
                            # Notify transcript callbacks
                            for callback in self.transcript_callbacks:
                                if asyncio.iscoroutinefunction(callback):
                                    asyncio.create_task(callback(transcript))
                                else:
                                    callback(transcript)
                                    
                    elif message_type == "agent_response":
                        response = data.get("agent_response_event", {}).get("agent_response")
                        if response:
                            logger.info(f"Agent response: {response}")
                            # Notify response callbacks
                            for callback in self.response_callbacks:
                                if asyncio.iscoroutinefunction(callback):
                                    asyncio.create_task(callback(response))
                                else:
                                    callback(response)
                                    
                    elif message_type == "audio":
                        audio_event = data.get("audio_event", {})
                        audio_base64 = audio_event.get("audio_base_64")
                        if audio_base64:
                            audio_data = base64.b64decode(audio_base64)
                            logger.info(f"Received audio: {len(audio_data)} bytes")
                            # Notify audio callbacks immediately without saving
                            for callback in self.audio_callbacks:
                                try:
                                    if asyncio.iscoroutinefunction(callback):
                                        asyncio.create_task(callback(audio_data))
                                    else:
                                        callback(audio_data)
                                except Exception as e:
                                    logger.error(f"Error in audio output callback: {e}")
                    elif message_type == "ping":
                        event_id = data.get("ping_event", {}).get("event_id")
                        # Send pong response
                        if event_id:
                            await self.websocket.send(json.dumps({
                                "type": "pong",
                                "event_id": event_id
                            }))
                    
                    # Add handlers for other message types as needed
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse WebSocket message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
            self.running = False
            
        except Exception as e:
            logger.error(f"Error in WebSocket receive loop: {e}")
            self.running = False
        
        # Log end of conversation
        with open(transcript_path, "a") as transcript_file:
            transcript_file.write(f"\n=== End of conversation ===\nEnded at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    async def send_audio(self, audio_data):
        """Send audio data to the conversation."""
        if not self.websocket or not self.running:
            logger.warning("Cannot send audio: WebSocket not connected or not running")
            return False
            
        try:
            # Log original audio data details
            logger.debug(f"Original audio data: type={type(audio_data)}, " + 
                         (f"length={len(audio_data)} bytes" if isinstance(audio_data, bytes) else 
                          f"shape={audio_data.shape}, dtype={audio_data.dtype}" if hasattr(audio_data, 'shape') else 
                          "unknown format"))
            
            # Convert to base64 if it's not already
            if isinstance(audio_data, bytes):
                audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                logger.debug(f"Converted {len(audio_data)} bytes to base64 ({len(audio_base64)} chars)")
            else:
                # Convert numpy array to bytes
                if isinstance(audio_data, np.ndarray):
                    # Ensure int16 format for audio
                    if audio_data.dtype != np.int16:
                        audio_data = (audio_data * 32767).astype(np.int16)
                        logger.debug(f"Converted numpy array to int16 dtype")
                    
                    audio_bytes = audio_data.tobytes()
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                    logger.debug(f"Converted numpy array of shape {audio_data.shape} to base64 ({len(audio_base64)} chars)")
                else:
                    logger.error(f"Unsupported audio data type: {type(audio_data)}")
                    return False
            
            # Send the audio data as a chunk
            message = json.dumps({
                "user_audio_chunk": audio_base64
            })
            logger.debug(f"Sending WebSocket message with {len(message)} chars")
            await self.websocket.send(message)
            logger.debug("Audio data sent successfully to ElevenLabs")
            
            return True
        except Exception as e:
            logger.error(f"Error sending audio to ElevenLabs: {e}")
            return False
    
    async def end_session(self):
        """End the WebSocket conversation session."""
        self.running = False
        
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info(f"Closed WebSocket connection, conversation ID: {self.conversation_id}")
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {e}")
                
        return self.conversation_id

class ElevenLabsClient:
    """Client for ElevenLabs Conversational AI."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ElevenLabs client.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        
        # API credentials
        self.api_key = config.get('api_key') or os.getenv("ELEVENLABS_API_KEY")
        self.agent_id = config.get('agent_id') or os.getenv("AGENT_ID")
        
        # Choose implementation strategy
        if not ELEVENLABS_SDK_AVAILABLE:
            logger.warning("ElevenLabs SDK not available, using direct WebSocket implementation")
            self.use_sdk = False
        else:
            logger.info("Using ElevenLabs SDK for conversation")
            self.use_sdk = True
            
        # Conversation state
        self.session_active = False
        self.audio_interface = CustomAudioInterface()
        self.client = None
        self.conversation = None
        self.conversation_id = None
        self.websocket_conversation = None
        
        # Response callbacks
        self.response_callbacks = []
        
        # Stats and metrics
        self.session_start_time = None
        self.last_request_time = None
        self.total_requests = 0
        self.total_responses = 0
        
        # Check if we have valid credentials
        if not self.api_key:
            logger.warning("No ElevenLabs API key provided, functionality will be limited")
            
        if not self.agent_id:
            logger.warning("No ElevenLabs Agent ID provided, functionality will be limited")
        
        logger.info(f"ElevenLabs client initialized (enabled: {self.enabled}, using SDK: {self.use_sdk})")
        
    async def initialize(self):
        """Initialize the ElevenLabs client."""
        if not self.enabled:
            logger.warning("ElevenLabs integration is disabled")
            return False
            
        try:
            if self.use_sdk:
                # SDK initialization
                self.client = ElevenLabs(api_key=self.api_key)
                self.audio_interface.register_output_callback(self._handle_audio_output)
                logger.info("ElevenLabs SDK client initialized successfully")
            else:
                # No special initialization needed for WebSocket implementation
                logger.info("WebSocket implementation ready")
                
            return True
        except Exception as e:
            logger.error(f"Failed to initialize ElevenLabs client: {e}")
            self.use_sdk = False  # Fall back to WebSocket implementation
            logger.info("Falling back to WebSocket implementation")
            return True  # Still return true to not block the pipeline
            
    async def start_session(self):
        """Start a new conversation session with ElevenLabs.
        
        Returns:
            bool: True if session started successfully
        """
        if not self.enabled:
            logger.info("ElevenLabs integration is disabled")
            self.session_active = True
            self.session_start_time = time.time()
            await self._simulate_response("Hello, I'm your voice assistant. How can I help you today?")
            return True
            
        # End any existing session
        if self.session_active:
            await self.end_session()
            
        try:
            logger.info("Starting new ElevenLabs conversation session")
            self.session_start_time = time.time()
            
            if self.use_sdk:
                # SDK implementation
                # Define callbacks
                def on_agent_response(response):
                    logger.info(f"Agent: {response}")
                    self._notify_response_callbacks(response)
                    
                def on_transcript(transcript):
                    logger.info(f"User: {transcript}")
                    
                # Create conversation instance
                self.conversation = Conversation(
                    # API client and agent ID
                    self.client,
                    self.agent_id,
                    
                    # Assume auth is required when API_KEY is set
                    requires_auth=bool(self.api_key),
                    
                    # Use our custom audio interface
                    audio_interface=self.audio_interface,
                    
                    # Callbacks
                    callback_agent_response=on_agent_response,
                    callback_user_transcript=on_transcript,
                    
                    # Additional configuration from config
                    conversation_config_override=self.config.get('conversation_config', {}),
                )
                
                # Start the session
                self.conversation.start_session()
                self.session_active = True
                
                logger.info("ElevenLabs SDK conversation session started")
            else:
                # WebSocket implementation
                self.websocket_conversation = WebSocketConversation(
                    self.api_key, 
                    self.agent_id,
                    self.config.get('conversation_config', {})
                )
                
                # Register callbacks
                self.websocket_conversation.register_response_callback(self._handle_response)
                self.websocket_conversation.register_audio_callback(self._handle_audio_output)
                
                # Start the session
                success = await self.websocket_conversation.start_session()
                if success:
                    self.session_active = True
                    logger.info("ElevenLabs WebSocket conversation session started")
                else:
                    logger.error("Failed to start WebSocket conversation")
                    await self._simulate_response("I'm sorry, but I couldn't connect to the voice service. I'll respond in text only.")
                    self.session_active = True
            
            return True
        except Exception as e:
            logger.error(f"Failed to start ElevenLabs session: {e}")
            logger.info("Falling back to simulated responses")
            self.session_active = True
            await self._simulate_response("Hello, I'm your voice assistant running in fallback mode.")
            return True
    
    async def end_session(self):
        """End the current conversation session."""
        if not self.session_active:
            return
            
        try:
            logger.info("Ending ElevenLabs conversation session")
            
            if self.use_sdk and self.conversation:
                # End SDK session
                self.conversation_id = self.conversation.end_session()
            elif self.websocket_conversation:
                # End WebSocket session
                self.conversation_id = await self.websocket_conversation.end_session()
                self.websocket_conversation = None
            
            # Calculate session duration
            if self.session_start_time:
                duration = time.time() - self.session_start_time
                logger.info(f"Session ended. Duration: {duration:.1f}s, Conversation ID: {self.conversation_id}")
                
            # Reset state
            self.session_active = False
            self.conversation = None
        except Exception as e:
            logger.error(f"Error ending ElevenLabs session: {e}")
            self.session_active = False
    
    async def process_speech(self, speech_data):
        """Process speech data and get AI response.
        
        Args:
            speech_data: Raw audio data to send to ElevenLabs
            
        Returns:
            bool: True if processing was successful
        """
        if not self.session_active:
            logger.warning("Cannot process speech: session not active")
            return False
            
        try:
            # Log details about the input audio
            if isinstance(speech_data, bytes):
                logger.info(f"Processing {len(speech_data)} bytes of raw audio data")
            elif isinstance(speech_data, np.ndarray):
                logger.info(f"Processing numpy array of shape {speech_data.shape} and dtype {speech_data.dtype}")
            else:
                logger.info(f"Processing speech data of type {type(speech_data)}")
                
            self.last_request_time = time.time()
            self.total_requests += 1
            
            # Convert to the right format if needed
            processed_speech_data = speech_data
            
            # If it's a numpy array, convert to bytes
            if isinstance(processed_speech_data, np.ndarray):
                # Ensure it's int16 format
                if processed_speech_data.dtype != np.int16:
                    processed_speech_data = (processed_speech_data * 32767).astype(np.int16)
                
                # Convert to bytes
                processed_speech_data = processed_speech_data.tobytes()
                logger.info(f"Converted numpy array to {len(processed_speech_data)} bytes")
                
            # Now processed_speech_data should be in bytes format
            if not isinstance(processed_speech_data, bytes):
                logger.error(f"Unexpected data type after processing: {type(processed_speech_data)}")
                return False
                
            # Log the size of the processed data
            logger.info(f"Sending {len(processed_speech_data)} bytes to ElevenLabs")
            
            if self.use_sdk and self.conversation:
                # Send using SDK
                logger.info("Sending audio via SDK")
                await self.audio_interface.add_audio(processed_speech_data)
            elif self.websocket_conversation:
                try:
                    # Send using WebSocket
                    logger.info("Sending audio via WebSocket")
                    success = await self.websocket_conversation.send_audio(processed_speech_data)
                    
                    # If sending failed, create a fallback response
                    if not success:
                        logger.warning("Failed to send audio through WebSocket, using fallback response")
                        await self._simulate_response("I couldn't process your audio. Could you please try again?")
                        
                    # Check if WebSocket is still connected
                    if not self.websocket_conversation.running:
                        logger.warning("WebSocket connection closed, falling back to simulated responses")
                        await self._simulate_response("I'm sorry, but my voice service connection was lost. I'm operating in text-only mode now.")
                        
                        # If this agent requires a specific connection protocol, we may need to reinitialize
                        # Rather than repeatedly failing, use simulated responses
                        if not hasattr(self, 'websocket_failure_count'):
                            self.websocket_failure_count = 0
                        self.websocket_failure_count += 1
                        
                        # After 3 failures, stop trying to use the WebSocket
                        if self.websocket_failure_count >= 3:
                            logger.error("Too many WebSocket failures, switching to simulation mode permanently")
                            self.websocket_conversation = None
                except Exception as e:
                    logger.error(f"Error sending audio through WebSocket: {e}")
                    await self._simulate_response("I encountered a technical issue. Please try again in a moment.")
            else:
                # Fallback to mock response
                await self._simulate_response(self._generate_mock_response())
            
            return True
        except Exception as e:
            logger.error(f"Error processing speech: {e}")
            # Fall back to mock response on error
            await self._simulate_response("I'm sorry, I encountered an error processing your speech.")
            return False
    
    async def _handle_audio_output(self, audio_data):
        """Handle audio data from ElevenLabs.
        
        Args:
            audio_data: Audio data in raw PCM format
        """
        try:
            # Convert any base64 encoded data to bytes if needed
            if isinstance(audio_data, str):
                decoded_data = base64.b64decode(audio_data)
                logger.debug(f"Decoded {len(audio_data)} chars of base64 to {len(decoded_data)} bytes")
                audio_data = decoded_data
            
            # Save audio to file for logging/debugging
            if self.config.get('save_audio_responses', True):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"logs/audio_responses/response_{timestamp}.wav"
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                
                # Convert raw PCM to WAV for easier playback
                with wave.open(filename, 'wb') as wf:
                    wf.setnchannels(1)  # Mono
                    wf.setsampwidth(2)  # 16-bit
                    wf.setframerate(16000)  # 16 kHz
                    wf.writeframes(audio_data)
                    
                logger.info(f"Saved audio response to {filename}")
            
            # Forward audio to GStreamer for Pi5 playback
            try:
                # Import here to avoid circular imports
                from pi3_gstreamer_sender import send_audio_to_pipeline
                
                # Forward directly to GStreamer
                if send_audio_to_pipeline(audio_data):
                    logger.info("Forwarded audio to GStreamer pipeline for Pi5")
                else:
                    logger.warning("Failed to forward audio to GStreamer pipeline")
            except ImportError:
                logger.warning("GStreamer sender not available, audio not forwarded to Pi5")
            except Exception as e:
                logger.error(f"Error forwarding audio to GStreamer: {e}")
            
            # Play audio locally if configured (useful for testing)
            if self.config.get('play_audio_locally', False):
                if self.audio_interface:
                    await self.audio_interface.play_audio(audio_data)
                    logger.info("Played audio response locally")
                else:
                    logger.warning("No audio interface available for local playback")
            
            logger.info("Received audio response from ElevenLabs")
            
        except Exception as e:
            logger.error(f"Error handling audio output: {e}")
            # Continue operation despite errors in audio handling
    
    async def _handle_response(self, text):
        """Handle text response from ElevenLabs WebSocket."""
        logger.info(f"Received response: {text}")
        
        # Print to console in a visible format
        print("\n" + "="*60)
        print("🤖 ELEVENLABS RESPONSE:")
        print(f"{text}")
        print("="*60 + "\n")
        
        self._notify_response_callbacks(text)
    
    def _notify_response_callbacks(self, text):
        """Notify all registered callbacks with the text response."""
        for callback in self.response_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(text))
                else:
                    callback(text)
            except Exception as e:
                logger.error(f"Error in response callback: {e}")
    
    def register_response_callback(self, callback: Callable):
        """Register a callback to receive responses from ElevenLabs.
        
        Args:
            callback: Function to call with response text
        """
        self.response_callbacks.append(callback)
        
    async def cleanup(self):
        """Clean up resources."""
        await self.end_session()
        logger.info("ElevenLabs client cleaned up")
        
    # Mock functionality for fallback
    def _generate_mock_response(self):
        """Generate a mock response."""
        # Simple set of possible responses
        responses = [
            "I'm running in fallback mode since the ElevenLabs service isn't fully connected.",
            "I heard you, and I'm generating a mock response.",
            "Your request was received. In a real deployment, ElevenLabs would provide a fuller response.",
            "The wake word detection is working! This is a fallback response.",
            "Thanks for your input. This is a simulated response while ElevenLabs is being configured."
        ]
        import random
        return random.choice(responses)
    
    async def _simulate_response(self, text):
        """Simulate a response from the assistant."""
        logger.info(f"Simulated response: {text}")
        self.total_responses += 1
        
        # Notify all registered callbacks
        self._notify_response_callbacks(text)
    
    async def _direct_audio_capture(self):
        """Capture audio directly and send to ElevenLabs."""
        try:
            logger.info("Direct audio capture started")
            
            # Listen for a longer duration
            capture_duration = 60.0  # seconds (increased from 30 to 60)
            capture_start = time.time()
            
            # For monitoring silence
            silence_start = None
            min_audio_chunks = 20  # Increased buffer size for better transcription
            audio_buffer = []  # Add audio buffering
            
            while self.direct_audio_capture and time.time() - capture_start < capture_duration:
                # Process audio if we have enough
                if len(self.audio_chunks) >= min_audio_chunks:
                    # Combine chunks with overlap to prevent cutting
                    combined_audio = b''.join(self.audio_chunks[-min_audio_chunks:]) if isinstance(self.audio_chunks[0], bytes) else np.concatenate(self.audio_chunks[-min_audio_chunks:])
                    
                    # Check audio level for silence detection
                    if isinstance(combined_audio, bytes):
                        audio_data = np.frombuffer(combined_audio, dtype=np.int16)
                    else:
                        audio_data = combined_audio
                        
                    audio_level = np.sqrt(np.mean(np.square(audio_data.astype(np.float32))))
                    
                    # Update silence tracking with more granular control
                    current_time = time.time()
                    if audio_level > self.silence_threshold:
                        self.last_speech_time = current_time
                        silence_start = None
                        logger.debug(f"Speech detected, audio level: {audio_level:.2f}")
                    elif silence_start is None:
                        silence_start = current_time
                        logger.debug(f"Silence started, audio level: {audio_level:.2f}")
                    elif current_time - silence_start > self.silence_duration:
                        logger.info(f"Ending conversation due to {self.silence_duration}s of silence")
                        break
                    
                    # Keep a sliding window of chunks instead of clearing
                    self.audio_chunks = self.audio_chunks[-min_audio_chunks:]
                    
                    # Send to ElevenLabs with retry mechanism
                    max_retries = 3
                    for retry in range(max_retries):
                        try:
                            logger.info(f"Sending {len(combined_audio)} bytes of audio to ElevenLabs")
                            await self.elevenlabs_client.process_speech(combined_audio)
                            break
                        except Exception as e:
                            if retry == max_retries - 1:
                                logger.error(f"Failed to send audio after {max_retries} attempts: {e}")
                            else:
                                logger.warning(f"Retry {retry + 1}/{max_retries} sending audio: {e}")
                                await asyncio.sleep(0.5)
                
                # Small sleep to avoid CPU spinning but be more responsive
                await asyncio.sleep(0.05)
            
            # Process any remaining audio
            if len(self.audio_chunks) > 0:
                combined_audio = b''.join(self.audio_chunks) if isinstance(self.audio_chunks[0], bytes) else np.concatenate(self.audio_chunks)
                logger.info(f"Sending final {len(combined_audio)} bytes of audio to ElevenLabs")
                await self.elevenlabs_client.process_speech(combined_audio)
            
            # End direct capture
            self.direct_audio_capture = False
            self.in_conversation = False
            logger.info("Direct audio capture completed")
            
        except Exception as e:
            logger.error(f"Error in direct audio capture: {e}")
            self.direct_audio_capture = False
            self.in_conversation = False