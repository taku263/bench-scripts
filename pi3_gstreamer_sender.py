#!/usr/bin/env python3
import os
import sys
import subprocess
import threading
import tempfile
import logging
import time
import json
import wave
import math
import struct
import redis

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pi3_gstreamer")

# Initialize Redis for signaling
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    redis_client.ping()  # Test connection
    logger.info("Connected to Redis server")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    redis_client = None

# Network settings - use Redis to share these with Pi5
PI5_IP = os.environ.get('PI5_IP', '100.81.91.108')  # Default to Pi5 Tailscale IP
RTP_PORT = 5000

# Set up network configuration 
def update_network_config():
    """Update network configuration in Redis for the Pi5 to discover"""
    if redis_client:
        redis_client.set('audio_stream:pi3_config', json.dumps({
            'rtp_port': RTP_PORT,
            'timestamp': time.time()
        }))
        logger.info(f"Updated network config in Redis: RTP port {RTP_PORT}")

# Track if pipeline is running
pipeline_process = None

def start_streaming_pipeline():
    """Start the GStreamer pipeline to stream audio to Pi5"""
    global pipeline_process
    
    if pipeline_process:
        logger.warning("Pipeline already running, stopping it first")
        stop_streaming_pipeline()
    
    # Build GStreamer pipeline
    # This pipeline converts raw PCM audio to Opus, which is ideal for network streaming
    pipeline_cmd = f"""
    gst-launch-1.0 -v fdsrc ! 
    audio/x-raw,format=S16LE,channels=1,rate=16000 ! 
    audioconvert ! audioresample ! 
    opusenc bitrate=64000 ! rtpopuspay ! 
    queue ! udpsink host={PI5_IP} port={RTP_PORT}
    """
    
    try:
        # Start pipeline process with pipe for input
        pipeline_process = subprocess.Popen(
            pipeline_cmd.split(),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE,
            bufsize=0  # Unbuffered
        )
        logger.info(f"Started GStreamer pipeline streaming to {PI5_IP}:{RTP_PORT}")
        
        # Update Redis with our configuration
        update_network_config()
        
        # Start a thread to monitor stderr for errors
        def monitor_stderr():
            for line in iter(pipeline_process.stderr.readline, b''):
                line = line.decode('utf-8').strip()
                if "ERROR" in line or "WARN" in line:
                    logger.warning(f"GStreamer: {line}")
        
        stderr_thread = threading.Thread(target=monitor_stderr, daemon=True)
        stderr_thread.start()
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to start GStreamer pipeline: {e}")
        return False

def stop_streaming_pipeline():
    """Stop the GStreamer pipeline"""
    global pipeline_process
    
    if pipeline_process:
        try:
            # Send SIGTERM to the process group
            pipeline_process.terminate()
            pipeline_process.wait(timeout=3)
            logger.info("Stopped GStreamer pipeline")
        except Exception as e:
            logger.error(f"Error stopping pipeline: {e}")
            try:
                pipeline_process.kill()
                logger.info("Forcefully killed GStreamer pipeline")
            except:
                pass
        
        pipeline_process = None

def send_audio_to_pipeline(audio_data):
    """Send audio data to the GStreamer pipeline
    
    Args:
        audio_data: Raw PCM audio data (bytes)
    """
    global pipeline_process
    
    if not pipeline_process or pipeline_process.poll() is not None:
        logger.warning("Pipeline not running, starting it now")
        if not start_streaming_pipeline():
            logger.error("Failed to start pipeline, cannot send audio")
            return False
    
    try:
        # Send raw audio data to the pipeline
        pipeline_process.stdin.write(audio_data)
        pipeline_process.stdin.flush()
        logger.debug(f"Sent {len(audio_data)} bytes to GStreamer pipeline")
        return True
    except Exception as e:
        logger.error(f"Error sending audio to pipeline: {e}")
        stop_streaming_pipeline()  # Reset the pipeline
        return False

def convert_wav_to_pcm(audio_data):
    """Convert WAV audio data to raw PCM
    
    Args:
        audio_data: WAV audio data (bytes)
        
    Returns:
        PCM audio data (bytes)
    """
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp:
            temp_path = temp.name
            temp.write(audio_data)
        
        # Use wave module to extract PCM data
        with wave.open(temp_path, 'rb') as wf:
            # Verify format
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            
            logger.debug(f"WAV format: channels={channels}, width={sample_width}, rate={framerate}")
            
            # Read frames
            pcm_data = wf.readframes(wf.getnframes())
        
        # Clean up
        os.unlink(temp_path)
        
        return pcm_data
    except Exception as e:
        logger.error(f"Error converting WAV to PCM: {e}")
        return audio_data  # Return original as fallback

if __name__ == "__main__":
    # Start by updating network config
    update_network_config()
    
    # Simple command-line interface for testing
    if len(sys.argv) > 1:
        if sys.argv[1] == "start":
            start_streaming_pipeline()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                stop_streaming_pipeline()
                
        elif sys.argv[1] == "stop":
            stop_streaming_pipeline()
            
        elif sys.argv[1] == "test":
            # Generate 2 seconds of test audio (sine wave)
            rate = 16000
            duration = 2  # seconds
            frequency = 440  # Hz (A4 note)
            
            # Generate sine wave
            samples = []
            for i in range(int(rate * duration)):
                sample = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * i / rate))
                samples.append(sample)
            
            # Convert to bytes
            audio_data = struct.pack('<%dh' % len(samples), *samples)
            
            # Start pipeline and send audio
            start_streaming_pipeline()
            send_audio_to_pipeline(audio_data)
            
            # Keep running for a bit then stop
            time.sleep(3)
            stop_streaming_pipeline()
    
    else:
        print("Usage:")
        print("  python pi3_gstreamer_sender.py start - Start streaming pipeline")
        print("  python pi3_gstreamer_sender.py stop  - Stop streaming pipeline")
        print("  python pi3_gstreamer_sender.py test  - Send test audio") 