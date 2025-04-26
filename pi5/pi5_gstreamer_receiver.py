#!/usr/bin/env python3
import os
import sys
import subprocess
import threading
import logging
import time
import json
import signal
import redis

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("pi5_gstreamer")

# Initialize Redis for signaling
PI3_IP = os.environ.get('PI3_IP', '100.72.160.86')  # Default to Pi3 Tailscale IP
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', None)  # Get password from environment if set

try:
    # Connect to Redis with password if provided
    if REDIS_PASSWORD:
        redis_client = redis.Redis(host=PI3_IP, port=6379, db=0, password=REDIS_PASSWORD)
        logger.info(f"Connecting to Redis server on {PI3_IP} with authentication")
    else:
        redis_client = redis.Redis(host=PI3_IP, port=6379, db=0)
        logger.info(f"Connecting to Redis server on {PI3_IP} without authentication")
    
    redis_client.ping()  # Test connection
    logger.info(f"Successfully connected to Redis server on {PI3_IP}")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")
    redis_client = None

# Network settings - get these from Redis
RTP_PORT = 5000  # Default value

def get_network_config():
    """Get network configuration from Redis"""
    global RTP_PORT
    
    if redis_client:
        config_json = redis_client.get('audio_stream:pi3_config')
        if config_json:
            try:
                config = json.loads(config_json)
                RTP_PORT = config.get('rtp_port', RTP_PORT)
                logger.info(f"Got network config from Redis: RTP port {RTP_PORT}")
                return True
            except Exception as e:
                logger.error(f"Error parsing network config from Redis: {e}")
    
    logger.warning(f"Using default network config: RTP port {RTP_PORT}")
    return False

# Track if pipeline is running
pipeline_process = None

def start_reception_pipeline():
    """Start the GStreamer pipeline to receive audio from Pi3"""
    global pipeline_process
    
    if pipeline_process:
        logger.warning("Pipeline already running, stopping it first")
        stop_reception_pipeline()
    
    # Update configuration from Redis
    get_network_config()
    
    # Build GStreamer pipeline
    # This pipeline receives RTP Opus audio and plays it through the default audio device
    pipeline_cmd = f"""
    gst-launch-1.0 -v udpsrc port={RTP_PORT} caps="application/x-rtp,media=audio,clock-rate=48000,encoding-name=OPUS" ! 
    rtpopusdepay ! opusdec ! 
    audioconvert ! audioresample ! 
    autoaudiosink
    """
    
    try:
        # Start pipeline process
        pipeline_process = subprocess.Popen(
            pipeline_cmd.split(),
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE
        )
        logger.info(f"Started GStreamer reception pipeline on port {RTP_PORT}")
        
        # Start a thread to monitor stderr for errors
        def monitor_stderr():
            for line in iter(pipeline_process.stderr.readline, b''):
                line = line.decode('utf-8').strip()
                if "ERROR" in line or "WARN" in line:
                    logger.warning(f"GStreamer: {line}")
        
        stderr_thread = threading.Thread(target=monitor_stderr, daemon=True)
        stderr_thread.daemon = True
        stderr_thread.start()
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to start GStreamer reception pipeline: {e}")
        return False

def stop_reception_pipeline():
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

def monitor_redis_for_updates():
    """Monitor Redis for configuration updates"""
    if not redis_client:
        logger.warning("Redis not available, cannot monitor for updates")
        return
    
    try:
        pubsub = redis_client.pubsub()
        pubsub.subscribe('audio_stream:config_update')
        
        logger.info("Monitoring Redis for configuration updates")
        
        for message in pubsub.listen():
            if message['type'] == 'message':
                logger.info("Received configuration update notification")
                
                # Update configuration
                if get_network_config():
                    # Restart pipeline with new configuration
                    if pipeline_process:
                        logger.info("Restarting pipeline with new configuration")
                        stop_reception_pipeline()
                        start_reception_pipeline()
    
    except Exception as e:
        logger.error(f"Error monitoring Redis for updates: {e}")

if __name__ == "__main__":
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        logger.info("Stopping reception pipeline...")
        stop_reception_pipeline()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start Redis monitoring thread
    redis_thread = threading.Thread(target=monitor_redis_for_updates, daemon=True)
    redis_thread.start()
    
    # Simple command-line interface
    if len(sys.argv) > 1:
        if sys.argv[1] == "start":
            start_reception_pipeline()
            logger.info("Press Ctrl+C to stop")
            
            # Keep the script running
            while True:
                time.sleep(1)
                
                # Restart pipeline if it died
                if pipeline_process and pipeline_process.poll() is not None:
                    logger.warning("Pipeline died, restarting...")
                    start_reception_pipeline()
                
        elif sys.argv[1] == "stop":
            stop_reception_pipeline()
    
    else:
        print("Usage:")
        print("  python pi5_gstreamer_receiver.py start - Start reception pipeline")
        print("  python pi5_gstreamer_receiver.py stop  - Stop reception pipeline")
        
        # Start by default
        start_reception_pipeline()
        logger.info("Press Ctrl+C to stop")
        
        # Keep the script running
        try:
            while True:
                time.sleep(1)
                
                # Restart pipeline if it died
                if pipeline_process and pipeline_process.poll() is not None:
                    logger.warning("Pipeline died, restarting...")
                    start_reception_pipeline()
        except KeyboardInterrupt:
            stop_reception_pipeline() 