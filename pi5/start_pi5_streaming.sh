#!/bin/bash
echo "Making scripts executable"
chmod +x pi5_gstreamer_receiver.py
echo "Setting Pi3 IP address (Tailscale IP)"
export PI3_IP="100.72.160.86"  # Tailscale IP address for Pi3

# Uncomment and set this if you've configured a Redis password on Pi3
# export REDIS_PASSWORD="your_strong_password"

echo "Starting audio receiver on Pi5"
./pi5_gstreamer_receiver.py start &
echo "Audio receiver started"
echo "Pi5 receiver setup complete"
echo "You should now hear audio from Pi3 through Pi5's speakers" 