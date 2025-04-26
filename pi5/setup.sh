#!/bin/bash
# Pi5 ElevenLabs Listener Setup Script

echo "====================================================="
echo "Pi5 ElevenLabs Listener Setup"
echo "====================================================="

# Install system dependencies (required for PyAudio)
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-dev libatlas-base-dev

# Create directory structure
INSTALL_DIR="/home/pi/pi_pal"
echo "Creating installation directory: $INSTALL_DIR"
mkdir -p $INSTALL_DIR
mkdir -p $INSTALL_DIR/logs
mkdir -p $INSTALL_DIR/audio_responses

# Install dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Copy files
echo "Copying files to installation directory..."
cp pi5_elevenlabs_listener.py $INSTALL_DIR/
cp requirements.txt $INSTALL_DIR/

# Create environment file
ENV_FILE="$INSTALL_DIR/.env"
echo "Creating environment file: $ENV_FILE"
echo "# ElevenLabs API credentials" > $ENV_FILE
echo "ELEVENLABS_API_KEY=your_api_key_here" >> $ENV_FILE
echo "AGENT_ID=your_agent_id_here" >> $ENV_FILE

# Set permissions
echo "Setting permissions..."
chmod +x $INSTALL_DIR/pi5_elevenlabs_listener.py
chmod 600 $ENV_FILE  # Restricted permissions for API key

# Setup systemd service
echo "Setting up systemd service..."
SERVICE_FILE="service_file/elevenlabs-listener.service"
if [ -f "$SERVICE_FILE" ]; then
    sudo cp $SERVICE_FILE /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable elevenlabs-listener.service
    echo "Service installed and enabled. To start it, run: sudo systemctl start elevenlabs-listener.service"
else
    echo "Service file not found: $SERVICE_FILE"
    echo "Service installation skipped. You'll need to set this up manually."
fi

echo "====================================================="
echo "Setup complete!"
echo ""
echo "IMPORTANT STEPS TO COMPLETE SETUP:"
echo "1. Edit $ENV_FILE to add your actual ElevenLabs API key and Agent ID"
echo "2. Test the listener by running: python3 $INSTALL_DIR/pi5_elevenlabs_listener.py"
echo "3. Once confirmed working, start the service with: sudo systemctl start elevenlabs-listener.service"
echo "=====================================================" 