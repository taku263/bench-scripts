# Pi5 ElevenLabs Listener

This component is designed to run on a Raspberry Pi 5 and connect directly to the ElevenLabs Conversation API to listen for and play audio responses. It works together with the voice assistant running on a Raspberry Pi 3 that handles wake word detection and audio input.

## How It Works

The distributed voice assistant system works as follows:

1. The Raspberry Pi 3 handles:
   - Wake word detection ("Hey Pal")
   - Audio input (microphone)
   - Sending audio to ElevenLabs

2. The Raspberry Pi 5 handles:
   - Connecting to the same ElevenLabs conversation
   - Receiving audio responses
   - Playing audio through speakers

Both devices connect to the same ElevenLabs agent using the same credentials, creating a seamless experience where you speak to one device and hear responses from the other.

## API Details

This script uses the ElevenLabs WebSocket API for conversational AI:
- WebSocket URL: `wss://api.elevenlabs.io/v1/convai/conversation`
- The agent_id is passed as a query parameter
- Authentication is done via the `xi-api-key` header

## Requirements

- Raspberry Pi 5 with speakers or audio output
- Python 3.7+
- ElevenLabs API key
- Agent ID for your ElevenLabs conversation agent
- System dependencies:
  - portaudio19-dev (for audio playback)
  - python3-dev
  - libatlas-base-dev (for numpy acceleration)

## Installation

1. Clone this repository onto your Raspberry Pi 5:
   ```bash
   git clone https://github.com/yourusername/pi-pal.git
   cd pi-pal/pi5
   ```

2. Create a `.env` file with your credentials:
   ```bash
   cp .env.example .env
   nano .env  # Edit to add your API key and Agent ID
   ```

3. Run the setup script to install dependencies and configure the service:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

## Manual Installation

If you prefer to install manually:

1. Install system dependencies first:
   ```bash
   sudo apt-get update
   sudo apt-get install -y portaudio19-dev python3-dev libatlas-base-dev
   ```

2. Install Python dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

3. Copy the listener script to where you want to run it:
   ```bash
   mkdir -p /home/pi/pi_pal
   cp pi5_elevenlabs_listener.py /home/pi/pi_pal/
   ```

4. Create a `.env` file with your credentials:
   ```bash
   cp .env.example /home/pi/pi_pal/.env
   nano /home/pi/pi_pal/.env  # Edit this file
   ```

5. To run as a service, copy the service file:
   ```bash
   sudo cp service_file/elevenlabs-listener.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable elevenlabs-listener.service
   sudo systemctl start elevenlabs-listener.service
   ```

## Usage

Once installed and configured, the service will automatically start at boot and connect to ElevenLabs.

To start manually:
```bash
python3 /home/pi/pi_pal/pi5_elevenlabs_listener.py
```

To check the service status:
```bash
sudo systemctl status elevenlabs-listener.service
```

To view logs:
```bash
tail -f /home/pi/pi_pal/elevenlabs_listener.log
```

## Troubleshooting

- **No audio output**: Check that your Pi 5's audio output is properly configured and working.
  - Try running `aplay -l` to list available audio devices
  - Ensure your speakers are connected and volume is up using `alsamixer`
  - For USB audio devices, make sure they're properly connected
  
- **Connection issues**: Verify your API key and Agent ID are correct in the `.env` file.

- **Service not starting**: Check logs with `journalctl -u elevenlabs-listener.service`

- **PyAudio installation errors**: Make sure you've installed the system dependencies listed in the requirements section.

- **ALSA warnings**: The numerous ALSA warnings that appear in the console are normal and generally don't affect functionality. They are suppressed in the latest version of the script.

## Audio Device Selection

By default, the listener uses the default audio output. To select a specific audio device:

1. List available audio devices:
   ```bash
   aplay -l
   ```
   
   Example output:
   ```
   **** List of PLAYBACK Hardware Devices ****
   card 0: vc4hdmi [vc4-hdmi], device 0: MAI PCM i2s-hifi-0 [MAI PCM i2s-hifi-0]
   card 1: Headphones [bcm2835 Headphones], device 0: bcm2835 Headphones [bcm2835 Headphones]
   card 2: DAC [USB Audio DAC], device 0: USB Audio [USB Audio]
   ```

2. Choose the device you want to use and add it to the `.env` file:
   ```
   AUDIO_DEVICE=2  # For the USB Audio DAC in the example above
   ```

The script will automatically:
- List all available audio devices on startup
- Use the specified device if available
- Fall back to the default device if the specified one isn't available or doesn't work

## Force Raspberry Pi to Use Specific Audio Output

If you're still having issues with audio output selection:

1. Edit or create `/etc/asound.conf`:
   ```
   sudo nano /etc/asound.conf
   ```

2. For HDMI output, add:
   ```
   pcm.!default {
     type hw
     card 0
   }
   ctl.!default {
     type hw           
     card 0
   }
   ```

3. For headphone jack output, add:
   ```
   pcm.!default {
     type hw
     card 1
   }
   ctl.!default {
     type hw           
     card 1
   }
   ```

4. For a USB audio device, add:
   ```
   pcm.!default {
     type hw
     card 2
   }
   ctl.!default {
     type hw           
     card 2
   }
   ```

## Monitoring

The listener creates logs in two locations:
- Service logs: `journalctl -u elevenlabs-listener.service`
- Application logs: `/home/pi/pi_pal/elevenlabs_listener.log`

Conversation transcripts and fallback audio files are saved to:
- `/home/pi/pi_pal/audio_responses/` 