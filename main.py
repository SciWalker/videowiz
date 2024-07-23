import requests
import uuid
import yaml
import json
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
import io

# Read from yml
with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

free_tier = True
voice_id = cfg['voice_id']
model_id = cfg['model_id']
lang_code = cfg['lang_code']
url="https://api.elevenlabs.io/v1/text-to-speech"
headers = {"Content-Type": "application/json"}
# Read the text file
with open('data/text.txt', 'rb') as f:
    file = f.read()
    text = file.decode('utf-8')

# Set up ElevenLabs client
if free_tier:
    key = None
    # Chunk the text into around 400-500 characters, ensure the last character is a full stop
    text_list = []
    while len(text) > 450:
        last_period = text[:450].rfind(".")
        text_list.append(text[:last_period+1])
        text = text[last_period+1:]
    text_list.append(text)
else:
    text_list = [text]
    key_mode = cfg['key_mode']
    if key_mode == "local":
        key = json.load(open('keys.json'))['api-key']
    elif key_mode == "aws_parameter_store":
        key = ssm.get_parameter(Name=cfg['key_name'], WithDecryption=True)['Parameter']['Value']

client = ElevenLabs(api_key=key)

# Process each chunk and store the audio data
audio_segments = []
for i, text_chunk in enumerate(text_list):
    if free_tier:
        payload={
            "text": text_chunk,
            "voice": voice_id,
            "lang_code": lang_code,
            "voice_settings": {
                "stability": 0,
                "similarity_boost": 1,
                "style": 0,
                "use_speaker_boost": True
            }
        }
        response = requests.request("POST", url, json=payload, headers=headers)
        if response.status_code != 200:
            print(f"An error occurred: {response.text}")
        # Collect all chunks from the generator
        audio_data = b''.join(chunk for chunk in response if chunk is not None)
        
        # Convert the audio data to an AudioSegment
        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
        audio_segments.append(audio_segment)
    else:
        response = client.text_to_speech.convert(
            text=text_chunk,
            model_id=model_id,
            voice_id=voice_id,
            voice_settings=VoiceSettings(
                stability=0,
                similarity_boost=1,
                style=0,
                use_speaker_boost=True
            ),
        )
    
        # Collect all chunks from the generator
        audio_data = b''.join(chunk for chunk in response if chunk is not None)
        
        # Convert the audio data to an AudioSegment
        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
        audio_segments.append(audio_segment)

# Combine all audio segments
combined_audio = sum(audio_segments)

# Generate a unique filename
save_file_path = f"{uuid.uuid4()}.mp3"

# Export the combined audio to a file
combined_audio.export(save_file_path, format="mp3")

print(f"A new audio file was saved successfully at {save_file_path}")