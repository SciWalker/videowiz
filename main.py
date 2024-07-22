import requests
import uuid
#read from yml
import yaml
import json
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
free_tier = False
voice_id = cfg['voice_id']
model_id = cfg['model_id']
lang_code = cfg['lang_code']
key_mode=cfg['key_mode']
print(voice_id)
if key_mode=="local":
    key=json.load(open('keys.json'))['api-key']
elif key_mode=="aws_parameter_store":
    key = ssm.get_parameter(Name=cfg['key_name'], WithDecryption=True)['Parameter']['Value']
url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

with open ('data/text.txt', 'rb') as f:
    file = f.read()
    text= file.decode('utf-8')

payload = {
    "text": text,
    "model_id": model_id,
    "voice_id": voice_id,
    # "language_code": lang_code,
    "voice_settings": {
        "stability": 0,
        "similarity_boost": 1,
        "style": 0,
        "use_speaker_boost": True
    },
    # "pronunciation_dictionary_locators": [
    #     {
    #         "pronunciation_dictionary_id": "<string>",
    #         "version_id": "<string>"
    #     }
    # ],
    # "seed": 123,
    # "previous_text": "<string>",
    # "next_text": "<string>",
    # "previous_request_ids": ["<string>"],
    # "next_request_ids": ["<string>"]
}

headers = {"Content-Type": "application/json"}
client = ElevenLabs(
    api_key=key,
)
if free_tier:
    response = requests.request("POST", url, json=payload, headers=headers)
    if response.status_code != 200:
        print(f"An error occurred: {response.text}")
else:
    response = client.text_to_speech.convert(
        text= text,
        model_id= model_id,
        voice_id= voice_id,
        # "language_code": lang_code,
        voice_settings= VoiceSettings(
            stability= 0,
            similarity_boost= 1,
            style= 0,
            use_speaker_boost= True
        ),
    )

# Generating a unique file name for the output MP3 file
save_file_path = f"{uuid.uuid4()}.mp3"
# Writing the audio stream to the file



with open(save_file_path, "wb") as f:
    for chunk in response:
        if chunk:
            f.write(chunk)

print(f"A new audio file was saved successfully at {save_file_path}")
