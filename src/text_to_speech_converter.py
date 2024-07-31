import requests
import uuid
import yaml
import json
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment
import io

class TextToSpeechConverter:
    def __init__(self, config_file, text_file):
        with open(config_file, 'r') as ymlfile:
            self.cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        
        self.free_tier = True
        self.voice_id = self.cfg['voice_id']
        self.model_id = self.cfg['model_id']
        self.lang_code = self.cfg['lang_code']
        self.text_file = text_file
        
        self.url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        self.headers = {"Content-Type": "application/json"}
        
        self.client = None
        if not self.free_tier:
            key_mode = self.cfg['key_mode']
            if key_mode == "local":
                self.key = json.load(open('keys.json'))['api-key']
            elif key_mode == "aws_parameter_store":
                self.key = ssm.get_parameter(Name=self.cfg['key_name'], WithDecryption=True)['Parameter']['Value']
            self.client = ElevenLabs(api_key=self.key)

    def _read_text(self):
        with open(self.text_file, 'rb') as f:
            file = f.read()
            text = file.decode('utf-8')
        return text

    def _chunk_text(self, text):
        text_list = []
        if self.free_tier:
            while len(text) > 450:
                last_period = text[:450].rfind(".")
                text_list.append(text[:last_period+1])
                text = text[last_period+1:]
            text_list.append(text)
        else:
            text_list = [text]
        return text_list

    def _process_chunk(self, text_chunk):
        if self.free_tier:
            payload = {
                "text": text_chunk,
                "voice": self.voice_id,
                "lang_code": self.lang_code,
                "voice_settings": {
                    "stability": 0,
                    "similarity_boost": 1,
                    "style": 0,
                    "use_speaker_boost": True
                }
            }
            response = requests.request("POST", self.url, json=payload, headers=self.headers)
            if response.status_code != 200:
                raise Exception(f"An error occurred: {response.text}")
            audio_data = b''.join(chunk for chunk in response if chunk is not None)
        else:
            response = self.client.text_to_speech.convert(
                text=text_chunk,
                model_id=self.model_id,
                voice_id=self.voice_id,
                voice_settings=VoiceSettings(
                    stability=0,
                    similarity_boost=1,
                    style=0,
                    use_speaker_boost=True
                ),
            )
            audio_data = b''.join(chunk for chunk in response if chunk is not None)
        
        audio_segment = AudioSegment.from_mp3(io.BytesIO(audio_data))
        return audio_segment

    def convert_text_to_speech(self):
        text = self._read_text()
        text_chunks = self._chunk_text(text)
        audio_segments = [self._process_chunk(chunk) for chunk in text_chunks]
        combined_audio = sum(audio_segments)
        
        save_file_path = f"data/{uuid.uuid4()}_{self.voice_id}.mp3"
        combined_audio.export(save_file_path, format="mp3")
        print(f"A new audio file was saved successfully at {save_file_path}")
        return save_file_path

# Example usage
# converter = TextToSpeechConverter(config_file="config.yml", text_file="data/text.txt")
# converter.convert_text_to_speech()
