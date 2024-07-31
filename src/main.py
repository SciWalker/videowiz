# text_to_speech_script.py

from text_to_speech_converter import TextToSpeechConverter
from video_to_audio import extract_audio
def main():
    config_file = "config.yml"
    text_file = "data/text.txt"
    video_path = "data/external/downloaded_videos/federal.mp4"
    audio_file=extract_audio(video_path, "data/output/new_audio.mp3")
    text_to_speech_converter = TextToSpeechConverter(config_file=config_file, text_file=text_file)
    
    audio_file_path = text_to_speech_converter.convert_text_to_speech()
    
    print(f"Audio file saved at: {audio_file_path}")

if __name__ == "__main__":
    main()
