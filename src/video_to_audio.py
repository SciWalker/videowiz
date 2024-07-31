import subprocess
def extract_audio(video_file, output_audio_file):
    command = f'ffmpeg  -y -i {video_file} -vn -acodec libmp3lame -ar 44100 -ac 2 {output_audio_file}'
    subprocess.run(command, shell=True)