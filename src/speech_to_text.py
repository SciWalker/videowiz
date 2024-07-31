import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import time
import csv
import os
import json
import boto3
import requests
import botocore
import yaml
import numpy as np
from pydub import AudioSegment
from deepgram import DeepgramClient, PrerecordedOptions, FileSource

def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def upload_to_s3(local_file_path, bucket_name, s3_client=None):
    s3_file_key = "vocal-language-converter/" + os.path.basename(local_file_path)
    if s3_client is None:
        s3_client = boto3.client("s3")

    try:
        try:
            s3_client.head_object(Bucket=bucket_name, Key=s3_file_key)
            print(f"File already exists in S3: {s3_file_key}")
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "404":
                s3_client.upload_file(local_file_path, bucket_name, s3_file_key)
                print(f"File uploaded to S3: {s3_file_key}")
            else:
                print(f"Error checking file in S3: {e}")
                return None

        return f"s3://{bucket_name}/{s3_file_key}"
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        return None

def amazon_transcribe_from_s3(s3_uri, language_code, transcribe_client=None):
    if transcribe_client is None:
        transcribe_client = boto3.client("transcribe")

    job_name = f"{language_code}_{os.path.basename(s3_uri).split('.')[0]}"
    media_format = s3_uri.split(".")[-1]

    response = transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        LanguageCode=language_code,
        MediaFormat=media_format,
        Media={"MediaFileUri": s3_uri},
    )

    while True:
        response = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        job_status = response["TranscriptionJob"]["TranscriptionJobStatus"]
        
        if job_status == "COMPLETED":
            transcript_uri = response["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
            transcript_response = requests.get(transcript_uri)
            transcript_data = transcript_response.json()
            return transcript_data["results"]["transcripts"][0]["transcript"]
        elif job_status == "FAILED":
            return None
        
        time.sleep(5)
        print("Transcribing...")

def whisper_transcribe(audio_path, model_id="openai/whisper-large-v3"):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=512,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipe(
        audio_path,
        generate_kwargs={"task": "transcribe", "temperature": 1, "do_sample": True},
    )

    return result["text"]
def normalize_audio_samples(samples):
    """Normalize audio samples to the range [-1, 1]."""
    sample_type = samples.dtype
    if np.issubdtype(sample_type, np.integer):
        info = np.iinfo(sample_type)
        samples = samples.astype(np.float32) / max(abs(info.min), abs(info.max))
    elif np.issubdtype(sample_type, np.floating):
        samples = np.clip(samples, -1.0, 1.0)
    return samples

def whisper_transcribe_with_token_chunking(audio_path, model_id="openai/whisper-large-v3", max_tokens=448):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    
    # Adjust max_new_tokens to account for initial tokens
    adjusted_max_tokens = max_tokens - 3  # Subtracting 3 for the initial tokens
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=adjusted_max_tokens,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )

    # Load and preprocess audio file
    audio = AudioSegment.from_file(audio_path)
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)
    if audio.channels == 2:
        audio = audio.set_channels(1)
    
    samples = np.array(audio.get_array_of_samples())
    samples = normalize_audio_samples(samples)

    full_transcript = ""
    offset = 0
    
    while offset < len(samples):
        end = min(offset + 16000 * 30, len(samples))  # Start with 30 seconds chunk
        chunk = {"array": samples[offset:end], "sampling_rate": 16000}
        
        result = pipe(chunk, generate_kwargs={"task": "transcribe", "temperature": 1, "do_sample": True})
        chunk_transcript = result["text"]
        chunk_tokens = processor.tokenizer.encode(chunk_transcript)
        
        if len(chunk_tokens) > adjusted_max_tokens:
            # If chunk produces too many tokens, reduce chunk size and retry
            while len(chunk_tokens) > adjusted_max_tokens and end > offset:
                end = offset + (end - offset) // 2
                chunk = {"array": samples[offset:end], "sampling_rate": 16000}
                result = pipe(chunk, generate_kwargs={"task": "transcribe", "temperature": 1, "do_sample": True})
                chunk_transcript = result["text"]
                chunk_tokens = processor.tokenizer.encode(chunk_transcript)
        
        full_transcript += f" {chunk_transcript}"
        offset = end
        
        print(f"Processed chunk, current offset: {offset}/{len(samples)}")

    return full_transcript.strip()

def deepgram_transcribe(audio_path, api_key):
    try:
        deepgram = DeepgramClient(api_key)

        with open(audio_path, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            diarize=True,
            detect_language=True,
        )

        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)
        print(response)
        # Extract the transcript from the response
        # transcript = response.results.channels[0].alternatives[0].transcript

        return response

    except Exception as e:
        print(f"Deepgram transcription error: {e}")
        return None

def transcribe_audio(audio_path, config_path, transcriber="whisper"):
    config = load_config(config_path)
    bucket_name = config["bucket_name"]
    tgt_language = config["tgt_language"]

    start_time = time.time()

    if transcriber == "amazon":
        s3_file_uri = upload_to_s3(audio_path, bucket_name)
        if s3_file_uri:
            result = amazon_transcribe_from_s3(s3_file_uri, tgt_language)
        else:
            result = "Error uploading file to S3"
    elif transcriber == "deepgram":
        with open("keys.json") as file:
            api_key = json.load(file).get("deepgram-api-key")
        if not api_key:
            raise ValueError("Deepgram API key not found in config file")
        result = deepgram_transcribe(audio_path, api_key)
    else:  # default to whisper
        result = whisper_transcribe_with_token_chunking(audio_path)

    end_time = time.time()
    total_time = end_time - start_time

    return {
        "transcription": result,
        "runtime": total_time,
        "model": transcriber
    }

def save_result(result, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    try:
        with open(output_file, "w") as json_file:
            json.dump(result, json_file, indent=2)
    except Exception as e:
        with open("output.txt" , "w") as file:
            file.write(str(result))

def append_to_csv(result, csv_file, audio_path):
    with open(csv_file, "a", newline="") as file:
        fieldnames = ["model", "file_name", "run_time", "results"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if os.stat(csv_file).st_size == 0:
            writer.writeheader()
        writer.writerow({
            "model": result["model"],
            "file_name": os.path.basename(audio_path),
            "run_time": result["runtime"],
            "results": result["transcription"]
        })

if __name__ == "__main__":
    config_path = "src/param.yaml"
    audio_path = "data/output/new_audio.mp3"
    output_folder = "./outputs"
    
    result = transcribe_audio(audio_path, config_path, transcriber="deepgram")
    
    base_filename = os.path.splitext(os.path.basename(audio_path))[0]
    output_file = os.path.join(output_folder, f"{base_filename}_result.json")
    save_result(result, output_file)
    
    csv_file = os.path.join(output_folder, "runtime.csv")
    append_to_csv(result, csv_file, audio_path)
    
    print(f"Transcription: {result['transcription']}")
    print(f"Total Runtime: {result['runtime']}s")