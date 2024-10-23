import os
import grpc
from concurrent import futures
import time
import tts_pb2
import tts_pb2_grpc
import torch
import numpy as np
import logging
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from huggingface_hub import snapshot_download
from scipy.signal import resample
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Global variables for the model and conditioning parameters
model = None
gpt_cond_latent = None
speaker_embedding = None
def split_text_into_chunks(text, max_length=180):
    """
    Splits the input text into chunks of maximum specified length.
    Preferably splits on sentence boundaries or punctuation marks.

    Args:
        text (str): The input text to be split.
        max_length (int): Maximum number of characters per chunk.

    Returns:
        List[str]: A list of text chunks.
    """
    # Split text into sentences using punctuation as delimiters
    sentence_endings = re.compile(r'(?<=[.!?]) +')
    sentences = sentence_endings.split(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_length:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # If the sentence itself is longer than max_length, split it further
            if len(sentence) > max_length:
                for i in range(0, len(sentence), max_length):
                    part = sentence[i:i + max_length].strip()
                    if part:
                        chunks.append(part)
                current_chunk = ""
            else:
                current_chunk = sentence

    if current_chunk:
        chunks.append(current_chunk)

    return chunks



def convert_float32_to_pcm(input_audio, input_sample_rate=24000, target_sample_rate=16000):
    """
    Converts a numpy array of audio data from float32 format to int16 PCM format, with resampling.

    Parameters:
    - input_audio: numpy array with dtype=float32. Shape can be (n_samples,) for mono or (n_samples, 2) for stereo.
    - input_sample_rate: Sample rate of the input audio (e.g., 24000)
    - target_sample_rate: Desired sample rate for output audio (e.g., 8000 or 16000)

    Returns:
    - PCM encoded byte data of the resampled audio
    """
    # Ensure audio data is a numpy array
    if not isinstance(input_audio, np.ndarray):
        raise ValueError("Input audio must be a numpy array.")

    # Normalize the data to be in the range [-1.0, 1.0]
    input_audio = np.clip(input_audio, -1.0, 1.0)

    # Determine if the input is stereo or mono
    if len(input_audio.shape) == 1:
        # Mono input
        is_stereo = False
        input_audio = input_audio[:, np.newaxis]  # Convert to (n_samples, 1)
    elif len(input_audio.shape) == 2 and input_audio.shape[1] == 2:
        # Stereo input
        is_stereo = True
    else:
        raise ValueError("Input audio must be mono or stereo with shape (n_samples,) or (n_samples, 2).")

    # Resample the audio
    num_samples = int(len(input_audio) * target_sample_rate / input_sample_rate)
    if is_stereo:
        resampled_audio = np.zeros((num_samples, 2), dtype=np.float32)
        resampled_audio[:, 0] = resample(input_audio[:, 0], num_samples)
        resampled_audio[:, 1] = resample(input_audio[:, 1], num_samples)
    else:
        resampled_audio = resample(input_audio[:, 0], num_samples)
        resampled_audio = resampled_audio[:, np.newaxis]  # Convert back to 2D for mono

    # Convert to int16 PCM format
    pcm_audio = (resampled_audio * 32767).astype(np.int16)

    # Return as bytes
    return pcm_audio.tobytes()


class TTSService(tts_pb2_grpc.TTSServiceServicer):
    def __init__(self):
        global model, gpt_cond_latent, speaker_embedding
        # Initialize the TTS model
        checkpoint_path = snapshot_download("coqui/XTTS-v2")
        config_path = os.path.join(checkpoint_path, "config.json")
        config = XttsConfig()
        config.load_json(config_path)
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=checkpoint_path)
        if torch.cuda.is_available():
            logging.info("CUDA is available, moving model to GPU.")
            model.cuda()
        else:
            logging.info("CUDA is not available, moving model to CPU.")
            model.cpu()
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=["ref5.wav"])
        logging.info("TTS Model loaded and ready.")

    def StreamTTS(self, request_iterator, context):
        """
        Bi-directional streaming RPC for TTS.
        Receives TTSRequest messages and yields TTSResponse messages.
        """
        # Check if request_iterator is actually a single request
        if isinstance(request_iterator, tts_pb2.TTSRequest):
            request_iterator = [request_iterator]  # Convert to an iterable

        for request in request_iterator:
            message = request.message
            language = request.language
            speed = request.speed
            logging.info(f"Received TTS request: {message} (Language: {language}, Speed: {speed})")

            if not message:
                context.set_details("No message provided.")
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                continue
            message_chunks = split_text_into_chunks(message)
            # Split the message into chunks if necessary
            # Implement chunking logic if needed

            try:
                # Perform TTS inference
                for chunk in message_chunks:
                    # Perform TTS inference for each chunk
                    with torch.no_grad():
                        audio_chunks = model.inference_stream(
                            chunk,
                            language=language,
                            gpt_cond_latent=gpt_cond_latent,
                            speaker_embedding=speaker_embedding,
                            speed=speed
                        )

                    logging.info(f"Streaming TTS for chunk: {chunk}")

                    for audio_chunk in audio_chunks:
                        chunk_np = audio_chunk.cpu().numpy().squeeze().astype('float32')
                        audio_bytes = convert_float32_to_pcm(chunk_np, 24000, 8000)
                        yield tts_pb2.TTSResponse(audio_chunk=audio_bytes, end_of_audio=False)

                # Indicate end of audio
                yield tts_pb2.TTSResponse(audio_chunk=b'', end_of_audio=True)

            except Exception as e:
                logging.error(f"TTS processing error: {e}")
                context.set_details(str(e))
                context.set_code(grpc.StatusCode.INTERNAL)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    tts_pb2_grpc.add_TTSServiceServicer_to_server(TTSService(), server)
    server_address = '[::]:50051'
    server.add_insecure_port(server_address)
    logging.info(f"Starting gRPC server on {server_address}")
    server.start()
    try:
        while True:
            time.sleep(86400)  # Keep the server running for a day
    except KeyboardInterrupt:
        logging.info("Shutting down gRPC server.")
        server.stop(0)

if __name__ == "__main__":
    serve()