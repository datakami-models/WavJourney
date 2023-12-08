# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path

import yaml
import os
import subprocess
import sys
import shutil
import nltk
import pipeline
import time
from transformers import BarkModel
from audiocraft.models import AudioGen, MusicGen
from voicefixer import VoiceFixer
from VoiceParser.model import VoiceParser

WAVJOURNEY_SERVICE_PORT = '8021'
WAVJOURNEY_SERVICE_URL = '127.0.0.1'
WAVJOURNEY_MAX_SCRIPT_LINES = '999'


def convert_wav_to_mp3(wav_file_path, mp3_file_path):
    if not shutil.which("ffmpeg"):
        raise FileNotFoundError("Could not find ffmpeg")

    subprocess.check_call(["ffmpeg", "-i", f"{wav_file_path}", "-vn", "-ar", "44100", "-ac", "2", "-b:a", "192k", f"{mp3_file_path}"])


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download models with code corresponding to ./scripts/download_models.py
        # Read the YAML file
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        # Extract values for each application
        ttm_model_size = config['AudioCraft']['ttm_model_size']
        tta_model_size = config['AudioCraft']['tta_model_size']
        nltk.download('punkt')

        print('Step 1: Downloading TTS model ...')
        BarkModel.from_pretrained("suno/bark")

        print('Step 2: Downloading TTA model ...')
        tta_model = AudioGen.get_pretrained(f"facebook/audiogen-{tta_model_size}")

        print('Step 3: Downloading TTM model ...')
        tta_model = MusicGen.get_pretrained(f"facebook/musicgen-{ttm_model_size}")

        print('Step 4: Downloading SR model ...')
        vf = VoiceFixer()

        print('Step 5: Downloading VP model ...')
        vp = VoiceParser(device="cpu")

        print('All models successfully downloaded!')
        os.environ['WAVJOURNEY_SERVICE_PORT'] = WAVJOURNEY_SERVICE_PORT
        os.environ['WAVJOURNEY_SERVICE_URL'] = WAVJOURNEY_SERVICE_URL
        os.environ['WAVJOURNEY_MAX_SCRIPT_LINES'] =WAVJOURNEY_MAX_SCRIPT_LINES
        service_env = os.environ.copy()
        print('Starting Audio Generation API ...')
        Path("./logs").mkdir(exist_ok=True)
        with open("./logs/services_stdout.txt","wb") as out, open("./logs/services_stderr.txt","wb") as err:
            subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "services.py")], env=service_env, stdout=out, stderr=err)
            time.sleep(30)
        print('Audio Generation API started.')

    def predict(
        self,
        prompt: str = Input(description="Prompt that describes the entire audio segment", default="A wizard giving an interview to the WizardTimes saying he lost his favorite hat"),
        model: str = Input(description="Choose the model to generate the audio script. GPT4 is standard, Llama2 is possible but has low success rate (10 percent according to paper).", default="gpt", choices=["gpt", "llama2"]),
        gpt_api_key: str = Input(description="OpenAI API key when using the gpt model for audio script generation.", default=""),
        replicate_api_key: str = Input(description="Replicate API key when using the llama2 70B model for audio script generation.", default=""),
    ) -> Path:
        """Run a single prediction on the model"""

        session_id = pipeline.init_session()

        if model == "llama2":
            os.environ["REPLICATE_API_TOKEN"] = replicate_api_key
            api_key = replicate_api_key
        else:
            os.environ["WAVJOURNEY_OPENAI_KEY"] = gpt_api_key
            api_key = gpt_api_key

        pipeline.full_steps(session_id, prompt, api_key, model)

        result_file_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "sessions", session_id, "audio", f"res_{session_id}.wav")
        if not os.path.isfile(result_file_path):
            raise FileNotFoundError(f"Could not find resulting audio file: {result_file_path}")

        output_file_path=os.path.join(result_file_path[:-3] + "mp3")
        convert_wav_to_mp3(result_file_path, output_file_path)
        return Path(output_file_path)
