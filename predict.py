# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, File
import os
import time
import string
import random
import datetime
import shutil
import subprocess
from cog_setup.load_cog_env_vars import load_env_variables


def convert_wav_to_mp3(wav_file_path, mp3_file_path):
    if not shutil.which("ffmpeg"):
        print("ffmpeg is not installed")
        return False

    try:
        command = f"ffmpeg -i \"{wav_file_path}\" -vn -ar 44100 -ac 2 -b:a 192k \"{mp3_file_path}\""
        subprocess.run(command, check=True, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during conversion: {e}")
        return False


def uid8():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        load_env_variables('cog_setup/cog.env')
        os.system('/opt/miniconda3/bin/conda run --live-stream -n WavJourney python cog_setup/download_tts.py')
        os.system('bash cog_setup/start_services.sh')
        time.sleep(5)

    def predict(
        self,
        prompt: str = Input(description="Prompt that describes the entire audio segment", default="A wizard giving an interview to the WizardTimes saying he lost his favorite hat")
    ) -> Path:
        """Run a single prediction on the model"""
        session_id = f'{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}_{uid8()}'
        os.system(f'/opt/miniconda3/bin/conda run --live-stream -n WavJourney python wavjourney_cli.py -f --input-text "{prompt}" --session-id "{session_id}"')
        
        output_file_path=os.path.join("output", "sessions")
        result_dir_path=os.path.join(output_file_path, session_id, "audio")
        result_file_path=os.path.join(result_dir_path, "res_" + session_id + ".wav")
        if not os.path.isfile(result_file_path):
            raise FileNotFoundError(f"Could not find resulting audio file: {result_file_path}")
        
        output_file_path=os.path.join(result_dir_path, "res_" + session_id + ".mp3")
        convert_wav_to_mp3(result_file_path, output_file_path)
        
        return Path(output_file_path)
