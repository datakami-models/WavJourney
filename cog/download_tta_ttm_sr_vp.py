import yaml
import os

# Read the YAML file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Extract values for each application
ttm_model_size = config['AudioCraft']['ttm_model_size']
tta_model_size = config['AudioCraft']['tta_model_size']

# Download nltk
import nltk
nltk.download('punkt')

print('Step 1: Downloading TTA model ...')
os.system(f'conda run --live-stream -n WavJourney python -c \'from audiocraft.models import AudioGen; tta_model = AudioGen.get_pretrained("facebook/audiogen-{tta_model_size}")\'')

print('Step 2: Downloading TTM model ...')
os.system(f'conda run --live-stream -n WavJourney python -c \'from audiocraft.models import MusicGen; tta_model = MusicGen.get_pretrained("facebook/musicgen-{ttm_model_size}")\'')

print('Step 3: Downloading SR model ...')
os.system(f'conda run --live-stream -n WavJourney python -c \'from voicefixer import VoiceFixer; vf = VoiceFixer()\'')

print('Step 4: Downloading VP model ...')
os.system(f'conda run --live-stream -n WavJourney python -c \'from VoiceParser.model import VoiceParser; vp = VoiceParser(device="cpu")\'')

print('All models successfully downloaded!')
