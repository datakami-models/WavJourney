import os

# # Download nltk
import nltk
nltk.download('punkt')

# Downloading the TTS models
print('Downloading TTS model ...')
os.system(f'conda run --live-stream -n WavJourney python -c \'from transformers import BarkModel; BarkModel.from_pretrained("suno/bark")\'')

print('TTS model successfully downloaded!')
