import os
import time
import argparse

import utils
import pipeline

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--full', action='store_true', help='Go through the full process')
parser.add_argument('--input-text', type=str, default='', help='input text or text file')
parser.add_argument('--session-id', type=str, default='', help='session id, if set to empty, system will allocate an id')
parser.add_argument('--model-name', type=str, default='gpt', help='model name used to generate audio script: gpt (default) or llama2')
args = parser.parse_args()

if args.full:
    input_text = args.input_text

    start_time = time.time()
    session_id = pipeline.init_session(args.session_id)

    model_name = args.model_name
    if model_name == "llama2":
        api_key = os.getenv("REPLICATE_API_TOKEN")
    else:
        api_key = utils.get_api_key()

    assert api_key != None, "Please set your openai_key in the environment variable."
    
    print(f"Session {session_id} is created.")

    pipeline.full_steps(session_id, input_text, api_key, model_name)
    end_time = time.time()

    print(f"WavJourney took {end_time - start_time:.2f} seconds to complete.")
