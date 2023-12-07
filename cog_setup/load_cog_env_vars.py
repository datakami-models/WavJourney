import os


def load_env_variables(env_file_path):
    with open(env_file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            os.environ[key] = value