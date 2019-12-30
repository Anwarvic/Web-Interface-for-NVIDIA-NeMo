import yaml
import json
from librosa import get_duration


def parse_yaml(filepath="conf.yaml"):
    """
    This method parses the YAML configuration file and returns the parsed info
    as python dictionary.
    Args:
        filepath (string): relative path of the YAML configuration file
    """
    with open(filepath, 'r') as fin:
        try:
            d = yaml.safe_load(fin)
            return d
        except Exception as exc:
            print("ERROR while parsing YAML conf.")
            return exc


def build_manifest(wav_path):
    """
    This function takes the absolute path of the wav file and writes
    the metadata of the wav into json file on the following format:
    {
        "audio_filepath": wav_path,
        "duration": 100,
        "text": ""
    }
    """
    filename = 'audio.json'
    duration = get_duration(filename=wav_path)
    d = {
        "audio_filepath": wav_path,
        "duration": duration,
        "text": ""
    }
    with open(filename, 'w') as fout:
        json.dump(d, fout)
    return filename



