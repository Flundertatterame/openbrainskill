from pathlib import Path
import json


def load_config(config_file):
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)



def resolve_sample_paths(sample_id, config_file):

    config = load_config(config_file)

    data_root = Path(config["data_root"])


    psg_file = (
        data_root /
        f"{sample_id}{config['psg_suffix']}"
    )


    hypnogram_id = (
        sample_id[:-1] + "C"
    )


    hypnogram_file = (
        data_root /
        f"{hypnogram_id}-{config['hypnogram_suffix']}"
    )


    return {
        "psg": str(psg_file),
        "hypnogram": str(hypnogram_file)
    }



if __name__ == "__main__":


    result = resolve_sample_paths(
        "SC4001E0",
        "sample_path_config.example.json"
    )


    print(result)