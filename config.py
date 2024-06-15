from pathlib import Path

def get_config():
    return {
        "lang_src": "Vietnamese",  # Đặt ngôn ngữ nguồn là Vietnamese
        "lang_tgt": "English",     # Đặt ngôn ngữ đích là English
        "seq_len": 80,
        "batch_size": 128,
        "datasource": "harouzie",
        "model_folder": "model_weights",
        "experiment_name": "translation_experiment",
        "lr": 1e-4,
        "d_model": 256,
        "num_epochs": 10,
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{lang}.json"  # Đường dẫn để lưu tokenizer
    }


def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
