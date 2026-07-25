from pathlib import Path
import json


def load_config(config_file):
    """
    读取配置文件
    """
    with open(config_file, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_sample_paths(sample_id, config_file):
    """
    根据样本编号解析所有需要的路径

    Parameters
    ----------
    sample_id : str
        例如 SC4001E0

    config_file : str
        sample_path_config.example.json

    Returns
    -------
    dict
        {
            "psg": "...",
            "hypnogram": "...",
            "truth": "..."
        }
    """

    # 如果传入的是相对路径，则自动定位到项目根目录
    config_path = Path(config_file)

    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent.parent / config_file

    config = load_config(config_path)

    data_root = Path(config["data_root"])

    # -------------------------
    # PSG
    # -------------------------
    psg_file = (
        data_root /
        f"{sample_id}{config['psg_suffix']}"
    )

    # -------------------------
    # Hypnogram
    # SC4001E0 -> SC4001EC
    # -------------------------
    hypnogram_id = sample_id[:-1] + "C"

    hypnogram_file = (
        data_root /
        f"{hypnogram_id}-{config['hypnogram_suffix']}"
    )

    # -------------------------
    # True labels
    # 每个样本独立保存
    # outputs_examples/<sample_id>/true_labels.csv
    # -------------------------

    output_dir = (
        Path(config.get("output_root", "outputs_examples"))
        /
        sample_id
)

    output_root = Path(
    config["output_root"]
)


    truth_file = (
        output_root
        /
        sample_id
        /
        config["truth_file"]
    )

    return {
        "psg": str(psg_file),
        "hypnogram": str(hypnogram_file),
        "truth": str(truth_file),
        "output_dir": str(output_dir)
    }


if __name__ == "__main__":

    result = resolve_sample_paths(
        "SC4001E0",
        "sample_path_config.example.json"
    )

    print("Resolved paths:")
    print(result)