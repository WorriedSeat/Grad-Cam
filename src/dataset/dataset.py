import yaml
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def _load_config():
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "config" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"ERROR: config not found at {config_path}.\n"
            f"Ensure the file exists and the project structure is intact."
        )
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_data_splits():
    """Legacy: FER2013 splits. Use get_rafdb_splits() for RAF-DB."""
    config = _load_config()
    repo_root = Path(__file__).resolve().parents[2]
    data_path = Path(config["paths"]["data"])
    if not data_path.is_absolute():
        data_path = repo_root / data_path
    data = pd.read_csv(data_path)
    
    train_df = data[data['Usage'] == 'Training'].reset_index(drop=True)
    val_df   = data[data['Usage'] == 'PublicTest'].reset_index(drop=True)
    test_df  = data[data['Usage'] == 'PrivateTest'].reset_index(drop=True)
    
    return train_df, val_df, test_df


def get_rafdb_splits(val_ratio: float = 0.1, seed: int = 42):
    """
    Load RAF-DB from Hugging Face and return train/val/test datasets.
    Uses deanngkl/raf-db-7emotions. If only 'train' split exists, splits it.
    """
    from datasets import load_dataset

    ds = load_dataset("deanngkl/raf-db-7emotions")
    splits = list(ds.keys())

    if "train" in splits and "test" in splits:
        train_hf = ds["train"]
        test_hf = ds["test"]
        split = train_hf.train_test_split(test_size=val_ratio, seed=seed)
        train_hf = split["train"]
        val_hf = split["test"]
        return train_hf, val_hf, test_hf
    elif "train" in splits:
        train_hf = ds["train"]
        split = train_hf.train_test_split(test_size=val_ratio * 2, seed=seed)
        train_hf = split["train"]
        val_test = split["test"].train_test_split(test_size=0.5, seed=seed)
        val_hf = val_test["train"]
        test_hf = val_test["test"]
        return train_hf, val_hf, test_hf
    else:
        raise ValueError(f"Unexpected RAF-DB splits: {splits}")


class RAFDBDataset(Dataset):
    """
    PyTorch Dataset wrapping Hugging Face RAF-DB split.
    Expects columns: image (PIL), label (int 0-6).
    """

    def __init__(self, hf_split, transform=None):
        self.hf_split = hf_split
        self.transform = transform
        self.itoc = {
            0: "Anger", 1: "Disgust", 2: "Fear", 3: "Happy",
            4: "Sad", 5: "Surprise", 6: "Neutral",
        }

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        row = self.hf_split[idx]
        img = row["image"]
        if not hasattr(img, "convert"):
            import numpy as np
            img = np.array(img)
            from PIL import Image
            img = Image.fromarray(img).convert("RGB")
        elif img.mode != "RGB":
            img = img.convert("RGB")
        label = int(row["label"])
        if label not in range(7):
            label = max(0, min(6, label))
        if self.transform:
            img = self.transform(img)
        return img, label

class FERDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        self.itoc = {0:'Anger', 
                     1:'Disgust', 
                     2:'Fear', 
                     3:'Happy', 
                     4:'Sad', 
                     5:'Surprise', 
                     6:'Neutral'}
        self.ctoi = {v:k for k, v in self.itoc.items()}
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pixels = self.df.loc[idx, 'pixels']
        img = np.fromstring(pixels, dtype=np.uint8, sep=' ').reshape(48, 48)
        img = np.stack((img,) * 3, axis=-1)   # (48, 48, 3)

        label = self.df.loc[idx, 'emotion']

        if self.transform:
            img = self.transform(img)
        return img, label

def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )
