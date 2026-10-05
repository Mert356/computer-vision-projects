import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMAGE_SIZE, MANIFEST_PATH


# ImageNet mean and std per RGB channel
normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    normalize
])

eval_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    normalize
])


# Labels are cat 0 and dog 1
def read_manifest(split):
    df = pd.read_csv(MANIFEST_PATH)
    df = df[df['split'] == split]

    records = []
    for _, row in df.iterrows():
        label = 0 if row['label'] == 'cat' else 1
        records.append({'path': row['path'], 'label': label})
    return records

def check_no_overlap(splits):
    seen = set()

    for records in splits.values():
        for record in records:
            path = record['path']
            if path in seen:
                raise ValueError(f"Image used twice: {path}")
            seen.add(path)



class Pets(Dataset):
    def __init__(self, base, records, transform):
        self.base = base
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = self.base / record['path']

        with Image.open(path) as image:
            image = image.convert('RGB')
            image = self.transform(image)

        label = torch.tensor(record['label'], dtype=torch.float32)
        return image, label


