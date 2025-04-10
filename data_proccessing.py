import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import re

# Load dataset with absolute path
dataset_path = os.path.abspath(r"C:\Users\Senior Mitnik\Downloads\try 5\model\resume_dataset.csv")

# BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Function to clean text (removes symbols, multiple spaces, etc.)
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)  # Keep only alphanumeric and spaces
        text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
        return text.lower()
    return ""

# Custom PyTorch Dataset
class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):  # Reduce max_length to 256
        self.texts = [clean_text(text) for text in texts]
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

# Function to load and preprocess dataset
def load_data(file_path):
    df = pd.read_csv(file_path, encoding="utf-8")

    # Clean column names
    df.columns = df.columns.str.strip()

    # Drop rows with missing values
    df.dropna(subset=["Resume Text", "Industry"], inplace=True)

    # Check for small dataset
    if len(df) < 2:
        raise ValueError("Dataset too small! Add more resumes.")

    # Create label mapping for industries
    label_mapping = {label: idx for idx, label in enumerate(df["Industry"].unique())}
    df["labels"] = df["Industry"].map(label_mapping)

    return df["Resume Text"].tolist(), df["labels"].tolist(), label_mapping

# Function to get DataLoaders
def get_dataloaders(file_path, batch_size=4):  # Reduce batch size to 4
    texts, labels, label_mapping = load_data(file_path)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    train_dataset = ResumeDataset(train_texts, train_labels, tokenizer)
    val_dataset = ResumeDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader, label_mapping

# Run for testing
if __name__ == "__main__":
    train_loader, val_loader, label_mapping = get_dataloaders(dataset_path)
    print(f"Data Loaded: {len(train_loader.dataset)} training samples, {len(val_loader.dataset)} validation samples")
