import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm  # For progress bars

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
dataset_path = r"C:\Users\Senior Mitnik\Downloads\try 5\model\resume_dataset.csv"  # Update with your dataset path
model_save_path = os.path.join(os.getcwd(), "bert_resume_classifier.pth")  # Path to save the trained model

# Hyperparameters
MAX_LENGTH = 256  # Maximum sequence length for BERT
BATCH_SIZE = 8  # Batch size for training
LEARNING_RATE = 5e-5  # Learning rate for AdamW optimizer
EPOCHS = 3  # Number of training epochs

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path, encoding="utf-8")

    # Clean column names and drop rows with missing values
    df.columns = df.columns.str.strip()
    df.dropna(subset=["Resume Text", "Industry"], inplace=True)

    # Create label mapping for industries
    label_mapping = {label: idx for idx, label in enumerate(df["Industry"].unique())}
    df["labels"] = df["Industry"].map(label_mapping)

    return df["Resume Text"].tolist(), df["labels"].tolist(), label_mapping

# Custom PyTorch Dataset
class ResumeDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
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

# Load data
texts, labels, label_mapping = load_data(dataset_path)

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Create datasets and data loaders
train_dataset = ResumeDataset(train_texts, train_labels, tokenizer)
val_dataset = ResumeDataset(val_texts, val_labels, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Load BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_mapping))
model.to(device)

# Optimizer (using PyTorch's AdamW)
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()  # Set model to training mode
    total_train_loss = 0

    # Training phase
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Training"):
        optimizer.zero_grad()  # Clear gradients

        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss  # Extract loss
        total_train_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    # Calculate average training loss
    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS} - Training Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0
    correct_predictions = 0

    with torch.no_grad():  # Disable gradient calculation
        for batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} - Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_val_loss += loss.item()

            # Calculate accuracy
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            correct_predictions += torch.sum(preds == labels).item()

    # Calculate validation metrics
    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = correct_predictions / len(val_loader.dataset)
    print(f"Epoch {epoch + 1}/{EPOCHS} - Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Save the trained model
torch.save(model.state_dict(), model_save_path)
print(f"Model saved at {model_save_path}")

# Save label mapping to a file
with open("label_mapping.txt", "w") as f:
    for label, idx in label_mapping.items():
        f.write(f"{label},{idx}\n")