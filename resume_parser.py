import torch
from transformers import BertTokenizer, BertForSequenceClassification
import os
import re

def load_label_mapping(label_mapping_path):
    """Loads the label mapping from a file."""
    label_mapping = {}
    with open(label_mapping_path, "r") as f:
        for line in f:
            label, idx = line.strip().split(",")
            label_mapping[int(idx)] = label  # Reverse mapping for predictions
    return label_mapping

# Paths
model_path = os.path.join(os.getcwd(), r"C:\Users\Senior Mitnik\Downloads\try 5\model\bert_resume_classifier.pth")
label_mapping_path = os.path.join(os.getcwd(), r"C:\Users\Senior Mitnik\Downloads\try 5\model\label_mapping.txt")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
label_mapping = load_label_mapping(label_mapping_path)

# Load trained model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(label_mapping))
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

def extract_details_from_resume(resume_text):
    """Extracts details like name, email, contact, summary, skills, experience, and leadership skills from resume text."""
    details = {
        "Name": "N/A",
        "Email": "N/A",
        "Contact": "N/A",
        "Summary": "N/A",
        "Industry": "N/A",
        "Skills": [],
        "Experience": [],
        "Leadership": []
    }

    # Extract Name (assuming the first line is the name)
    lines = resume_text.splitlines()
    if lines:
        details["Name"] = lines[0].strip()

    # Extract Email
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    email_match = re.search(email_pattern, resume_text)
    if email_match:
        details["Email"] = email_match.group(0)

    # Extract Contact (assuming it's a phone number)
    phone_pattern = r"\+?\d[\d -]{8,12}\d"
    phone_match = re.search(phone_pattern, resume_text)
    if phone_match:
        details["Contact"] = phone_match.group(0)

    # Extract Summary (assuming it's the first few lines after the name)
    if len(lines) > 1:
        details["Summary"] = " ".join(lines[1:4]).strip()

    # Extract Skills, Experience, and Leadership (basic keyword matching)
    skills_keywords = ["skill", "proficient", "expert", "knowledge", "ability"]
    experience_keywords = ["experience", "work", "job", "position", "role"]
    leadership_keywords = ["lead", "manage", "supervise", "direct", "team"]

    for line in lines:
        if any(keyword in line.lower() for keyword in skills_keywords):
            details["Skills"].append(line.strip())
        if any(keyword in line.lower() for keyword in experience_keywords):
            details["Experience"].append(line.strip())
        if any(keyword in line.lower() for keyword in leadership_keywords):
            details["Leadership"].append(line.strip())

    return details

def calculate_score(details):
    """Calculates a score based on skills, experience, and leadership skills."""
    max_score = 100
    score = 0

    # Score for skills (up to 40%)
    skills_score = min(len(details["Skills"]) * 5, 40)  # 5 points per skill, max 40
    score += skills_score

    # Score for experience (up to 30%)
    experience_score = min(len(details["Experience"]) * 5, 30)  # 5 points per experience, max 30
    score += experience_score

    # Score for leadership (up to 30%)
    leadership_score = min(len(details["Leadership"]) * 5, 30)  # 5 points per leadership, max 30
    score += leadership_score

    return score

def parse_resume_with_bert(resume_text):
    """Classifies resume text into an industry category and extracts details."""
    # Extract details
    details = extract_details_from_resume(resume_text)

    # Classify industry
    encoding = tokenizer(
        resume_text, padding="max_length", truncation=True, max_length=256, return_tensors="pt"
    )
    with torch.no_grad():
        output = model(encoding["input_ids"], attention_mask=encoding["attention_mask"])
        prediction = torch.argmax(output.logits, dim=1).item()
    details["Industry"] = label_mapping[prediction]

    # Calculate score
    details["Score"] = calculate_score(details)

    return details