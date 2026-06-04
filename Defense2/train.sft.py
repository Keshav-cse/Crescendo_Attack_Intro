import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import confusion_matrix, classification_report

model_name = "distilbert-base-uncased"
data = Path("./data") #dir
sav = Path("./model") #dir

EPOCHS = 5
BATCH_SIZE = 12
MAX_LEN = 256
LR = 5e-5
DEVICE = "cpu"
class JailbreakDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_len: int):
        self.texts = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
 
    def __len__(self):
        return len(self.texts)
 
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }
 

def load_dataset_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        if len(df.columns) >= 2:
            text_col = df.columns[0]
            label_col = df.columns[1]
            df = df.rename(columns={text_col: "text", label_col: "label"})
            print(f"[WARN] {path}: renamed columns {text_col!r} -> 'text', {label_col!r} -> 'label'")
        else:
            raise ValueError(
                f"Dataset at {path} must contain 'text' and 'label' columns, or at least two columns to infer them."
            )
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Dataset at {path} must contain 'text' and 'label' columns after renaming. "
            f"Found columns: {list(df.columns)}"
        )
    return df[["text", "label"]]


def train():
    df = load_dataset_csv(data / "train.csv")
    pf = load_dataset_csv(data / "val.csv")
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(DEVICE)

    train_dataset = JailbreakDataset(df, tokenizer, MAX_LEN)
    val_dataset = JailbreakDataset( pf,tokenizer, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE) 

    optimizer = AdamW(model.parameters(), lr=LR) 
    total_steps = len(train_loader) * EPOCHS 
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps//10, num_training_steps=total_steps) 
    best_val_acc = 0

    for epoch in range(EPOCHS):
        model.train()
        totalloss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            output = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels = labels
            )

            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #backpropagation means 
            optimizer.step()
            scheduler.step()
            totalloss += loss.item()
        avg_loss = totalloss / len(train_loader)
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                output = model(
                    input_ids = input_ids,
                    attention_mask = attention_mask
                )

                preds = torch.argmax(output.logits, dim=1).numpy() 
                all_preds.extend(preds)
                all_labels.extend(labels)
            val_acc = np.mean(np.array(all_preds) == np.array(all_labels)) #it shows accuracy by 
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model.save_pretrained(sav)
                tokenizer.save_pretrained(sav)
        print(f"Done and best accuracy:{best_val_acc} ")
        
        test = load_dataset_csv(data / "test.csv")
        test_dataset = JailbreakDataset(test, tokenizer, MAX_LEN)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        model.eval()
        all_preds, all_labels = [], []  
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"].to(DEVICE)

                output = model(
                    input_ids = input_ids,
                    attention_mask = attention_mask
                )

                preds = torch.argmax(output.logits, dim=1).numpy() 
                all_preds.extend(preds)
                all_labels.extend(labels)

            print(classification_report(all_labels, all_preds, target_names=["safe", "jailbreak"]))
            
            print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    train()

