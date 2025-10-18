import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
from transformers import (
    PreTrainedTokenizerFast,
    BertConfig,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
import string
import json, os
# ------------------------------
# 1. Parse FASTA
# ------------------------------
def load_fasta(path):
    seqs = {}
    with open(path) as f:
        seq_id = None
        seq = ''
        for line in f:
            if line.startswith('>'):
                if seq_id:
                    seqs[seq_id] = seq
                seq_id = line[1:].strip()
                seq = ''
            else:
                seq += line.strip().upper()
        if seq_id:
            seqs[seq_id] = seq
    return seqs


# ------------------------------
# 2. Load data
# ------------------------------
fasta_path = r'dataset\training.fa'
labels_path = r'dataset\training_class.csv'

seq_dict = load_fasta(fasta_path)
df = pd.read_csv(labels_path, names=['id', 'label'])  # assuming no header in CSV
df['sequence'] = df['id'].map(seq_dict)
df = df.dropna()

# Ensure labels are numeric
df['label'] = pd.Categorical(df['label']).codes
df['label'] = df['label'].astype(int)


print(f"Loaded {len(df)} sequences.")

# ------------------------------
# 3. Train/val split
# ------------------------------
train_df, val_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)


# ------------------------------
# 4. Tokenizer (character-level)
# ------------------------------
all_chars = set(''.join(df['sequence'].values))
char_vocab = sorted(list(all_chars))
vocab_dict = {ch: idx + 2 for idx, ch in enumerate(char_vocab)}  # +2 to reserve 0 (pad), 1 (unk)
vocab_dict['[PAD]'] = 0
vocab_dict['[UNK]'] = 1

class SimpleCharTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.ids_to_tokens = {i: t for t, i in vocab.items()}
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.pad_token_id = self.vocab[self.pad_token]
        self.unk_token_id = self.vocab[self.unk_token]

    def encode(self, text, max_length=512, padding='max_length', truncation=True):
        tokens = [self.vocab.get(ch, self.unk_token_id) for ch in text]
        if truncation:
            tokens = tokens[:max_length]
        if padding == 'max_length':
            tokens = tokens + [self.pad_token_id] * max(0, max_length - len(tokens))
        return tokens

    def decode(self, token_ids):
        return ''.join([self.ids_to_tokens.get(i, self.unk_token) for i in token_ids])
    
    def save_pretrained(self, save_directory):
        """Save vocab to a directory (so Trainer can checkpoint)."""
        
        os.makedirs(save_directory, exist_ok=True)
        vocab_path = os.path.join(save_directory, "vocab.json")
        with open(vocab_path, "w") as f:
            json.dump(self.vocab, f)
        print(f"Tokenizer saved to {vocab_path}")
    @classmethod
    def from_pretrained(cls, load_directory):
        """Load vocab from a saved directory."""
        import json, os
        vocab_path = os.path.join(load_directory, "vocab.json")
        with open(vocab_path, "r") as f:
            vocab = json.load(f)
        print(f"✅ Tokenizer loaded from {vocab_path}")
        return cls(vocab)



# Use this tokenizer
tokenizer = SimpleCharTokenizer(vocab_dict)


# ------------------------------
# 5. Dataset Classes
# ------------------------------
class RNASequenceDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        seq = self.data.loc[idx, 'sequence']
        label = self.data.loc[idx, 'label']
        tokens = self.tokenizer.encode(seq, max_length=self.max_len, padding='max_length', truncation=True)
        attn_mask = [1 if t != self.tokenizer.pad_token_id else 0 for t in tokens]
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attn_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class RNATestDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=512):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data.loc[idx, 'sequence']
        tokens = [self.tokenizer.vocab.get(ch, 1) for ch in seq[:self.max_len]]
        pad_len = self.max_len - len(tokens)
        tokens = tokens + [0] * pad_len
        attn_mask = [1 if t != 0 else 0 for t in tokens]
        return {
            'input_ids': torch.tensor(tokens, dtype=torch.long),
            'attention_mask': torch.tensor(attn_mask, dtype=torch.long)
        }


# ------------------------------
# 6. Model Config
# ------------------------------
vocab_size = len(vocab_dict)
config = BertConfig(
    vocab_size=vocab_size,
    hidden_size=128,
    num_hidden_layers=4,
    num_attention_heads=4,
    intermediate_size=256,
    max_position_embeddings=512,
    num_labels=2
)

model = BertForSequenceClassification(config)


# ------------------------------
# 7. Prepare datasets
# ------------------------------
train_dataset = RNASequenceDataset(train_df, tokenizer)
val_dataset = RNASequenceDataset(val_df, tokenizer)


# ------------------------------
# 8. Training arguments
# ------------------------------

training_args = TrainingArguments(
    output_dir='./rna-transformer',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    logging_dir='./logs',
    logging_steps=100,      # log every 100 steps
    save_total_limit=2,
    do_eval=True,           # enable evaluation
    do_train=True,          # enable training
)

# ------------------------------
# 9. Compute metrics
# ------------------------------
def compute_metrics(p):
    preds = p.predictions.argmax(axis=-1)
    labels = p.label_ids

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='binary')
    recall = recall_score(labels, preds, average='binary')
    f1 = f1_score(labels, preds, average='binary')
    mcc = matthews_corrcoef(labels, preds)

    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'mcc': mcc
    }


# ------------------------------
# 10. Trainer setup
# ------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)


# ------------------------------
# 11. Train
# ------------------------------
trainer.train()


# ------------------------------
# 12. Evaluate
# ------------------------------
eval_results = trainer.evaluate()
print("\n Evaluation Results:")
for k, v in eval_results.items():
    print(f"{k}: {v:.4f}")


# ------------------------------
# 13. Test predictions
# ------------------------------
test_df = pd.read_csv(r'dataset\test.csv')

# If test.csv has only sequences
if 'sequence' not in test_df.columns:
    test_df.columns = ['sequence']

test_dataset = RNATestDataset(test_df, tokenizer)
predictions = trainer.predict(test_dataset)
predicted_labels = predictions.predictions.argmax(axis=1)

test_df['predicted_label'] = predicted_labels
test_df.to_csv('result\test_predictions.csv', index=False)
print("Predictions saved to 'test_predictions.csv'")
