import train_transformer
from transformers import BertForSequenceClassification, BertConfig
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef

model = BertForSequenceClassification.from_pretrained('./rna-transformer/final_model')
tokenizer = SimpleCharTokenizer.from_pretrained('./rna-transformer/final_model')

trainer.save_model('./rna-transformer/final_model')
tokenizer.save_pretrained('./rna-transformer/final_model')


# Load your test predictions and true labels
df = pd.read_csv('test_predictions.csv')
# Suppose your CSV has columns: 'sequence', 'predicted_label', 'true_label'

# Calculate metrics
y_true = df['true_label']
y_pred = df['predicted_label']

print("Accuracy:", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, average='binary'))
print("Recall:", recall_score(y_true, y_pred, average='binary'))
print("F1:", f1_score(y_true, y_pred, average='binary'))
print("MCC:", matthews_corrcoef(y_true, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
