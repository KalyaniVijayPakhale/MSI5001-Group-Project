import pandas as pd
from collections import Counter

# Function to parse FASTA files
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
                seq += line.strip()
        if seq_id:
            seqs[seq_id] = seq
    return seqs

# Load FASTA sequences
fasta_path = r'dataset\training.fa'
seq_dict = load_fasta(fasta_path)

# Load class df (fixing column names: name, class)
csv_path = r'dataset\training_class.csv'
df = pd.read_csv(csv_path)

# Rename columns if needed (optional safety check)
if 'name' in df.columns:
    df = df.rename(columns={'name': 'id', 'class': 'label'})

# Map sequences from FASTA file to the dataframe using 'id'
df['sequence'] = df['id'].map(seq_dict)

# Drop rows where sequence is missing (in case of mismatches)
df = df.dropna(subset=['sequence'])

# Function to extract k-mer counts from a sequence
def kmer_counts(seq, k=3):
    return Counter([seq[i:i+k] for i in range(len(seq)-k+1)])

# Apply k-mer function to each sequence
df['kmer_feat'] = df['sequence'].apply(lambda x: kmer_counts(x, k=3))

# Preview the result
print(df.head())



df.to_csv(r'dataset\\df.csv')