import pandas as pd
from collections import Counter

# Parse FASTA
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

# Load data
seq_dict = load_fasta('dataset\training.fa')
labels = pd.read_csv('dataset\training_class.csv')
labels['sequence'] = labels['id'].map(seq_dict)

# Example: k-mer feature extraction (k=3)
def kmer_counts(seq, k=3):
    return Counter([seq[i:i+k] for i in range(len(seq)-k+1)])

labels['kmer_feat'] = labels['sequence'].apply(lambda x: kmer_counts(x, k=3))
