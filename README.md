# mRNA Classification Using Machine Learning

**Project:** MSI5001 Introduction to AI - Team 15  
**GitHub:** https://github.com/Team15/mRNA-Classification  

---

## ⚡ Quick Start (30 seconds)

git clone https://github.com/Team15/mRNA-Classification.git
cd mRNA-Classification
pip install -r requirements.txt
python train_models.py --model all
cat outputs/model_performance.csv


**Expected Runtime:** 15 min (CPU)  
**Expected Accuracy:** Random Forest 81.63% ✓

---

## 📁 Repository Structure

data/ # Pre-processed 4-mer features
├── kmer_4_train.csv # 18,954 sequences (balanced)
└── kmer_4_test.csv # Test set

data_preprocessing/ # Feature engineering
model_training/ # Model implementations
├── random_forest_mrna.pkl # ★ Best model
├── lstm_kmer_model.pth # LSTM checkpoint
└── rnn_tokenisation.pth # RNN checkpoint

result/ # Outputs
├── model_performance.csv
└── confusion_matrices/

notebooks/
├── MSI5001_Team15_mRNA.ipynb # Complete pipeline
└── Jupyter-Test.ipynb

requirements.txt
README.md


---

## 📊 Expected Results

Running the pipeline produces:

Model Accuracy F1-Score ROC-AUC
─────────────────────────────────────────────
Logistic Reg 67.75% 0.68 0.7050
Random Forest ★ 81.63% 0.8309 0.8926 ← BEST
RNN 69.84% 0.71 -0.70
LSTM 81.57% 0.82 0.8943

✓ Results saved to: outputs/model_performance.csv


---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| ModuleNotFoundError | `pip install -r requirements.txt` |
| No dataset file | Check `ls data/` — should see kmer_4_*.csv |
| CUDA memory error | `export CUDA_VISIBLE_DEVICES=""` |
| Metrics differ ±0.5% | Normal variance; seed=42 ensures reproducibility |

---

## ✅ Verification Checklist

After running, verify:
- [ ] `outputs/model_performance.csv` exists
- [ ] Random Forest accuracy ≥ 80%
- [ ] All 4 models trained
- [ ] Confusion matrices generated
- [ ] Runtime < 30 minutes

---

## 📝 Project Notes

- **Best Model:** Random Forest (81.63% accuracy, 1/7000 parameters vs LSTM)
- **Feature Representation:** 4-mer k-mer encoding captures codon bias
- **Training:** 5-fold stratified cross-validation
- **Dataset Balance:** SMOTE applied (9,477 mRNA + 9,477 non-mRNA)

See report for full analysis: `MSI5001_Team15_mRNAClassification_Report.pdf`


# Dataset Description
The central dogma of molecular biology states that DNA is transcribed into RNA, and RNA is then translated into proteins. We call these RNAs, messenger RNAs (mRNAs). Nevertheless, recent studies have shown that RNAs are much more versatile, serving to inhibit certain enzymes if a certain criteria is met, etc. In this dataset, you are tasked to classify RNAs into messenger RNAs and those that aren't.

# Dataset Details
- The dataset consists of three files 
    (1) training.fa (The training fasta fiiles)
    (2) training_class.csv
    (3) test.csv
- The fasta file is a text file consisting of multiple sequences. Each sequence begins with a ">" followed by the sequence ID in a single line. All the following line is the sequence string.
- A lot more negative sequences compared to positive sequences in the training set
- Class 0 => Not a messenger RNA
- Class 1 -> Messenger RNA
- While all sequences in the test dataset consists of only 4 types of letters (i.e., "A" for adenine, "U" for urasil, "G" for guanine, "C" for cytosine), that is not the case in the training dataset due to experimental errors. When the type of nucleotide could not be distinguished accurately. (e.g., It could be a adenine or guanine, it's represented as a different letter "R") (https://www.bioinformatics.org/sms/iupac.html)

# Expected Task Description
- You need to train and tune your model using train.fa
- Finally, you need to test on the test.csv
- As test performance metric, you need use sensitivity, specificity and MCC score
- Remember to explore:
    - character level language models
    - consider positional embeddings 
    - various feature extraction techniques
    - class balancing methods during training
