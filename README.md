# MSI5001 Group Project
This repository has been created fot MSI5001 Group Project work.
Dataset Description
The central dogma of molecular biology states that DNA is transcribed into RNA, and RNA is then translated into proteins. We call these RNAs, messenger RNAs (mRNAs). Nevertheless, recent studies have shown that RNAs are much more versatile, serving to inhibit certain enzymes if a certain criteria is met, etc. In this dataset, you are tasked to classify RNAs into messenger RNAs and those that aren't.
Dataset Details
⦁	The dataset consists of three files
(1) training.fa (The training fasta fiiles)
(2) training_class.csv
(3) test.csv
⦁	The fasta file is a text file consisting of multiple sequences. Each sequence begins with a ">" followed by the sequence ID in a single line. All the following line is the sequence string.
⦁	A lot more negative sequences compared to positive sequences in the training set
⦁	Class 0 => Not a messenger RNA
⦁	Class 1 -> Messenger RNA
⦁	While all sequences in the test dataset consists of only 4 types of letters (i.e., "A" for adenine, "U" for urasil, "G" for guanine, "C" for cytosine), that is not the case in the training dataset due to experimental errors. When the type of nucleotide could not be distinguished accurately. (e.g., It could be a adenine or guanine, it's represented as a different letter "R") (https://www.bioinformatics.org/sms/iupac.html)
Expected Task Description
⦁	You need to train and tune your model using train.fa
⦁	Finally, you need to test on the test.csv
⦁	As test performance metric, you need use sensitivity, specificity and MCC score
⦁	Remember to explore:
⦁	character level language models
⦁	consider positional embeddings
⦁	various feature extraction techniques
⦁	class balancing methods during training