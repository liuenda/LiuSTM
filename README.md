# Cross-lingual news article comparison system using bi-graph clustering and Siamese LSTM

Desription
Given two article, in Japanese and English respectively, calculate the simialrity between them.
Basing on the paper  Mueller, J and Thyagarajan, A. Siamese Recurrent Architectures for Learning Sentence Similarity as well as the research of mine.

Data
Training data: 1000 "relative" Japanese-English news reports pairs + 1000 "unrelavant" Japanese-English news reports pairs
Testing data: 1000 Japanese individual reprots and 1000 English individual reports, generated basing on 1000 "relative" Japanese-English news reports

Files:
*.ipynb
Testing script

lstm.py
Modified basing on the description and codes from the paper  Mueller, J and Thyagarajan, A. Siamese Recurrent Architectures for Learning Sentence Similarity. Adjust for the multilingual cases.

sentences.py
Embedding the words(cluster number )appeared in the original

main.py
Core scripts. Convert the origianl news articles to the series of multilingual cluster numbers. And then generate training data and testing data basing on the rules.

./pickles
Containing preprocessed training data and testing data

./weights
Containing the trained model for MaLSTM
e5/e10 refers to the epoches times
1k1k refers to the number of "relative" training pairs (similairty =1) 1k and "unrelavant" training pairs 1k (similairty = 0)

Ignored files:
./log
Log histories

./data
Dataset and results for multilingual bi-graph clustering system (see my other paper published soon)