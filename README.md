# Replication-and-Extension-of-MAC
This is the repository for final project of DS-GA 1011: Natural Language Processing with Representation Learning
## Replication and Extension of *Hierarchical Multi-head Attentive Network for Evidence-aware Fake News Detection*

## Usage
### 1. Install required packages
We use Pytorch 1.13.0 and python 3.8.6.
```
pip install -r requirements.txt
```
### 2. Data Preprocessing
1. download pre-trained word embeddings file

We use [glove.840B.300d](https://github.com/stanfordnlp/GloVe) ([Downloading](https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip)) with a dimension of 300 as pre-trained word embeddings.

2. convert `json` file into `pkl`

For PolitiFact dataset, run
```
python preprocess.py --dataset="PolitiFact" --embeddings_file="glove.840B.300d.txt"
```

For Snopes dataset, run
```
python preprocess.py --dataset="Snopes" --embeddings_file="glove.840B.300d.txt"
```

For FakeNewsNet+ dataset, run
```
python preprocess_tweet.py --dataset='FakeNewsNet_plus' --embeddings_file="glove.840B.300d.txt"
```

### 3. Running experiment
For PolitiFact dataset, run
```
python mac.py --dataset="PolitiFact" --use_post_sources --use_article_sources --mac_nhead_1=3 --mac_nhead_2=1
```

For Snopes dataset, run
```
python mac.py --dataset="Snopes" --use_post_sources --use_article_sources --mac_nhead_1=5 --mac_nhead_2=2
```

For FakeNewsNet+ dataset, run
```
python mac_tweet.py --dataset='FakeNewsNet_plus'
```
