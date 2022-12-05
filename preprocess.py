import os
from tqdm import tqdm
import pickle
import numpy as np
from argparse import ArgumentParser
import nltk
import torch


def load_embeddings(embeddings_file, save_dir):
    embeddings_index = {}
    with open(embeddings_file, 'r') as f:
        lines = f.readlines()
        lines = [l.rstrip() for l in lines]
        for line in tqdm(lines):
            word, coefs = line.split(' ', maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    print('File: {}, there are {} vectors'.format(
        embeddings_file, len(embeddings_index)))

    word2idx = {k: i + 1 for i, k in enumerate(embeddings_index)}
    idx2word = {v: k for k, v in word2idx.items()}
    D = len(embeddings_index[idx2word[1]])

    embedding_weight = torch.zeros(1 + len(word2idx), D, dtype=torch.float)
    for idx, word in idx2word.items():
        embedding_weight[idx] = torch.as_tensor(embeddings_index[word])
    torch.save(embedding_weight, os.path.join(save_dir, 'embedding_weight.pt'))
    del embeddings_index, embedding_weight

    return word2idx, idx2word


def text2token(pieces, word2idx):
    pieces_words = [nltk.word_tokenize(p.lower())
                    for p in pieces]
    pieces_tokens = [[word2idx.get(w, 0) for w in words]
                     for words in pieces_words]
    return pieces_tokens


def init_dataset(file):
    with open(file, 'r') as f:
        pieces = f.readlines()

    datasets = dict()
    for piece in pieces[1:]:
        claim_id, label, _, claim, claim_source, _, evidence, evidence_source = piece.strip().split('\t')
        if claim_id in datasets:
            datasets[claim_id]['evidence'].append(evidence)
            datasets[claim_id]['evidence_source'].append(evidence_source)
        else:
            datasets[claim_id] = {'claim': claim, 'claim_source': claim_source,
                                  'evidence': [evidence], 'evidence_source': [evidence_source],
                                  'label': label}
    return datasets


def get_post_source(pieces):
    sources = set([pieces[p]['claim_source'] for p in pieces])
    source2idx = {k: i + 1 for i, k in enumerate(sources)}
    return source2idx


def get_article_source(pieces):
    sources = list()
    for p in pieces:
        sources += pieces[p]['evidence_source']
    sources = set(sources)
    source2idx = {k: i + 1 for i, k in enumerate(sources)}
    return source2idx


if __name__ == "__main__":
    parser = ArgumentParser(description='Tokenize by NLTK')
    parser.add_argument('--embeddings_file', type=str, default='glove.840B.300d.txt')
    parser.add_argument('--dataset', default='PolitiFact', choices=['PolitiFact', 'Snopes'])
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of folds for cross validation')
    args = parser.parse_args()

    print("Load embedding from {}...".format(args.embeddings_file))
    args.save_dir = 'dataset/{}/pkl'.format(args.dataset)
    word2idx, idx2word = load_embeddings(args.embeddings_file, args.save_dir)

    print("Load done.")

    for i in range(args.num_folds):
        print('Load fold{}...'.format(i))
        for t in ['train', 'val', 'test']:
            print("Load {}_dataset...".format(t))
            file = 'dataset/{}/mapped_data/5fold/{}_{}.tsv'.format(args.dataset, t, i)
            dataset = init_dataset(file)

            posts = text2token([dataset[p]['claim'] for p in dataset], word2idx)
            with open(os.path.join(args.save_dir, '{}_post_{}.pkl'.format(t, i)), 'wb') as f:
                pickle.dump(posts, f)

            post_articles = dict()
            for post_idx, p in enumerate(dataset):
                articles = dataset[p]['evidence']
                article_tokens = text2token(articles, word2idx)
                post_articles[post_idx] = article_tokens

            with open(os.path.join(args.save_dir, '{}_article_{}.pkl'.format(t, i)), 'wb') as f:
                pickle.dump(post_articles, f)

            print("Successfully save pkl file.")
