import os
import time
from tqdm import tqdm
import pickle
import random
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import RandomSampler, SequentialSampler, DataLoader, Dataset
from torch.optim import Adam
from sklearn.metrics import classification_report, roc_auc_score

INDEX_OF_LABEL = {'true': 0, 'false': 1}
INDEX2LABEL = ['true', 'false']


def config():
    parser = ArgumentParser(description='MAC')

    # ======================== Dataset ========================

    parser.add_argument('--dataset', type=str, default='PolitiFact',
                        choices=['PolitiFact', 'Snopes'])
    parser.add_argument('--num_folds', type=int, default=5,
                        help='Number of folds for cross validation')
    parser.add_argument('--save', type=str, default='result',
                        help='folder to save the final model')
    parser.add_argument('--embeddings_file', type=str, default='glove.840B.300d.txt',
                        choices=['glove.840B.300d.txt'])
    parser.add_argument('--unfreeze_embedding', action='store_true')
    parser.add_argument('--folder', type=str, default='')

    # ======================== Framework ========================

    parser.add_argument('--relevant_articles_num', type=int, default=30)
    parser.add_argument('--use_post_sources', action='store_true')
    parser.add_argument('--use_article_sources', action='store_true')
    parser.add_argument('--mac_input_max_sequence_length', type=int, default=30)
    parser.add_argument('--mac_max_doc_length', type=int, default=100)
    parser.add_argument('--mac_input_dim', type=int, default=300)
    parser.add_argument('--mac_hidden_dim', type=int, default=300)
    parser.add_argument('--mac_one_hot', type=int, default=128)
    parser.add_argument('--drop_off', action='store_true')
    parser.add_argument('--mac_dropout_doc', type=float, default=0.2)
    parser.add_argument('--mac_dropout_query', type=float, default=0.2)
    parser.add_argument('--mac_nhead_1', type=int, default=3)
    parser.add_argument('--mac_nhead_2', type=int, default=1)
    parser.add_argument('--category_num', type=int, default=2)

    # ======================== Training ========================

    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
    parser.add_argument('--early_stop', type=int, default=5,
                        help='stop training process when F1 macro '
                             'on the validation data continuously decreases')

    # ======================== Devices ========================

    parser.add_argument('--seed', type=int, default=9,
                        help='random seed')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--save_dir', default='')
    parser.add_argument('--save_model', action='store_true')

    return parser


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


class DatasetLoader(Dataset):
    def __init__(self, args, pieces, post, article):
        self.args = args

        with open(post, 'rb') as f:
            posts = pickle.load(f)
        self.posts = posts

        self.post_sources = [postSource2idx.get(pieces[p]['claim_source'], 0) for p in pieces]
        self.labels = [INDEX_OF_LABEL[pieces[p]['label']] for p in pieces]

        with open(article, 'rb') as f:
            articles = pickle.load(f)
        self.post_articles_tokens = articles
        self.post_articles_masks = dict()
        self.post_articles_sources = dict()
        for post_idx, p in enumerate(tqdm(pieces)):
            articles = pieces[p]['evidence']
            sz = len(articles)

            mask = torch.zeros(sz, 1, dtype=torch.float, device=self.args.device)
            if sz != 0:
                mask[:-sz] = 1 / sz
            self.post_articles_masks[post_idx] = mask

            articles_sources = pieces[p]['evidence_source']
            articles_sources = [articleSource2idx.get(p, 0) for p in articles_sources]
            self.post_articles_sources[post_idx] = articles_sources

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = (idx, self.labels[idx])
        return sample


class MAC(nn.Module):
    '''Hierarchical Multi-head Attentive Network for Evidence-aware Fake News Detection. EACL 2021.'''
    """
        Refer to https://github.com/nguyenvo09/EACL2021/blob/9d04d8954c1ded2110daac23117de11221f08cc6/Models/FCWithEvidences/hierachical_multihead_attention.py
    """

    def __init__(self, args):
        super(MAC, self).__init__()
        self.args = args

        self.embeddings_file = args.embeddings_file
        self.relevant_articles_num = args.relevant_articles_num
        self.use_post_sources = args.use_post_sources
        self.use_article_sources = args.use_article_sources
        self.max_sequence_length = args.mac_input_max_sequence_length
        self.max_doc_len = args.mac_max_doc_length
        self.input_dim = args.mac_input_dim
        self.num_post_source = args.num_post_source
        self.num_article_source = args.num_article_source
        self.one_hot_dim = args.mac_one_hot
        self.hidden_size = args.mac_hidden_dim
        self.drop_off = args.drop_off
        self.dropout_doc = args.mac_dropout_doc
        self.dropout_query = args.mac_dropout_query
        self.num_heads_1 = args.mac_nhead_1
        self.num_heads_2 = args.mac_nhead_2
        self.num_layers = 1
        self.category_num = args.category_num

        weight = torch.load(args.folder + 'dataset/{}/pkl/embedding_weight.pt'.format(args.dataset))
        weight.to(args.device)
        self.embedding = nn.Embedding.from_pretrained(weight)
        if args.unfreeze_embedding:
            self.embedding.requires_grad = True
        self.input_dim = weight.shape[1]

        self.post_source_emb = nn.Embedding(self.num_post_source, self.one_hot_dim)
        nn.init.uniform_(self.post_source_emb.weight, -0.2, 0.2)
        self.article_source_emb = nn.Embedding(self.num_article_source, self.one_hot_dim)
        nn.init.uniform_(self.article_source_emb.weight, -0.2, 0.2)

        self.doc_bilstm = nn.LSTM(self.input_dim, self.hidden_size, self.num_layers,
                                  bidirectional=True, batch_first=True, dropout=self.dropout_doc)
        self.query_bilstm = nn.LSTM(self.input_dim, self.hidden_size, self.num_layers,
                                    bidirectional=True, batch_first=True, dropout=self.dropout_query)

        self.W1 = nn.Linear(4 * self.hidden_size, 2 *
                            self.hidden_size, bias=False)
        self.W2 = nn.Linear(2 * self.hidden_size, self.num_heads_1, bias=False)

        w3_dim = (self.num_heads_1 + 1) * 2 * self.hidden_size
        if self.use_post_sources:
            w3_dim += self.one_hot_dim
        if self.use_article_sources:
            w3_dim += self.one_hot_dim
        self.W3 = nn.Linear(w3_dim, 2 * self.hidden_size, bias=False)

        self.W4 = nn.Linear(2 * self.hidden_size, self.num_heads_2, bias=False)

        self.query_features_dropout = nn.Dropout(self.dropout_query)
        self.doc_features_dropout = nn.Dropout(self.dropout_doc)

        last_output_dim = 2 * self.hidden_size * (1 + self.num_heads_1 * self.num_heads_2)
        if self.use_post_sources:
            last_output_dim += self.one_hot_dim
        if self.use_article_sources:
            last_output_dim += self.one_hot_dim * self.num_heads_2
        self.W5 = nn.Linear(last_output_dim, 2 * self.hidden_size, bias=True)

        self.W6 = nn.Linear(2 * self.hidden_size, self.category_num, bias=True)

    def forward(self, idxs, dataset):
        # ====== Input ======
        # --- Post ---
        post_inputs = [self._encode(dataset.posts[idx.item()],
                                    max_len=self.max_sequence_length) for idx in idxs]

        # (bs, max_len)
        post_input_ids = torch.tensor(
            [i[0] for i in post_inputs], dtype=torch.long, device=self.args.device)
        # (bs, max_len, 1)
        post_masks = torch.stack([i[1] for i in post_inputs])
        # (bs, max_len, D)
        post = self.embedding(post_input_ids)

        # (bs, 1)
        speaker_inputs = [dataset.post_sources[idx.item()] for idx in idxs]
        speaker_inputs_ids = torch.tensor(speaker_inputs, dtype=torch.long, device=self.args.device)
        # (bs, D1)
        speaker = self.post_source_emb(speaker_inputs_ids)

        # --- Articles ---
        article_inputs_ids = []
        article_masks = []
        article_sources = []
        article_sources_masks = []
        for idx in idxs:
            # (#doc, max_doc_len)
            inputs = [self._encode(doc, max_len=self.max_doc_len)
                      for doc in dataset.post_articles_tokens[idx.item()]]
            input_ids = torch.tensor(
                [i[0] for i in inputs], dtype=torch.long, device=self.args.device)
            padding_length = self.relevant_articles_num - len(inputs)
            input_padding = torch.zeros(padding_length, self.max_doc_len, dtype=torch.long, device=self.args.device)
            input_ids = torch.cat((input_ids, input_padding), dim=0)

            # (#doc, max_doc_len, 1)
            masks = torch.stack([i[1] for i in inputs])
            mask_padding = torch.zeros(padding_length, self.max_doc_len, 1, dtype=torch.float, device=self.args.device)
            masks = torch.cat((masks, mask_padding), dim=0)

            # (#doc)
            sources = dataset.post_articles_sources[idx.item()]
            sources_ids = torch.tensor(sources, dtype=torch.long, device=self.args.device)
            source_padding = torch.zeros(padding_length, dtype=torch.long, device=self.args.device)
            sources_ids = torch.cat((sources_ids, source_padding), dim=0)

            # (#doc, 1)
            sources_mask = dataset.post_articles_masks[idx.item()].view(-1, 1)
            sources_mask = torch.ones_like(sources_mask)
            sources_mask = torch.cat((sources_mask, source_padding.view(-1, 1)), dim=0)

            article_inputs_ids.append(input_ids)
            article_masks.append(masks)
            article_sources.append(sources_ids)
            article_sources_masks.append(sources_mask)

        # (bs, #doc, max_doc_len)
        article_inputs_ids = torch.stack(article_inputs_ids)
        # (bs, #doc, max_doc_len, 1)
        article_masks = torch.stack(article_masks)
        # (bs, #doc)
        article_sources = torch.stack(article_sources)
        # (bs, #doc, 1)
        article_sources_masks = torch.stack(article_sources_masks)

        # (bs, #doc, max_doc_len, D)
        articles = self.embedding(article_inputs_ids)
        # (bs, #doc, D2)
        article_sources = self.article_source_emb(article_sources)
        article_sources = article_sources.masked_fill((article_sources_masks == 0), 0)

        # ====== Forward ======

        # post: (bs, max_len, D)
        # post_masks: (bs, max_len, 1)
        # speakers: (bs, D1)
        # articles: (bs, #doc, max_doc_len, D)
        # article_masks: (bs, #doc, max_doc_len, 1)
        # publishers: (bs, #doc, D2)

        # (bs, max_len, 2H)
        query_hiddens, _ = self.query_bilstm(post)
        # (bs, 2H)
        query_repr = torch.sum(query_hiddens * post_masks, dim=1)
        if self.drop_off:
            query_repr = self.query_features_dropout(query_repr)

        df_sizes = articles.size()
        # (bs * #doc, max_doc_len, D)
        doc_hiddens = articles.view(-1, df_sizes[-2], df_sizes[-1])
        # (bs * #doc, max_doc_len, 2H)
        doc_hiddens, _ = self.doc_bilstm(doc_hiddens)
        # (bs, #doc, max_doc_len, 2H)
        doc_hiddens = doc_hiddens.view(
            df_sizes[0], df_sizes[1], df_sizes[2], doc_hiddens.size()[-1])
        if self.drop_off:
            doc_hiddens = self.doc_features_dropout(doc_hiddens)

        # ---------- Multi-head Word Attention Layer ----------
        C1 = query_repr.unsqueeze(1).unsqueeze(1).repeat(
            1, doc_hiddens.shape[1], doc_hiddens.shape[2], 1)  # [batch_size, #doc, doc_len, hidden_size * 2]
        # [batch_size, #doc, doc_len, hidden_size*4]
        A1 = torch.cat((doc_hiddens, C1), dim=-1)
        # [batch_size, #doc, doc_len, head_num_1]
        A1 = self.W2(torch.tanh(self.W1(A1)))

        # exclude the padding words in each doc
        A1 = F.softmax(A1, dim=-2)  # [batch_size, #doc, doc_len, head_num_1]
        A1 = A1.masked_fill((article_masks == 0), 0)

        # [batch_size * #doc, doc_len, head_num_1]
        A1_tmp = A1.reshape(-1, A1.shape[-2], A1.shape[-1])
        # [batch_size * #doc, doc_len, hidden_size * 2]
        doc_hiddens_tmp = doc_hiddens.reshape(-1, doc_hiddens.shape[-2], doc_hiddens.shape[-1])
        # [batch_size*#doc, head_num_1, doc_len] * [batch_size*#doc, doc_len, hidden_size * 2]
        D = torch.bmm(A1_tmp.permute(0, 2, 1), doc_hiddens_tmp)
        # [batch_size, #doc, head_num_1 * hidden_size * 2]
        D = D.view(A1.shape[0], A1.shape[1], -1)

        # ---------- Multi-head Document Attention Layer ----------
        if self.use_post_sources:
            # [batch_size, hidden_size * 2 + one_hot_dim]
            query_repr = torch.cat((query_repr, speaker), dim=-1)
        if self.use_article_sources:
            # [batch_size, #doc, head_num_1 * hidden_size * 2 + one_hot_dim]
            D = torch.cat((D, article_sources), dim=-1)
        # [batch_size, #doc, hidden_size * 2]
        C2 = query_repr.unsqueeze(1).repeat(1, D.shape[1], 1)
        # [batch_size, #doc, (head_num_1 + 1) * hidden_size * 2]
        A2 = torch.cat((D, C2), dim=-1)
        A2 = self.W4(torch.tanh(self.W3(A2)))  # [batch_size, #doc, head_num_2]

        # [batch_size, #doc, 1]
        A2 = F.softmax(A2, dim=-2)  # [batch_size, #doc, head_num_2]
        # A2 = A2.masked_fill((article_num_masks == 0), 0)

        # [batch_size, #doc, head_num_2] * [batch_size, #doc, head_num_1 * hidden_size * 2]
        D = torch.bmm(A2.permute(0, 2, 1), D)
        # [batch_size, head_num_2 * head_num_1 * hidden_size * 2] Eq.(9)
        D = D.view(D.shape[0], -1)

        # Output Layer
        # [batch_size, (head_num_2 * head_num_1 * 2 + 2) * hidden_size]
        output = torch.cat((query_repr, D), dim=-1)
        # (bs, 2H)
        output = self.W5(output)
        # (bs, 2)
        output = self.W6(output)

        return output

    def _encode(self, doc, max_len):
        doc = doc[:max_len]

        padding_length = max_len - len(doc)
        input_ids = doc + [0] * padding_length

        mask = torch.zeros(max_len, 1, dtype=torch.float,
                           device=self.args.device)

        if len(doc) != 0:
            mask[:len(doc)] = 1 / len(doc)

        return input_ids, mask


def get_metric(outputs, labels, get_all):
    classifying_ans = []
    classifying_pred = []
    classifying_score = []

    for i in range(len(outputs)):
        # o: [idx, 0_class_score, 1_class_score]
        scores = outputs[i]
        ans = int(labels[i])
        pred = int(np.array(scores).argmax())

        classifying_ans.append(ans)
        classifying_pred.append(pred)
        classifying_score.append(scores[1])

    classifying_ans = np.array(classifying_ans)
    classifying_pred = np.array(classifying_pred)
    classifying_score = np.array(classifying_score)

    class_report = classification_report(classifying_ans, classifying_pred,
                                         target_names=INDEX2LABEL, digits=4, output_dict=True)

    if get_all:
        f1_macro = class_report['macro avg']['f1-score']
        f1_micro = class_report['accuracy']
        true = class_report['true']
        false = class_report['false']

        AUC = roc_auc_score(classifying_ans, classifying_score)

        result = {"AUC": round(AUC, 5), "f1_macro": round(f1_macro, 5), "f1_micro":  round(f1_micro, 5),
                  "true_f1":  round(true["f1-score"], 5), "true_precision":  round(true["precision"], 5),
                  "true_recall": round(true["recall"], 5), "false_f1":  round(false["f1-score"], 5),
                  "false_precision":  round(false["precision"], 5), "false_recall":  round(false["recall"], 5)}
    else:
        result = class_report['macro avg']['f1-score']

    return result


def evaluate(args, loader, model, criterion, get_all_metrics=False):
    model.eval()
    eval_loss = 0.
    outputs = []

    with torch.no_grad():
        for idxs, labels in tqdm(loader):
            out = model(idxs, loader.dataset)
            labels = labels.long().to(args.device)
            loss = criterion(out, labels)
            eval_loss += loss.item()
            score = out.cpu().numpy().tolist()
            outputs += score
        eval_loss /= len(loader)

    # compute metrics
    classification_metrics = get_metric(outputs, loader.dataset.labels, get_all_metrics)

    return eval_loss, classification_metrics, outputs


if __name__ == "__main__":
    parser = config()
    args = parser.parse_args()

    print('Experimental Dataset: {}'.format(args.dataset))
    if args.dataset == "Snopes":
        args.use_post_sources = False

    args.save = args.folder + args.save
    print('save path: ', args.save)


    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    all_labels = []
    all_test_output = []

    for i in range(args.num_folds):
        print('Loading fold{}...'.format(i))
        start = time.time()

        train_file = args.folder + 'dataset/{}/mapped_data/5fold/train_{}.tsv'.format(args.dataset, i)
        val_file = args.folder + 'dataset/{}/mapped_data/5fold/val_{}.tsv'.format(args.dataset, i)
        test_file = args.folder + 'dataset/{}/mapped_data/5fold/test_{}.tsv'.format(args.dataset, i)

        train_post = args.folder + 'dataset/{}/pkl/train_post_{}.pkl'.format(args.dataset, i)
        val_post = args.folder + 'dataset/{}/pkl/val_post_{}.pkl'.format(args.dataset, i)
        test_post = args.folder + 'dataset/{}/pkl/test_post_{}.pkl'.format(args.dataset, i)

        train_article = args.folder + 'dataset/{}/pkl/train_article_{}.pkl'.format(args.dataset, i)
        val_article = args.folder + 'dataset/{}/pkl/val_article_{}.pkl'.format(args.dataset, i)
        test_article = args.folder + 'dataset/{}/pkl/test_article_{}.pkl'.format(args.dataset, i)

        train_dataset = init_dataset(train_file)
        val_dataset = init_dataset(val_file)
        test_dataset = init_dataset(test_file)

        postSource2idx = get_post_source(train_dataset)
        args.num_post_source = len(postSource2idx) + 1
        print('there are {} post sources'.format(len(postSource2idx)))
        articleSource2idx = get_article_source(train_dataset)
        args.num_article_source = len(articleSource2idx) + 1
        print('there are {} article sources'.format(len(articleSource2idx)))

        train_dataset = DatasetLoader(args, train_dataset, train_post, train_article)
        val_dataset = DatasetLoader(args, val_dataset, val_post, val_article)
        test_dataset = DatasetLoader(args, test_dataset, test_post, test_article)

        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)
        test_sampler = SequentialSampler(test_dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            pin_memory=(torch.cuda.is_available()),
            drop_last=False,
            sampler=train_sampler)

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=(torch.cuda.is_available()),
            drop_last=False,
            sampler=val_sampler
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=(torch.cuda.is_available()),
            drop_last=False,
            sampler=test_sampler
        )

        print('Loading model...')
        model = MAC(args)
        model = model.to(args.device)
        print(model)

        criterion = nn.CrossEntropyLoss().to(args.device)
        # optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), weight_decay=1e-3, lr=args.lr)

        print('Arguments: \n')
        for arg in vars(args):
            v = getattr(args, arg)
            s = '{}\t{}'.format(arg, v)
            print(s)

        # Training
        print('Start training...')

        # Best results on validation dataset
        best_val_result = 0
        best_val_epoch = -1
        best_test_output = None

        start = time.time()
        steps = 0
        for epoch in range(args.epochs):
            args.current_epoch = epoch
            print('Start Training Epoch', epoch)
            model.train()

            train_loss = 0.

            lr = optimizer.param_groups[0]['lr']
            print_step = int(len(train_loader) / 10)
            for step, (idxs, labels) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()

                out = model(idxs, train_dataset)
                labels = labels.long().to(args.device)
                loss = criterion(out, labels)

                if step % print_step == 0:
                    print('\nEpoch: {}, Step: {}, CELoss = {:.4f}'.format(
                        epoch, step, loss.item()))

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            val_loss, val_result, _ = evaluate(args, val_loader, model, criterion)
            test_loss, test_result, output = evaluate(args, test_loader, model, criterion)

            print('Epoch: {}/{}'.format(epoch, args.epochs), 'lr: {}'.format(lr))
            print('[Loss]\nTrain: {:.6f}\tVal: {:.6f}\tTest: {:.6f}'.format(
                train_loss, val_loss, test_loss))
            print('[Macro F1]\nVal: {:.6f}\tTest: {:.6f}\n'.format(
                val_result, test_result))

            if val_result >= best_val_result:
                steps = 0
                if args.save_model:
                    torch.save({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()
                    },
                        os.path.join(args.save, 'fold{}_{}.pt'.format(i, epoch))
                    )

                    if best_val_epoch != -1:
                        os.system('rm {}'.format(os.path.join(
                            args.save, 'fold{}_{}.pt'.format(i, best_val_epoch))))

                best_val_result = val_result
                best_val_epoch = epoch
                best_test_output = output
            else:
                steps += 1

            if steps > args.early_stop:
                break

        print('Training Time: {:.2f}s'.format(time.time() - start))

        all_labels += test_dataset.labels
        all_test_output += best_test_output
    final_result = get_metric(all_test_output, all_labels, get_all=True)
    for key, value in final_result.items():
        print(key, value)
