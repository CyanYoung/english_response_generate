import json
import pickle as pk

import numpy as np

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


embed_len = 200
max_vocab = 10000
seq_len = 100

bos, eos = '*', '#'

path_word_vec = 'feat/word_vec.pkl'
path_word2ind = 'model/word2ind.pkl'
path_embed = 'feat/embed.pkl'


def add_flag(texts):
    flag_texts = list()
    for text in texts:
        flag_texts.append(' '.join([bos, text, eos]))
    return flag_texts


def shift(flag_texts):
    sents = [text[:-2] for text in flag_texts]
    labels = [text[2:] for text in flag_texts]
    return sents, labels


def tokenize(texts, path):
    model = Tokenizer(num_words=max_vocab, filters='', lower=True, oov_token='oov')
    model.fit_on_texts(texts)
    with open(path, 'wb') as f:
        pk.dump(model, f)


def embed(path_word2ind, path_word_vec, path_embed):
    with open(path_word2ind, 'rb') as f:
        model = pk.load(f)
    word_inds = model.word_index
    with open(path_word_vec, 'rb') as f:
        word_vecs = pk.load(f)
    vocab = word_vecs.keys()
    vocab_num = min(max_vocab + 1, len(word_inds) + 1)
    embed_mat = np.zeros((vocab_num, embed_len))
    for word, ind in word_inds.items():
        if word in vocab:
            if ind < max_vocab:
                embed_mat[ind] = word_vecs[word]
    with open(path_embed, 'wb') as f:
        pk.dump(embed_mat, f)


def align(sents, path_sent, mode):
    with open(path_word2ind, 'rb') as f:
        model = pk.load(f)
    seqs = model.texts_to_sequences(sents)
    if mode == 'decode':
        pad_seqs = pad_sequences(seqs, maxlen=seq_len, padding='post', truncating='post')
    else:
        pad_seqs = pad_sequences(seqs, maxlen=seq_len, padding='pre', truncating='pre')
    with open(path_sent, 'wb') as f:
        pk.dump(pad_seqs, f)


def vectorize(paths, mode):
    with open(paths['data'], 'r') as f:
        pairs = json.load(f)
    text1s, text2s = zip(*pairs)
    sent1s, text2s = list(text1s), list(text2s)
    flag_text2s = add_flag(text2s)
    sent2s, labels = shift(flag_text2s)
    print(sent2s)
    print(labels)
    if mode == 'train':
        tokenize(sent1s + flag_text2s, path_word2ind)
        embed(path_word2ind, path_word_vec, path_embed)
    align(sent1s, paths['sent1'], 'encode')
    align(sent2s, paths['sent2'], 'decode')
    align(labels, paths['label'], 'decode')


if __name__ == '__main__':
    paths = dict()
    paths['data'] = 'data/train.json'
    paths['sent1'] = 'feat/sent1_train.pkl'
    paths['sent2'] = 'feat/sent2_train.pkl'
    paths['label'] = 'feat/label_train.pkl'
    vectorize(paths, 'train')
    paths['data'] = 'data/test.json'
    paths['sent1'] = 'feat/sent1_test.pkl'
    paths['sent2'] = 'feat/sent2_test.pkl'
    paths['label'] = 'feat/label_test.pkl'
    vectorize(paths, 'test')
