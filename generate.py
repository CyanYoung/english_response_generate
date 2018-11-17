import pickle as pk

import re

import numpy as np
from numpy.random import choice

from keras.models import Model
from keras.layers import Input, Embedding

from keras.preprocessing.sequence import pad_sequences

from preprocess import clean

from nn_arch import s2s_encode, s2s_decode, att_encode, att_decode

from util import load_word_re, map_item


def define_model(name, embed_mat, seq_len, mode):
    vocab_num, embed_len = embed_mat.shape
    embed = Embedding(input_dim=vocab_num, output_dim=embed_len, input_length=seq_len, name='embed')
    input = Input(shape=(seq_len,))
    if name == 'att':
        state = Input(shape=(seq_len, embed_len))
    else:
        state = Input(shape=(embed_len,))
    embed_input = embed(input)
    func = map_item('_'.join([name, mode]), funcs)
    if mode == 'decode':
        output = func(embed_input, state, vocab_num)
        return Model([input, state], output)
    else:
        output = func(embed_input)
        return Model(input, output)


def load_model(name, embed_mat, seq_len, mode):
    model = define_model(name, embed_mat, seq_len, mode)
    model.load_weights(map_item(name, paths), by_name=True)
    return model


def ind2word(word_inds):
    ind_words = dict()
    for word, ind in word_inds.items():
        ind_words[ind] = word
    return ind_words


seq_len = 100
max_len = 200

bos, eos = '*', '#'

path_stop_word = 'dict/stop_word.txt'
stop_word_re = load_word_re(path_stop_word)

path_embed = 'feat/embed.pkl'
path_word2ind = 'model/word2ind.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_word2ind, 'rb') as f:
    word2ind = pk.load(f)
word_inds = word2ind.word_index

ind_words = ind2word(word_inds)

funcs = {'s2s_encode': s2s_encode,
         's2s_decode': s2s_decode,
         'att_encode': att_encode,
         'att_decode': att_decode}

paths = {'s2s': 'model/s2s.h5',
         'att': 'model/att.h5'}

models = {'s2s_encode': load_model('s2s', embed_mat, seq_len, 'encode'),
          's2s_decode': load_model('s2s', embed_mat, seq_len, 'decode'),
          'att_encode': load_model('att', embed_mat, seq_len, 'encode'),
          'att_decode': load_model('att', embed_mat, seq_len, 'decode')}


def sample(probs, ind_words, cand):
    max_probs = np.array(sorted(probs, reverse=True)[:cand])
    max_probs = max_probs / np.sum(max_probs)
    max_inds = np.argsort(-probs)[:cand]
    next_ind = choice(max_inds, p=max_probs)
    return ind_words[next_ind]


def search():
    pass


def predict(text, name):
    sent1 = re.sub(stop_word_re, '', text.strip())
    seq1 = word2ind.texts_to_sequences([sent1])[0]
    pad_seq1 = pad_sequences([seq1], maxlen=seq_len, padding='pre', truncating='pre')
    encode = map_item(name + '_encode', models)
    state = encode.predict(pad_seq1)
    decode = map_item(name + '_decode', models)
    sent2 = bos
    next_word, count = '', 0
    while next_word != eos and count < max_len:
        sent2 = ' '.join([sent2, next_word])
        count = count + 1
        seq2 = word2ind.texts_to_sequences([sent2])[0]
        pad_seq2 = pad_sequences([seq2], maxlen=seq_len, padding='post', truncating='post')
        end = min(count - 1, seq_len - 1)
        probs = decode.predict([pad_seq2, state])[0][end]
        next_word = sample(probs, ind_words, cand=5)
    return sent2[2:]


if __name__ == '__main__':
    while True:
        text = input('text: ')
        clean_text = clean(text)
        print('s2s: %s' % predict(clean_text, 's2s'))
        print('att: %s' % predict(clean_text, 'att'))
