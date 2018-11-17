pos_dict = {'J': 'a',
            'V': 'v',
            'R': 'r'}


def map_pos(pos):
    if pos[0] in pos_dict:
        return pos_dict[pos[0]]
    else:
        return 'n'


def load_word(path):
    words = list()
    with open(path, 'r') as f:
        for line in f:
            words.append(line.strip())
    return words


def load_word_re(path):
    words = load_word(path)
    return '(' + ')|('.join(words) + ')'


def map_item(name, items):
    if name in items:
        return items[name]
    else:
        raise KeyError
