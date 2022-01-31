import os
import sys
import operator
import random
import time
import numpy as np

wordspos = {}
wordsneg = {}
wordidx = {}

pairspos = {}
pairsneg = {}
pairidx = {}

training_data = []
test_data = []


def get_review_words(path):
    with open(path, 'r') as file:
        text = file.read().replace('\n', ' ')
    words = text.lower().split(' ')
    for c in [' ', '', ',']:
        words = list(filter(lambda x: x != c, words))
    return words


def get_review_pairs(path):
    with open(path, 'r') as file:
        text = file.read().replace('\n', ' ')
    words = text.lower().split(' ')
    for c in [' ', '', ',']:
        words = list(filter(lambda x: x != c, words))
    pairs = []
    for i in range(len(words) - 1):
        pairs.append(words[i] + ' ' + words[i + 1])
    return pairs


def count_words(paths):
    dicts = [wordspos, wordsneg]
    for i in range(2):
        for test_path in os.listdir(paths[i]):
            words = get_review_words(os.path.join(paths[i], test_path))
            for word in words:
                if word not in dicts[i]:
                    dicts[i][word] = 1
                else:
                    dicts[i][word] += 1


def count_pairs(paths):
    dicts = [pairspos, pairsneg]
    for i in range(2):
        for test_path in os.listdir(paths[i]):
            pairs = get_review_pairs(os.path.join(paths[i], test_path))
            for pair in pairs:
                if pair not in dicts[i]:
                    dicts[i][pair] = 1
                else:
                    dicts[i][pair] += 1


def vectorize_words(words_per_set, ratio):
    pos = []
    neg = []
    wpclone = wordspos.copy()
    while len(pos) < words_per_set:
        w = max(wordspos.items(), key=operator.itemgetter(1))[0]
        if w not in wordsneg or wordspos[w] / wordsneg[w] >= ratio:
            pos.append(w)
        del wordspos[w]
    while len(neg) < words_per_set:
        w = max(wordsneg.items(), key=operator.itemgetter(1))[0]
        if w not in wpclone or wordsneg[w] / wpclone[w] >= ratio:
            neg.append(w)
        del wordsneg[w]
    for i in range(words_per_set):
        wordidx[pos[i]] = i
    for i in range(words_per_set):
        wordidx[neg[i]] = i + words_per_set


def vectorize_pairs(pairs_per_set, ratio):
    pos = []
    neg = []
    wpclone = pairspos.copy()
    while len(pos) < pairs_per_set:
        w = max(pairspos.items(), key=operator.itemgetter(1))[0]
        if w not in pairsneg or pairspos[w] / pairsneg[w] >= ratio:
            pos.append(w)
        del pairspos[w]
    while len(neg) < pairs_per_set:
        w = max(pairsneg.items(), key=operator.itemgetter(1))[0]
        if w not in wpclone or pairsneg[w] / wpclone[w] >= ratio:
            neg.append(w)
        del pairsneg[w]
    for i in range(pairs_per_set):
        pairidx[pos[i]] = i
    for i in range(pairs_per_set):
        pairidx[neg[i]] = i + pairs_per_set


# Extractor
def extract_features(words):
    features = np.zeros(len(wordidx) + len(pairidx))
    for word in words:
        if word in wordidx:
            features[wordidx[word]] += 1
    for i in range(len(words) - 1):
        pair = words[i] + ' ' + words[i + 1]
        if pair in pairidx:
            features[len(wordidx) + pairidx[pair]] += 1
    return features


def collect_data(pos, neg, test):
    for test_path in os.listdir(pos):
        words = get_review_words(os.path.join(pos, test_path))
        training_data.append((extract_features(words), 1))
    for test_path in os.listdir(neg):
        words = get_review_words(os.path.join(neg, test_path))
        training_data.append((extract_features(words), -1))
    for test_path in os.listdir(test):
        words = get_review_words(os.path.join(test, test_path))
        test_data.append((test_path, extract_features(words)))
        #test_data.append((extract_features(words), 1 if test_path[6] == 'p' else -1))


def run_test(network, data):
    success = 0
    tries = 0
    for d in data:
        tries += 1
        if network.predict(d[0]) == d[1]:
            success += 1
    return success / tries


class Perceptron:
    def __init__(self, n, m, lr):
        pos = np.random.uniform(0, 1/(2*(n + m) + 1), size=n)
        ppos = np.random.uniform(0, 1/(2*(n + m) + 1), size=m)
        neg = np.random.uniform(-1/(2*(n + m) + 1), 0, size=n)
        pneg = np.random.uniform(-1/(2*(n + m) + 1), 0, size=m)
        self.n = n
        self.m = m
        self.w = np.append(pos, neg)
        self.w = np.append(self.w, ppos)
        self.w = np.append(self.w, pneg)
        self.b = np.random.uniform(-1/(2*(n + m) + 1), 1/(2*(n + m) + 1))
        self.lr = lr

    # Predictor
    def predict(self, features):
        result = np.dot(self.w, features) + self.b
        return 1 if result > 0 else -1

    # SGD
    def fit(self, features, real):
        pred = self.predict(features)
        grad_w = 1/((self.n + self.m) * 2 + 1) * (pred - real) * features
        grad_b = 1/((self.n + self.m) * 2 + 1) * (pred - real)
        self.w -= self.lr * grad_w
        self.b -= self.lr * grad_b


if __name__ == '__main__':
    start = time.time()

    positive_path = sys.argv[1]
    negative_path = sys.argv[2]
    test_path = sys.argv[3]

    words_per_set = 100
    pairs_per_set = 60
    word_accept_ratio = 1.5
    pair_accept_ratio = 1.5
    learning_rate = 0.000006
    epochs = 5000

    count_words([positive_path, negative_path])
    count_pairs([positive_path, negative_path])
    vectorize_words(words_per_set, word_accept_ratio)
    vectorize_pairs(pairs_per_set, pair_accept_ratio)
    collect_data(positive_path, negative_path, test_path)

    perceptron = Perceptron(words_per_set, pairs_per_set, learning_rate)
    random.shuffle(training_data)

    for e in range(epochs):
        # SGD
        for d in training_data:
            perceptron.fit(d[0], d[1])
        print(run_test(perceptron, training_data))

    for test in test_data:
        x = perceptron.predict(test[1])
        print(test[0], x)

    stop = time.time()
    #print(stop - start)
