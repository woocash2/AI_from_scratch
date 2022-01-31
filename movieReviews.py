import sys
import os
import random
import operator
import time
import math


class WordBank:
    positive_str = "first-rate insightful clever charming comical charismatic enjoyable uproarious original tender" \
                   " hilarious absorbing sensitive riveting intriguing powerful fascinating pleasant surprising dazzling" \
                   " thought provoking imaginative legendary unpretentious"

    negative_str = "second-rate violent moronic third-rate flawed juvenile boring distasteful ordinary disgusting" \
                    " senseless static brutal confused disappointing bloody silly tired predictable stupid uninteresting" \
                    " weak tiresome trite uneven cliché-ridden outdated dreadful bland dull"

    positive_words = positive_str.split(' ')
    negative_words = negative_str.split(' ')


class Algebra:
    def dot(self, x, y):
        res = 0
        for i in range(min(len(x), len(y))):
            res += x[i] * y[i]
        return res


class TextProvider:
    def get_review(self, path):
        with open(path, 'r') as file:
            text = file.read().replace('\n', ' ')
        return text

    def get_words_normalized(self, txt):
        words = txt.lower().split(' ')
        for c in [' ', '', ',']:
            words = list(filter(lambda x: x != c, words))
        return words


class Extractor:

    def __init__(self, wordlist, pairlist):
        self.wlist = wordlist
        self.plist = pairlist
        self.widx = {}
        self.pidx = {}
        for i in range(len(wordlist)):
            self.widx[wordlist[i]] = i
        for i in range(len(pairlist)):
            self.pidx[pairlist[i]] = i

    def good_words_cnt(self, words):
        cnt = 0
        for word in words:
            if word in WordBank.positive_words:
                cnt += 1
        return cnt

    def bad_words_cnt(self, words):
        cnt = 0
        for word in words:
            if word in WordBank.negative_words:
                cnt += 1
        return cnt

    def extract_words(self, words, lst, idx):
        v = [0] * (1 + len(lst))
        found = 0
        for word in words:
            if word in lst:
                found += 1
                v[idx[word]] += 1
        for i in range(len(v)):
            v[i] = v[i] / (found + 1)
        v[-1] = 10 / (found + 1)
        return v

    def extract_singles(self, words):
        return self.extract_words(words, self.wlist, self.widx)

    def extract_pairs(self, words):
        wordpairs = []
        for i in range(len(words) - 1):
            wordpairs.append(words[i] + ' ' + words[i + 1])
        return self.extract_words(wordpairs, self.plist, self.pidx)

    def extract_all(self, words):
        a = []
        a.extend(self.extract_singles(words))
        x = a[-1]
        a.pop()
        b = self.extract_pairs(words)
        a.extend(b)
        a[-1] += x
        return a


class Network:
    def __init__(self, n, dev, bias):
        self.w = []
        self.bias = bias
        self.algebra = Algebra()
        for i in range(n + 1):
            self.w.append(random.uniform(-dev, dev))

    def evaluate(self, v):
        x = list.copy(v)
        x.append(self.bias)
        y = self.algebra.dot(self.w, x)
        return y

    def sigma(self, x):
        x = max(x, 100)
        x = min(x, -100)
        return 2 / (1 + math.exp(x)) - 1

    def decide(self, v):
        return 1 if self.evaluate(v) > 0 else -1


class Teacher:
    def __init__(self, network, train_pos_path, train_neg_path, extr):
        self.network = network
        self.algebra = Algebra()
        self.pos = train_pos_path
        self.neg = train_neg_path
        self.extr = extr

    def grad(self, x, res, ans):
        y = list.copy(x)
        for i in range(len(y)):
            y[i] = y[i] * 2 * (res - ans)
        return y

    def train(self, lr):
        sets = []
        for file_name in os.listdir(self.pos):
            sets.append((os.path.join(self.pos, file_name), 1))
        for file_name in os.listdir(self.neg):
            sets.append((os.path.join(self.neg, file_name), -1))
        random.shuffle(sets)
        all = 0
        sr = 0
        for s in sets:
            all += 1
            text_provider = TextProvider()
            review = text_provider.get_review(s[0])
            words = text_provider.get_words_normalized(review)
            x = self.extr.extract_all(words)
            r = self.network.decide(x)
            #print(r, s[1])
            gr = self.grad(x, r, s[1])
            #print(x[:10])
            #print(gr[:10])
            #print(self.network.w[:10])
            if r == s[1]:
                sr += 1
            for i in range(len(gr)):
                gr[i] *= lr
            for i in range(len(self.network.w)):
                self.network.w[i] -= gr[i]
        print(sr, all, sr / all)

    def teach(self, lr, epochs):
        for i in range(epochs):
            self.train(lr)


class Explorator:
    def __init__(self, ppath, npath):
        self.pos = {}
        self.neg = {}
        self.pospairs = {}
        self.negpairs = {}
        self.ppath = ppath
        self.npath = npath

    def find_words(self):
        for f in os.listdir(self.ppath):
            t = TextProvider()
            r = t.get_review(os.path.join(self.ppath, f))
            wrds = t.get_words_normalized(r)
            for w in wrds:
                if w in self.pos:
                    self.pos[w] += 1
                else:
                    self.pos[w] = 1

        for f in os.listdir(self.npath):
            t = TextProvider()
            r = t.get_review(os.path.join(self.npath, f))
            wrds = t.get_words_normalized(r)
            for w in wrds:
                if w in self.neg:
                    self.neg[w] += 1
                else:
                    self.neg[w] = 1

    def find_pairs(self):
        for f in os.listdir(self.ppath):
            t = TextProvider()
            r = t.get_review(os.path.join(self.ppath, f))
            wrds = t.get_words_normalized(r)
            for i in range(len(wrds) - 1):
                w = wrds[i] + ' ' + wrds[i + 1]
                if w in self.pospairs:
                    self.pospairs[w] += 1
                else:
                    self.pospairs[w] = 1

        for f in os.listdir(self.npath):
            t = TextProvider()
            r = t.get_review(os.path.join(self.npath, f))
            wrds = t.get_words_normalized(r)
            for i in range(len(wrds) - 1):
                w = wrds[i] + ' ' + wrds[i + 1]
                if w in self.negpairs:
                    self.negpairs[w] += 1
                else:
                    self.negpairs[w] = 1

    def gather_words(self, n, ratio, ps, ng):
        posi = []
        nega = []
        npos = ps.copy()
        while len(posi) < n:
            a = max(npos.items(), key=operator.itemgetter(1))[0]
            x = npos[a]
            if a not in ng:
                continue
            y = ng[a]
            if x / y > ratio:
                posi.append(a)
            del npos[a]
        npos = ps.copy()
        while len(nega) < n:
            a = max(ng.items(), key=operator.itemgetter(1))[0]
            y = ng[a]
            if a not in ps:
                continue
            x = ps[a]
            if y / x > ratio:
                nega.append(a)
            del ng[a]
        wlist = posi
        wlist.extend(nega)
        return wlist

    def gather_single(self, n, ratio):
        return self.gather_words(n, ratio, self.pos, self.neg)

    def gather_pairs(self, n, ratio):
        return self.gather_words(n, ratio, self.pospairs, self.negpairs)


def num_of_words(paths):
    all_words = set()
    for path in paths:
        for testpath in os.listdir(path):
            words = get_review_words(os.path.join(path, testpath))
            for word in words:
                if word[:6] not in all_words:
                    all_words.add(word[:6])
    return len(all_words)

if __name__ == '__main__':

    start = time.time()

    positive_path = sys.argv[1]
    negative_path = sys.argv[2]
    test_path = sys.argv[3]

    words_per_set = 100
    pairs_per_set = 0
    word_accept_ratio = 1.3
    pair_accept_ratio = 1.3
    epochs = 100
    learning_rate = 0.04

    exp = Explorator(positive_path, negative_path)
    exp.find_words()
    exp.find_pairs()
    wordlist = exp.gather_single(words_per_set, word_accept_ratio)
    pairlist = exp.gather_pairs(pairs_per_set, pair_accept_ratio)

    extractor = Extractor(wordlist, pairlist)
    network = Network(2 * words_per_set + 2 * pairs_per_set, 0.1, 1)

    teacher = Teacher(network, positive_path, negative_path, extractor)
    teacher.teach(learning_rate, epochs)
    num = 0
    snum = 0

    for test in os.listdir(test_path):
        num += 1
        ans = 1 if test[6] == 'p' else -1
        t = TextProvider()
        rev = t.get_review(os.path.join(test_path, test))
        words = t.get_words_normalized(rev)
        a = network.decide(extractor.extract_all(words))
        if a == ans:
            snum += 1
    print("Params:", "wps:", words_per_set, "pps:", pairs_per_set, "lr:", learning_rate, "eps:", epochs)
    print("Score:", snum / num)
    end = time.time()

    print(end - start, "seconds")