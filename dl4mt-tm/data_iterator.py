import numpy

import cPickle as pkl
import gzip


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class TextIterator:
    """a general text iterator."""

    def __init__(self,
                 dataset,
                 dicts,
                 voc_sizes=None,
                 batch_size=128,
                 maxlen=100):

        self.datasets  = [fopen(data, 'r') for data in dataset]
        self.dicts     = [pkl.load(open(dic, 'rb')) for dic in dicts]
        self.voc_sizes = voc_sizes
        self.buffers   = [[] for _ in self.datasets]

        self.batch_size = batch_size
        self.maxlen = maxlen
        self.nums = len(self.datasets)
        self.k = batch_size * 20  # cache=20
        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        for i in range(self.nums):
            self.datasets[i].seek(0)

    def fill(self):
        for i in range(self.nums):
            self.buffers[i] = []  # clean the buffers
        
        for _ in range(self.k):
            lines = [self.datasets[i].readline() for i in range(self.nums)]

            flag  = False
            for line in lines:
                if line == "":
                    flag = True
            if flag:
                break

            for ia in range(self.nums):
                self.buffers[ia].append(lines[ia].strip().split())

        # sort by target buffer --- dafult setting:  source, target, tm-source, tm-target
        tidx = numpy.array([len(t) for t in self.buffers[1]]).argsort()
        for ib in range(self.nums):
            self.buffers[ib] = [self.buffers[ib][j] for j in tidx]
            
    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        datasets = [[] for _ in self.datasets]

        # fill buffer, if it's empty
        assert len(self.buffers[0]) == len(self.buffers[1]), 'Buffer size mismatch!'

        if len(self.buffers[0]) == 0:
            self.fill()
            
        flag2 = False
        for ic in range(self.nums):
            if len(self.buffers[ic]) == 0:
                flag2 = True

        if flag2:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:
            
            # actual work here
            _samples = 0
            while True:

                # read from dataset file and map to word index
                # print _samples
                _lines = []
                flagx = False
                for id in range(self.nums):
                    try:
                        line = self.buffers[id].pop()
                    except IndexError:
                        self.fill()
                        flagx = True
                        break

                    line = [self.dicts[id][w] if w in self.dicts[id] else 1 for w in line]
                    if self.voc_sizes[id] > 0:  # so I need to input [0 0 0 0]
                        line = [w if w < self.voc_sizes[id] else 1 for w in line]
                    _lines.append(line)

                if flagx:
                    continue
                    
                    
                flag3 = False
                for line in _lines:
                    if len(line) > self.maxlen:
                        flag3 = True

                if flag3:
                    continue

                for ie in range(self.nums):
                    datasets[ie].append(_lines[ie])
                _samples += 1

                if _samples >= self.batch_size:
                    break

        except IOError:
            self.end_of_data = True

        flag4 = False
        for ig in range(self.nums):
            if len(datasets[ig]) <= 0:
                flag4 = True

        if flag4:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return datasets


# prepare batch
def prepare_data(seqs_x, maxlen=None, n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]

    if maxlen is not None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x <= maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)

        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    for idx, s_x in enumerate(seqs_x):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.
    return x, x_mask


def prepare_cross(seqs_x1, seqs_x2, maxlen_x1):
    n_samples = len(seqs_x1)
    t = numpy.zeros((maxlen_x1, n_samples)).astype('int64')
    t_mask = numpy.zeros((maxlen_x1, n_samples)).astype('float32')

    for idx, (x1, x2) in enumerate(zip(seqs_x1, seqs_x2)):

        match = [[(i, abs(i - j))
                  for i, xx2 in enumerate(x2) if xx1 == xx2]
                 for j, xx1 in enumerate(x1)]

        for jdx, m in enumerate(match):
            if len(m) > 0:
                if len(m) == 1:
                    t[jdx, idx] = m[0][0]
                else:
                    t[jdx, idx] = sorted(m, key=lambda a: a[1])[0][0]

                t_mask[jdx, idx] = 1.

    return t, t_mask
