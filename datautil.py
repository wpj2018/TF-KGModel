#coding:utf-8
import numpy as np
from collections import defaultdict
import tensorflow as tf

class DataUtil:

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.train_data = self.load_data(args.data_dir, "train", reverse=args.reverse)
        self.valid_data = self.load_data(args.data_dir, "valid", reverse=args.reverse)
        self.test_data = self.load_data(args.data_dir, "test", reverse=args.reverse)
        self.data = self.train_data + self.valid_data + self.test_data
        self.entities = self.get_entities(self.data)
        self.relations = self.get_relations(self.data)

        self.entity_idxs = {self.entities[i]: i for i in range(len(self.entities))}
        self.relation_idxs = {self.relations[i]: i for i in range(len(self.relations))}

        self.train_data_ids = self.get_data_idxs(self.train_data)
        self.test_data_ids = self.get_data_idxs(self.test_data)
        self.valid_data_ids = self.get_data_idxs(self.valid_data)
        self.data_ids = self.train_data_ids + self.test_data_ids + self.valid_data_ids

        self.total_sr_vocab = self.get_sr_vocab(self.data_ids)

    def get_data_idxs(self, data):
        data_idxs = [(self.entity_idxs[data[i][0]], self.relation_idxs[data[i][1]],
                      self.entity_idxs[data[i][2]]) for i in range(len(data))]
        return data_idxs

    def load_data(self, data_dir, data_type="train", reverse=False):
        with open("%s%s.txt" % (data_dir, data_type), "r") as f:
            data = f.read().strip().split("\n")
            data = [i.split() for i in data]
            if reverse:
                data += [[i[2], i[1]+"_reverse", i[0]] for i in data]
        return data

    def get_relations(self, data):
        relations = sorted(list(set([d[1] for d in data])))
        return relations

    def get_entities(self, data):
        entities = sorted(list(set([d[0] for d in data]+[d[2] for d in data])))
        return entities

    def get_sr_vocab(self, data):
        er_vocab = defaultdict(list)
        for triple in data:
            er_vocab[(triple[0], triple[1])].append(triple[2])
        return er_vocab

    def get_batch(self, data_set, shuffle):
        data = []
        if data_set == "train": data = self.train_data_ids
        if data_set == "valid": data = self.valid_data_ids
        if data_set == "test": data = self.test_data_ids

        sr_vocab = self.get_sr_vocab(data)
        if data_set == "train":
            data = list(sr_vocab.keys())

        data_size = len(data)

        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        batch_size = self.batch_size

        for batch_start in np.arange(0, data_size, batch_size):
            batch_indices = indices[batch_start: batch_start + batch_size]
            batch = [data[i] for i in batch_indices]

            targets = np.zeros((len(batch), len(self.entities)))
            for idx, pair in enumerate(batch):
                targets[idx, sr_vocab[pair]] = 1.

            yield np.array(batch), targets

    def train(self, model):

        freq_report = 1

        for epoch in range(1, self.epochs+1):
            eval_batches = self.get_batch('train', shuffle=False)

            losses = []
            for b_itx, (data_batch, label) in enumerate(eval_batches):
                s_idx = data_batch[:, 0]
                r_idx = data_batch[:, 1]
                feed_dict = {model.in_s: s_idx,
                             model.in_r: r_idx,
                             model.label: label}
                _, loss, predictions = model.sess.run([model.train_op, model.loss, model.predictions], feed_dict)
                losses.append(loss)

            print('after {} epoch avg loss is {}'.format(epoch, np.mean(losses)))

            if epoch % freq_report == 0:
                self.evaluate(model, self.valid_data_ids)

    def evaluate(self, model, data):
        hits = []
        ranks = []
        for i in range(10):
            hits.append([])

        print("Number of data points: %d" % len(data))
        batchs = self.get_batch('valid', shuffle=False)
        for b_itx, (data_batch, target) in enumerate(batchs):
            s_idx = data_batch[:, 0]
            r_idx = data_batch[:, 1]
            o_idx = data_batch[:, 2]
            feed_dict = {model.in_s: s_idx,
                         model.in_r: r_idx}
            predictions = model.sess.run(model.predictions, feed_dict)

            for j in range(data_batch.shape[0]):
                filt = self.total_sr_vocab[(data_batch[j][0], data_batch[j][1])]
                target_value = predictions[j, o_idx[j]]
                predictions[j, filt] = 0.0
                predictions[j, o_idx[j]] = target_value

            sort_idxs = np.argsort(-predictions, -1)

            for j in range(data_batch.shape[0]):
                rank = np.where(sort_idxs[j] == o_idx[j])[0][0]
                ranks.append(rank + 1)

                for hits_level in range(10):
                    if rank <= hits_level:
                        hits[hits_level].append(1.0)
                    else:
                        hits[hits_level].append(0.0)

        print('Hits @10: {0}'.format(np.mean(hits[9])))
        print('Hits @3: {0}'.format(np.mean(hits[2])))
        print('Hits @1: {0}'.format(np.mean(hits[0])))
        print('Mean rank: {0}'.format(np.mean(ranks)))
        print('Mean reciprocal rank: {0}'.format(np.mean(1. / np.array(ranks))))
