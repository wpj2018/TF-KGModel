#coding:utf-8
import logging
import tensorflow as tf
import numpy as np
eps = 1e-8

class Model(object):
    def __init__(self, args, pre_entity_emb=None, pre_relation_emb=None):
        self.logger = logging.getLogger("KGModel")
        self.logger.info("Init model")
        self.optim_type = args.optim_type
        self.learning_rate = args.learning_rate
        self.label_smoothing = args.label_smoothing
        self.entity_num = args.entity_num
        self.ndim = args.ndim
        if pre_entity_emb:
            self.E = tf.Variable(pre_entity_emb, dtype=tf.float32, name='entity_emb')
        else:
            self.E = tf.get_variable('entity_emb', [args.entity_num, args.ndim])

        if pre_relation_emb:
            self.R = tf.Variable(pre_relation_emb, dtype=tf.float32, name='relation_emb')
        else:
            self.R = tf.get_variable('relation_emb', [args.entity_num, args.ndim])

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)
        self._build_graph()
        self.saver = tf.train.Saver()
        # initialize the model
        self.sess.run(tf.global_variables_initializer())

    def _build_graph(self):
        """
        Builds the computation graph with Tensorflow
        """
        self._setup_placeholders()
        self._embed()
        self._encode()
        self._compute_loss()
        self._create_train_op()

        param_num = sum([np.prod(self.sess.run(tf.shape(v))) for v in self.all_params])
        self.logger.info('There are {} parameters in the model'.format(param_num))

    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.in_s = tf.placeholder(tf.int32, [None], name="subject")
        self.in_r = tf.placeholder(tf.int32, [None], name="relation")
        self.label = tf.placeholder(tf.float32, [None, None], name="label")

    def _embed(self):
        self.s_emb = tf.nn.embedding_lookup(self.E, self.in_s)
        self.r_emb = tf.nn.embedding_lookup(self.R, self.in_r)

    def _encode(self):
        self.predictions = tf.sigmoid(tf.matmul(self.s_emb + self.r_emb, self.E, transpose_b=True))

    def _BCELoss(self, logits, labels):
        if self.label_smoothing:
            labels = ((1.0 - self.label_smoothing) * labels) + 1.0/self.entity_num
        return -tf.reduce_mean(labels * tf.log(logits + eps) + (1-labels) * tf.log(1-logits + eps))

    def _compute_loss(self):
        self.loss = self._BCELoss(logits=self.predictions, labels=self.label)

        self.all_params = tf.trainable_variables()

    def _create_train_op(self):
        """
        Selects the training algorithm and creates a train operation with it
        """
        if self.optim_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        elif self.optim_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        elif self.optim_type == 'rprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        elif self.optim_type == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        else:
            raise NotImplementedError('Unsupported optimizer: {}'.format(self.optim_type))
        self.train_op = self.optimizer.minimize(self.loss)


class DistMult(Model):
    def __init__(self, args, pre_entity_emb=None, pre_relation_emb=None):
        Model.__init__(self, args, pre_entity_emb=pre_entity_emb, pre_relation_emb=pre_relation_emb)

    def _encode(self):
        self.predictions = tf.sigmoid(tf.matmul(self.s_emb * self.r_emb, self.E, transpose_b=True))


class ConvE(Model):
    def __init__(self, args, pre_entity_emb=None, pre_relation_emb=None):
        self.in_channels = args.in_channels
        self.out_channels = args.out_channels
        self.filt_h = args.filt_h
        self.filt_w = args.filt_w
        self.fc_length = (20-self.filt_h+1)*(20-self.filt_w+1)*self.out_channels
        Model.__init__(self, args, pre_entity_emb=pre_entity_emb, pre_relation_emb=pre_relation_emb)

    def _encode(self):
        s = tf.reshape(self.s_emb, [-1, 10, 20, 1])
        r = tf.reshape(self.r_emb, [-1, 10, 20, 1])
        x = tf.concat([s, r], 1)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.dropout(x, keep_prob=0.8)
        w = tf.get_variable('w', [self.filt_h, self.filt_w, self.in_channels, self.out_channels])
        x = tf.nn.conv2d(x, filter=w, strides=[1, 1, 1, 1], padding="VALID")
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        x = tf.nn.dropout(x, keep_prob=0.8)
        x = tf.reshape(x, [tf.shape(x)[0], self.fc_length])
        x = tf.layers.dense(x, units=self.ndim)
        x = tf.nn.dropout(x, keep_prob=0.7)
        x = tf.layers.batch_normalization(x)
        x = tf.nn.relu(x)
        b = tf.Variable(tf.zeros([self.entity_num]))
        self.predictions = tf.sigmoid(tf.matmul(x, self.E, transpose_b=True) + b)


class MyModel(Model):
    def __init__(self, args, pre_entity_emb=None, pre_relation_emb=None):
        Model.__init__(self, args, pre_entity_emb=pre_entity_emb, pre_relation_emb=pre_relation_emb)

    def _encode(self):
        tmp = tf.concat([self.s_emb, self.r_emb], -1)

        gate = tf.layers.dense(tmp, units=1, activation=tf.sigmoid)

        input = gate * self.s_emb * self.r_emb
        self.predictions = tf.sigmoid(tf.matmul(input, self.E, transpose_b=True))