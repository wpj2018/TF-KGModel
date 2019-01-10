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
        if pre_entity_emb:
            self.E = tf.Variable(pre_entity_emb, dtype=tf.float32, name='entity_emb')
        else:
            self.E = tf.get_variable('entity_emb', [args.entity_num, args.ndim])
        #tf.Variable(tf.truncated_normal(shape=[args.entity_num, args.ndim]))

        if pre_relation_emb:
            self.R = tf.Variable(pre_relation_emb, dtype=tf.float32, name='relation_emb')
        else:
            self.R = tf.get_variable('relation_emb', [args.entity_num, args.ndim])
        #self.R = tf.Variable(tf.truncated_normal(shape=[args.relation_num, args.ndim]))

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
        self.predictions = tf.sigmoid(tf.matmul(self.s_emb * self.r_emb, self.E, transpose_b=True))

    def _BCELoss(self, logits, labels):
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