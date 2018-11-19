import numpy as np
import tensorflow as tf
import gym
import time

################## hyper parameters ##################

LR_A = 0.001
LR_C = 0.002
GAMMA = 0.99
TAU = 0.001
MEMORY_CAPACITY = 1000000
BATCH_SIZE = 32
HIDDEN_SIZE = 64
REPLAY_START = 1000


################## DDPG algorithm ##################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float64)
        self.pointer = 0
        self.sess = tf.Session()
        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound
        self.replay_start = REPLAY_START
        self.S = tf.placeholder(tf.float64, [None, s_dim], name='S')
        self.S_ = tf.placeholder(tf.float64, [None, s_dim], name='S_')
        self.R = tf.placeholder(tf.float64, [None, ], name='reward')
        self.a = self._build_a(self.S)
        q = self._build_c(self.S, self.a)
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)  # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q, 这里我改为reduce_sum
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):  # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        s = np.reshape(s, [-1, self.s_dim])
        return self.sess.run(self.a, feed_dict={self.S: s})

    def learn(self):
        if self.pointer < self.replay_start:
            return
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]  # transitions data
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim:self.s_dim + self.a_dim]
        br = bt[:, self.s_dim + self.a_dim]
        bs_ = bt[:, -self.s_dim:]
        self.sess.run(self.atrain, feed_dict={self.S: bs})
        self.sess.run(self.ctrain, feed_dict={self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):  # 每次存储一个即可
        s = np.reshape(np.array(s), [self.s_dim, 1])
        a = np.reshape(np.array(a), [self.a_dim, 1])
        r = np.reshape(np.array(r), [1, 1])
        s_ = np.reshape(np.array(s_), [self.s_dim, 1])

        transition = np.vstack((s, a, r, s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = np.reshape(transition, [2 * self.s_dim + self.a_dim + 1, ])
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            h1 = tf.layers.dense(s, units=HIDDEN_SIZE, activation=tf.nn.relu, trainable=trainable)
            h2 = tf.layers.dense(h1, units=self.a_dim, activation=tf.nn.tanh, trainable=trainable)
            return h2 * self.a_bound

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            input_s = tf.reshape(s, [-1, self.s_dim])
            input_a = tf.reshape(a, [-1, self.a_dim])
            input_all = tf.concat([input_s, input_a], axis=1)  # s: [batch_size, s_dim]
            h1 = tf.layers.dense(input_all, units=HIDDEN_SIZE, activation=tf.nn.relu, trainable=trainable)
            h2 = tf.layers.dense(h1, units=1, activation=None, trainable=trainable)
            return h2
