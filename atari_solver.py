import random

from tensorflow.python.ops.init_ops import VarianceScaling
import tensorflow as tf
import gym_wrappers
import numpy as np
import os
from collections import deque
import gym
import cv2
from gym.wrappers import TimeLimit

cv2.ocl.setUseOpenCL(False)
os.environ.setdefault('PATH', '')

# Hyper-Parameters
ENV_NAME = "BreakoutNoFrameskip-v4"

GAMMA = 0.99
N = 2000000  # Number of Iterations
C = 10000  # Network weights update frequency: Every 'C' steps
LEARNING_RATE = 0.0001
EVALUATION_INTERVAL = 100000
MEMORY_SIZE = 10000
BATCH_SIZE = 32
TRAINING_INTERVAL = 4
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01


class QNetwork:
    def __init__(self, num_actions, memory_size, sess):
        self.number_of_actions = num_actions
        self.memory = deque(maxlen=memory_size)
        self.cur_state = tf.placeholder(tf.float32, shape=(None, 84, 84, 4), name='input_x')
        self.q_target_ph = tf.placeholder(dtype=tf.float32, shape=(32,))
        self.action_taken_ph = tf.placeholder(dtype=tf.int32, shape=[32, 2])
        self.online_logits = model(self.cur_state, weights_online, bias_online, "online")
        self.target_logits = model(self.cur_state, weights_target, bias_target, "target")
        self.assign_wt_op = assign_weights()
        self.q_pred = tf.reshape(tf.gather_nd(self.online_logits, self.action_taken_ph), shape=(32,))
        self.loss = tf.reduce_mean(
            tf.compat.v1.losses.mean_squared_error(predictions=self.q_pred, labels=self.q_target_ph))
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE, decay=GAMMA)
        self.train_op = self.optimizer.minimize(self.loss)

        self.sess = sess

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def predict(self, state, logits):
        q_value = self.sess.run(logits, feed_dict={self.cur_state: state})
        return q_value

    def act(self, state, exploration_rate):
        if np.random.rand() < exploration_rate:
            return random.randrange(self.number_of_actions)
        q_value = self.predict(state, self.online_logits)
        return np.argmax(q_value[0])

    def evaluate(self, env, exploration_rate):
        plays_rewards = []
        for play_num in range(30):
            play_reward = 0
            for eps_num in range(5):
                eps_reward = 0
                eps_done = False
                current_state = np.reshape(env.reset(), (1, 84, 84, 4))
                while not eps_done:
                    action = self.act(current_state, exploration_rate)
                    next_state, reward, eps_done, info = env.step(action)
                    current_state = np.reshape(next_state, (1, 84, 84, 4))
                    eps_reward += reward
                play_reward += eps_reward
            plays_rewards.append(play_reward)
        return np.mean(plays_rewards)

    def train(self):
        random_index = np.random.randint(0, len(self.memory), 32)
        sarso_list = [self.memory[x] for x in random_index]  # State Action Reward NextState Omega
        s = []
        a = []
        r = []
        s_1 = []
        omega = []
        for i, sarso in enumerate(sarso_list):
            s.append(sarso[0])
            a.append(sarso[1])
            r.append(sarso[2])
            s_1.append(sarso[3])
            omega.append(sarso[4])

        actions = []
        for idx, val in enumerate(a):
            actions.append([idx, val])
        not_omega = np.ones((32,)) - np.reshape(omega, (32,))
        return_next_step = np.multiply(not_omega, (
                    GAMMA * np.reshape(np.max(self.predict(s_1, self.target_logits), axis=1), (32,))))
        target_q_value = r + return_next_step
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.cur_state: np.reshape(s, (32, 84, 84, 4)),
                                                                       self.q_target_ph: target_q_value,
                                                                       self.action_taken_ph: actions})
        return loss

    def render(self, env):
        video_env = gym.wrappers.Monitor(env, "/", force=True)
        done = False
        current_state = video_env.reset()
        while not done:
            action = self.act(current_state, 0.0001)
            next_state, reward, done, info = video_env.step(action)


def model(x, weights, bias, model_name):
    conv_layer1 = tf.nn.conv2d(x, weights["cnn1"], strides=[1, 4, 4, 1], padding='SAME', name="conv1_" + model_name)
    cnn1 = tf.nn.relu(conv_layer1)

    conv_layer2 = tf.nn.conv2d(cnn1, weights["cnn2"], strides=[1, 2, 2, 1], padding='SAME', name="conv2_" + model_name)
    cnn2 = tf.nn.relu(conv_layer2)

    conv_layer3 = tf.nn.conv2d(cnn2, weights["cnn3"], strides=[1, 1, 1, 1], padding='SAME', name="conv3_" + model_name)
    cnn3 = tf.nn.relu(conv_layer3)

    flat = tf.contrib.layers.flatten(cnn3)

    fc1 = tf.matmul(flat, weights["fc1"]) + bias["fc1"]
    fc1 = tf.nn.relu(fc1)

    Z = tf.matmul(fc1, weights["fc2"]) + bias["fc2"]
    return Z  # n x 4


weights_online = {
    "cnn1": tf.get_variable("o_cnn1", shape=[8, 8, 4, 32], initializer=VarianceScaling(), trainable=True,
                            dtype=tf.float32),
    "cnn2": tf.get_variable("o_cnn2", shape=[4, 4, 32, 64], initializer=VarianceScaling(), trainable=True,
                            dtype=tf.float32),
    "cnn3": tf.get_variable("o_cnn3", shape=[3, 3, 64, 64], initializer=VarianceScaling(), trainable=True,
                            dtype=tf.float32),
    "fc1": tf.get_variable("o_fc1", shape=[11 * 11 * 64, 512], initializer=VarianceScaling(), trainable=True,
                           dtype=tf.float32),
    "fc2": tf.get_variable("o_fc2", shape=[512, 4], initializer=VarianceScaling(), trainable=True, dtype=tf.float32)
}

bias_online = {
    "fc1": tf.get_variable("o_fc1_b", shape=[512], initializer=VarianceScaling(), trainable=True, dtype=tf.float32),
    "fc2": tf.get_variable("o_fc2_b", shape=[4], initializer=VarianceScaling(), trainable=True, dtype=tf.float32)
}

weights_target = {
    "cnn1": tf.get_variable("t_cnn1", shape=[8, 8, 4, 32], initializer=VarianceScaling(), trainable=False,
                            dtype=tf.float32),
    "cnn2": tf.get_variable("t_cnn2", shape=[4, 4, 32, 64], initializer=VarianceScaling(), trainable=False,
                            dtype=tf.float32),
    "cnn3": tf.get_variable("t_cnn3", shape=[3, 3, 64, 64], initializer=VarianceScaling(), trainable=False,
                            dtype=tf.float32),
    "fc1": tf.get_variable("t_fc1", shape=[11 * 11 * 64, 512], initializer=VarianceScaling(), trainable=False,
                           dtype=tf.float32),
    "fc2": tf.get_variable("t_fc2", shape=[512, 4], initializer=VarianceScaling(), trainable=False, dtype=tf.float32)
}

bias_target = {
    "fc1": tf.get_variable("t_fc1_b", shape=[512], initializer=VarianceScaling(), trainable=False, dtype=tf.float32),
    "fc2": tf.get_variable("t_fc2_b", shape=[4], initializer=VarianceScaling(), trainable=False, dtype=tf.float32)
}


def copy_weights(sess, assign_wt_op):
    sess.run(assign_wt_op)


def assign_weights():
    return tf.group(
        [tf.assign(target, online) for (target, online) in zip(weights_target.values(), weights_online.values())] \
        + [tf.assign(target, online) for (target, online) in zip(bias_target.values(), bias_online.values())])


def plot_graphs(episodic_reward, evaluation_rewards, losses):
    import matplotlib.pyplot as plt
    episodic_reward = np.mean(np.resize(episodic_reward, len(episodic_reward) - (len(episodic_reward)%30)).reshape(-1,30), axis=1)[0:-1]

    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.plot(episodic_reward)
    plt.show()

    plt.xlabel("Evaluation Number")
    plt.ylabel("Score")
    plt.plot(evaluation_rewards)
    plt.show()

    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.plot(losses)
    plt.show()


# TODO : plot rewards, gradients and loss
def atari(COPY_WEIGHTS_INTERVAL=C):
    assign_weight_group = assign_weights()
    env = gym_wrappers.wrap_atari_deepmind(ENV_NAME, True)
    env_eval = gym_wrappers.wrap_atari_deepmind(ENV_NAME, False)
    action_space = env.action_space.n
    sess = tf.Session()
    q_network = QNetwork(action_space, MEMORY_SIZE, sess)
    sess.run(tf.global_variables_initializer())
    episode = 0
    losses = []
    step = 0
    episodic_reward = []
    evaluation_rewards = []
    # TODO : correct the loop indexing here
    while step < N:
        if episode % 100 == 0 and len(losses) > 0:
            print("Iterations: " + str(step) + "    " + "Episodes: " + str(episode))
            print("Loss: " + str(np.mean(losses)))

        current_state = np.reshape(env.reset(), (1, 84, 84, 4))
        # TODO : correct the calculation of epsilon
        exploration_rate = max(EXPLORATION_MAX - step / 100000, EXPLORATION_MIN)
        eps_reward = 0
        while step < N:
            if step % EVALUATION_INTERVAL == 0 and step != 0:
                evaluated_score = q_network.evaluate(env_eval, 0.001)
                evaluation_rewards.append(evaluated_score)
                print("*" * 16)
                print("Evaluated Score for iteration [{0}.]    --->   {1}.".format(str(step), str(evaluated_score)))
                print("*" * 16)
            action = q_network.act(current_state, exploration_rate)
            next_state, reward, terminal, info = env.step(action)
            step += 1
            q_network.remember(current_state, action, reward, next_state, terminal)
            current_state = np.reshape(next_state, (1, 84, 84, 4))
            eps_reward += reward
            if step > MEMORY_SIZE:
                if (step - MEMORY_SIZE) % TRAINING_INTERVAL == 0:
                    loss = q_network.train()
                    losses.append(loss)
                if step % COPY_WEIGHTS_INTERVAL == 0:
                    copy_weights(sess, assign_weight_group)
            if terminal:
                episode += 1
                episodic_reward.append(eps_reward)
                break
    # plot_graphs(episodic_reward, evaluation_rewards, losses)
    print("PROGRAM FINISHED.")
    return episodic_reward, evaluation_rewards, losses

atari()