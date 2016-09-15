import numpy as np
import tensorflow as tf

class lstm_layer:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        self.Wgx = tf.Variable(tf.random_normal((input_size, output_size)))
        self.Wix = tf.Variable(tf.random_normal((input_size, output_size)))
        self.Wfx = tf.Variable(tf.random_normal((input_size, output_size)))
        self.Wox = tf.Variable(tf.random_normal((input_size, output_size)))

        self.Wgh = tf.Variable(tf.random_normal((output_size, output_size)))
        self.Wih = tf.Variable(tf.random_normal((output_size, output_size)))
        self.Wfh = tf.Variable(tf.random_normal((output_size, output_size)))
        self.Woh = tf.Variable(tf.random_normal((output_size, output_size)))

        self.bg = tf.Variable(tf.random_normal((1, output_size)))
        self.bi = tf.Variable(tf.random_normal((1, output_size)))
        self.bf = tf.Variable(tf.random_normal((1, output_size)))
        self.bo = tf.Variable(tf.random_normal((1, output_size)))

    def get_var_list(self):
        return self.Wgx, self.Wix, self.Wfx, self.Wox, self.Wgh, self.Wih, \
            self.Wfh, self.Woh, self.bg, self.bi, self.bf, self.bo

    # compute output from input for one element (non-sequential)
    # x: size (num_examples, input_size)
    # s_prev, h_prev: size (num_examples, output_size)
    # return s, h: each size (num_examples, output_size)
    def forward_prop(self, x, s_prev, h_prev):
        g = tf.tanh(tf.matmul(x, self.Wgx) + tf.matmul(h_prev, self.Wgh) + self.bg)
        i = tf.sigmoid(tf.matmul(x, self.Wix) + tf.matmul(h_prev, self.Wih) + self.bi)
        f = tf.sigmoid(tf.matmul(x, self.Wfx) + tf.matmul(h_prev, self.Wfh) + self.bf)
        o = tf.sigmoid(tf.matmul(x, self.Wox) + tf.matmul(h_prev, self.Woh) + self.bo)
        s = g * i + f * s_prev
        h = tf.tanh(s) + o
        return s, h

    # train by gradient descent
    # y: size (num_examples, output_size)
    # learning_rate: scalar
    # return: gradient descent training module
    def training_operation(self, x, s_prev, h_prev, y, learning_rate):
        h = self.forward_prop(x, s_prev, h_prev)[1]
        loss = tf.reduce_mean(tf.square(h - y))
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        return optimizer.minimize(loss), loss

    def run(self, x, s_prev, h_prev, y, learning_rate, num_epochs):
        train, loss = self.training_operation(x, s_prev, h_prev, y, learning_rate)
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        for step in range(num_epochs):
            sess.run(train)
            print(sess.run(loss))
        return sess

alphabet = list("abcdefghijklmnopqrstuvwxyz")
char_to_ind = dict((c,i) for i, c in enumerate(alphabet))
def char_to_vec(c):
    vec = np.zeros((len(alphabet)))
    vec[char_to_ind[c]] = 1
    return vec
def vec_to_char(vec):
    ind = np.argmax(vec)
    return alphabet[ind]

def test_lstm_layer():
    num_examples = 5
    input_size = 7
    output_size = 6
    x_data = tf.constant(np.random.randn(num_examples, input_size), dtype=tf.float32)
    y_data = tf.constant(np.random.randn(num_examples, output_size), dtype=tf.float32)
    s_prev = tf.constant(np.random.randn(num_examples, output_size), dtype=tf.float32)
    h_prev = tf.constant(np.random.randn(num_examples, output_size), dtype=tf.float32)
    layer = lstm_layer(input_size, output_size)
    layer.run(x_data, s_prev, h_prev, y_data, 0.5, 10000)

def test_lstm_layer_chars():
    inp_str = "abcdef"
    outp_str = "ghijkl"
    x_data = tf.constant(np.array([char_to_vec(c) for c in inp_str]), dtype=tf.float32)
    y_data = tf.constant(np.array([char_to_vec(c) for c in outp_str]), dtype=tf.float32)
    num_examples = len(inp_str)
    input_size = len(alphabet)
    output_size = input_size
    s_prev = tf.constant(np.zeros((num_examples, output_size)), dtype=tf.float32)
    h_prev = tf.constant(np.zeros((num_examples, output_size)), dtype=tf.float32)
    layer = lstm_layer(input_size, output_size)
    sess = layer.run(x_data, s_prev, h_prev, y_data, 1.0, 5000)
    s, h = layer.forward_prop(x_data, s_prev, h_prev)
    outp = str([vec_to_char(vec) for vec in h.eval(session=sess)])
    print(outp)

if __name__ == "__main__":
    test_lstm_layer_chars()
