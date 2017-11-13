
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

#1.3.0-0
print(tf.__version__)
print(np.__version__)
#设置种子数
tf.set_random_seed(1)
np.random.seed(1)

#超参数（自定义的参数）
BATCH_SIZE = 64
TIME_STEP = 28
INPUT_SIZE =28
LR = 0.01

#数据
mnist = input_data.read_data_sets('./mnist', one_hot=True)
test_x = mnist.test.images[:2000]
test_y = mnist.test.labels[:2000]

#输出测试
print(mnist.train.images.shape)
print(mnist.train.labels.shape)
plt.imshow(mnist.train.images[0].reshape((28,28)),cmap='gray')
plt.title('%i' % np.argmax(mnist.train.labels[0]))
plt.show()

#tensorflow placeholders
##def placeholder(dtype, shape=None, name=None)：shape(batch , 28*28=784)
tf_x = tf.placeholder(tf.float32,[None, TIME_STEP *  INPUT_SIZE])
##def reshape(tensor, shape, name):shape(batch, height, width, channel)
image = tf.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE])
tf_y = tf.placeholder(tf.int32, [None, 10])

#RNN结构
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=64)
##def dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
##                dtype=None, parallel_iterations=None, swap_memory=False,
##                time_major=False, scope=None):
outputs, (h_c,h_n) = tf.nn.dynamic_rnn(
    rnn_cell,           #An instance of RNNCell
    image,              #inputs
    initial_state=None, #初始化的隐藏状态
    dtype=tf.float32,   #隐藏状态为None时,dtype必须设置
    time_major=False,   #False: (batch, time step, input); True: (time step, batch, input)
)
print(outputs)
output = tf.layers.dense(outputs[:, -1, :], 10)
##softmax算法,
loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
##adam(学习率)
train_op = tf.train.AdamOptimizer(LR).minimize(loss)
##def accuracy(labels, predictions, weights=None, metrics_collections=None,
##             updates_collections=None, name=None):　return (acc, update_op), and create 2 local variables
###def argmax(input,axis=None,name=None,dimension=None,output_type=dtypes.int64):
accuracy = tf.metrics.accuracy(
    labels=tf.argmax(tf_y,axis=1), predictions=tf.argmax(output,axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
# 在图中初始化参数
sess.run(init_op)

#训练
for step in range(1200):
    ##返回值,return self._images[start:end], self._labels[start:end]
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    ##run(self, fetches, feed_dict=None, options=None, run_metadata=None) :Runs operations and evaluates tensors in `fetches`.
    _, loss_ = sess.run([train_op,loss], {tf_x: b_x, tf_y: b_y})
    if step % 50 == 0:
        accuracy_ =sess.run(accuracy, {tf_x:test_x, tf_y:test_y})
        print('train loss: %.4f' % loss_, '| test accuracy: %.2f' % accuracy_)

#输出10个预测值
test_output = sess.run(output, {tf_x:test_x[:10]})
pred_y = np.argmax(test_output, 1)
print(pred_y, 'prediction number')
print(np.argmax(test_y[:10], 1), 'real number')