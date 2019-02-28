# KNU 딥러닝 특강 [2019.02.25~28]

## [2019.02.25] Python & numpy

[딥러닝(Deep-Learning)을 위한 프로그램 설치](https://m.post.naver.com/my/series/detail.nhn?memberNo=8098532&seriesNo=459452&prevVolumeNo=15102526)

아래 링크의 포스트를 순서대로 따라가면서 그래픽 드라이버, CUDA, CUDNN, Anaconda, tensorflow 를 설치한다.
내 노트북의 그래픽카드는 GeForce MX150 이므로 tensorflow_gpu-1.12.0 을 설치하기 위해 CUDA 9.0을 설치하고, 이에 맞추어 CUDNN 7.4.2의 라이브러리를 다운받아 추가하였다.
또한 CUDA 9.0이 Python 3.7을 지원하지 않으므로 Python의 버전을 3.6.8로 다운그레이드하였다.
가상환경 상의 파이썬에서 tensorflow를 구동시키기 위하여 Anaconda3을 설치한다.


### Colab
구글에서 제공하는 Jupyter notebook 기반의 온라인 딥러닝 구동 환경이다. 구글 드라이브, Github 등과 연동하여 사용할 수 있어 편리하다.
교육, 학습용으로 제공되는 서비스이기 때문에 일정 시간이상 딥러닝을 구동시키면 실행이 초기화된다.
하지만 이는 tensorflow의 기본적인 기능을 학습하는 데에 큰 지장은 없다.



## [2019.02.26] MLP

MLP : Mulitlayered Perceptron

### Iris

## [2019.02.27] CNN

CNN : Convolutional Neural Network

### MNIST 

```python
# Load the necessary package
## Packages and Data

import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets.mnist import load_data

# Load MNIST data
(x_train, y_train), (x_test, y_test) = load_data()

# Transform to 4-dim tensors

# print(x_train.shape)    # (60000, 28, 28)
x_train = np.expand_dims(x_train, axis=-1)   # axis=-1 : 텐서의 맨 뒤(인덱스:-1)를 expand
x_test = np.expand_dims(x_test, axis=-1)
# print(x_train.shape)    # (60000, 28, 28, 1)

## Normalization [0, 255] -> [0,1]
print(x_train.max())
x_train = x_train / 255
x_test = x_test / 255

# #Allocate placeholders for I/O
x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
y = tf.placeholder(tf.int64, shape=[None])

## Integer label -> Binary, one-hot label
y_onehot = tf.one_hot(y, 10)
print(y_onehot.shape)

## Dropout probability
keep_prob = tf.placeholder(tf.float32)

# Network Setting
## Convolution layer 1

# tf.truncated_normal : 2시그마 내의 정규분포
W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 1, 64], stddev=5e-2)) 
# [5, 5, 1, 64] : 흑백이라서 1 => 컬러면 rgb 채널 3개 추가 : [5, 5, 4, 64]
b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


## Convolution layer 2
W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

## Fully connedted layer 1
W_fc1 = tf.Variable(tf.truncated_normal(shape=[7*7*64, 384], stddev=5e-2))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

## Fully connected layer 2
W_fc2 = tf.Variable(tf.truncated_normal(shape=[384, 10], stddev=5e-2))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10]))
logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_pred = tf.nn.softmax(logits)

# LOSS AND OPTIMIZER
# Cross entropy loss with doftmax output
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_onehot, logits=logits))

# Adam optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

#Accuracy calculation
correct_prediction = tf.equal(tf.argmax(y_pred, 1), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




# LEARGNING STEPS

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        idx = np.random.randint(x_train.shape[0], size=128)
        batch = (x_train[idx], y_train[idx])
        if i%100==0:
            train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y:batch[1], keep_prob:1.})
            loss_print = loss.eval(feed_dict={x:batch[0], y:batch[1], keep_prob:1.})
            print("step:%d, acc:%f, loss:%f" %(i, train_accuracy, loss_print))
        sess.run(train_step, feed_dict={x:batch[0], y:batch[1], keep_prob:0.5})
    test_accuracy = accuracy.eval({x:x_test, y:y_test, keep_prob:1.})
    print("test acc:%f" %test_accuracy)
```


![](images/MNIST_CNN.png)


## [2019.02.28] Keras

