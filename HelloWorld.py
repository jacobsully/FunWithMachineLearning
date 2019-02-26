# activation function is the key
# linear vs non-linear regression vs logistic
# linear: y = mx + b
import tensorflow as tf 
import random
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

'''
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt 
'''
# one-hot vector -> 0 in all but one dimension
# softmax regression -> add up evidence of input being in a certian class
# softmax regression -> convert evidence into probabilities

learning_rate = 0.01
training_iteration = 30
batch_size = 100
display_step = 2


#placeholder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

#weights and biases
w = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

#softmax regression model
with tf.name_scope("Wx_b") as scope:
    model = tf.nn.softmax(tf.matmul(x, w) + b)

w_h = tf.summary.histogram("Weights", w)
b_h = tf.summary.histogram("biases", b)

with tf.name_scope("cost_function") as scope:
    cost_function = -tf.reduce_sum(y * tf.log(model))
    tf.summary.scalar("cost_function", cost_function)

with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

#cross-entropy loss model
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
#train_step = tf.train.GradientDescentOptimizer(0.4).minimize(cross_entropy)

init = tf.initialize_all_variables()

merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter('./logs', sess.graph)

    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict = {x: batch_xs, y: batch_ys})
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
            summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration * total_batch + i)
        if iteration % display_step == 0:
            print "Iteration:" '%04d' % (iteration + 1), "cost=", "{:9f}".format(avg_cost)

    print "Tuning completed!"

    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print "Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})




'''
x = [3,4,5]
h = [2,1,0]

y = np.convolve(x,h)
print(y)
'''

'''
# activation functions

step function
sigmoid function
    logistic, arc tangent, hyperbolic tangent (tanH)

'''


'''
# logistic regression steps

tf.matmul(x, weights)
tf.add(weighted_x, bias)
tf.nn.sigmoid(weighted_x_with_bias)
'''


'''
plt.rcParams['figure.figsize'] = (10, 6)

x = np.arange(0.0, 5.0, 0.1)
a = 1
b = 0
y = a * x + b

plt.plot(x,y)
plt.show()


x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 3 + 2
y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.1))(y_data)

a = tf.Variable(1.0)
b = tf.Variable(0.2)
y = a * x_data + b

# loss here is the squared error
# difference between prediction and actual value
loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

train_data = []
for step in range(100):
    evals = sess.run([train,a,b])[1:]
    if step % 5 == 0:
        print(step,evals)
        train_data.append(evals)

converter = plt.colors
cr, cg, cb = (1.0, 1.0, 0.0)
'''

'''
a = tf.constant([2])
b = tf.constant([3])

c = tf.add(a,b)

with tf.Session() as session:
    result = session.run(c)
    print(result)
#-------------------------
state = tf.Variable(0)
state1 = tf.Variable(1)
init_op = tf.initialize_all_variables()
'''


