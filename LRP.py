from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')

import tempfile, sys, os
sys.path.insert(0, os.path.abspath('..'))

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

# Download and import MNIST data
temp_dir = tempfile.gettempdir()
mnist = input_data.read_data_sets(temp_dir, one_hot=True)

# Parameters
lr = 0.005
epochs = 2000
batch = 128

# Network Parameters
no_of_hidden_1 = 256 # 1st layer number of neurons
no_of_hidden_2 = 256 # 2nd layer number of neurons
input_ = 784 # MNIST data input (img shape: 28*28)
no_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, input_])
Y = tf.placeholder("float", [None, no_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([input_, no_of_hidden_1], mean=0.0, stddev=0.05)),
    'h2': tf.Variable(tf.random_normal([no_of_hidden_1, no_of_hidden_2], mean=0.0, stddev=0.05)),
    'out': tf.Variable(tf.random_normal([no_of_hidden_2, no_classes], mean=0.0, stddev=0.05))
}
biases = {
    'b1': tf.Variable(tf.zeros([no_of_hidden_1])),
    'b2': tf.Variable(tf.zeros([no_of_hidden_2])),
    'out': tf.Variable(tf.zeros([no_classes]))
}

# Create and train model
def model(x, act=tf.nn.relu):  # < different activation functions lead to different explanations
    layer_1 = act(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    layer_2 = act(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
log = model(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=log, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=lr)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(log, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Train
def input_transform (x): 
    return (x - 0.5) *  2

sess = tf.Session()

# Run the initializer
sess.run(init)

for step in range(1, epochs+1):
    batch_x, batch_y = mnist.train.next_batch(batch)
    batch_x = input_transform(batch_x)
    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
    if step % 100 == 0 or step == 1:
        # Calculate batch loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                             Y: batch_y})
        print("Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))

# Calculate accuracy for MNIST test images
test_x = input_transform(mnist.test.images)
test_y = mnist.test.labels

print("Test accuracy:", \
    sess.run(accuracy, feed_dict={X: test_x, Y: test_y}))

# Import DeepExplain: PreDefined Library
from methods import DeepExplain
from utils import plot, plt
import warnings
warnings.filterwarnings('ignore')

# Define the input to be tested
test_image_index = 17
test_image = test_x[[test_image_index]]
test_image_prediction = test_y[test_image_index] 

with DeepExplain(session=sess) as de:
    log = model(X)
    attributions = {
    	'Gradient * Input': de.explain('grad*input', log * test_image_prediction, X, test_image),
        'Epsilon-LRP': de.explain('elrp', log * test_image_prediction, X, test_image)
    }

# Plot attributions
n_cols = len(attributions) + 1
fig, axes = plt.subplots(nrows=1, ncols=n_cols, figsize=(3*n_cols, 3))
plot(test_image.reshape(28, 28), cmap='Greys', axis=axes[0]).set_title('Original')
print(test_image_prediction)
for i in range(len(test_image_prediction)):
    if test_image_prediction[i]==float(1):
        print("Test output : ",i)
for i, method_name in enumerate(sorted(attributions.keys())):
    plt.savefig(plot(attributions[method_name].reshape(28,28), xi = test_image.reshape(28, 28), axis=axes[1+i]).set_title(method_name))