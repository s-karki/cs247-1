import numpy as np
import tensorflow as tf

num_points = 5

x_batch = tf.placeholder(tf.float32)
y_batch = tf.placeholder(tf.float32)

allLosses = tf.square(x_batch - y_batch)
loss = tf.reduce_mean(allLosses)

set1 = []

for i in xrange(num_points):
    x = np.random.randint(0,10)
    set1.append([x])

set2 = []

for i in xrange(num_points):
    x = np.random.randint(0,10)
    set2.append([x])


sess = tf.Session() 

print set1[0:5]
print set2[0:5]

print sess.run(allLosses,feed_dict={x_batch:set1,y_batch:set2})
print sess.run(loss,feed_dict={x_batch:set1,y_batch:set2})
 
 
