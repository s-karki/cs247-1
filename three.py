import numpy as np
import tensorflow as tf

num_points = 5

x_batch = tf.placeholder(tf.float32)
y_batch = tf.Variable(tf.random_uniform([num_points],-1,1))

allLosses = tf.square(x_batch - y_batch)
loss = tf.reduce_mean(allLosses)

set1 = []

for i in xrange(num_points):
    x = np.random.randint(0,10)
    set1.append(x)

sess = tf.Session() 
init = tf.global_variables_initializer()
sess.run(init)

print sess.run(x_batch,feed_dict={x_batch:set1})
print sess.run(y_batch)
                      
print sess.run(allLosses,feed_dict={x_batch:set1})
print sess.run(loss,feed_dict={x_batch:set1})

optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

for step in xrange(100):
    sess.run(train,feed_dict={x_batch:set1})
    print sess.run(loss,feed_dict={x_batch:set1})
    print sess.run(x_batch,feed_dict={x_batch:set1})
    print sess.run(y_batch)


    
 
