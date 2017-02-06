import tensorflow as tf


a = tf.placeholder("float") 
b = tf.placeholder("float")
y = tf.mul(a, b)

sess = tf.Session()

print(sess.run(y, feed_dict={a: [1,4,7], b: [2,1,7]}))
print(sess.run(y, feed_dict={a: [3,2,2], b: [2,1,7]}))

print tf.__version__
