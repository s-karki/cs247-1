import tensorflow as tf
a = [1,2,3]
b = [2,5,7]
#a = tf.placeholder("float") 
#b = tf.placeholder("float")

y = tf.mul(a, b)
sess = tf.Session()

print(sess.run(y))

a.append(117)
b.append(92)
print(sess.run(y))

a = [2,3,4]
b = [12,13,24]

print(sess.run(y))

y = tf.mul(a,b)
print(sess.run(y))

