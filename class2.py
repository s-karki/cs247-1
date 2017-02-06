import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def printXY(xs,ys,lab):
    plt.plot(xs, ys,'ro',label =lab)
    plt.legend()
    plt.show()


num_points = 1000 
train_set = []
batch_size = 1
batch_iter = 2000
print_how_often = 100

#generate the points to be used in the learning experiment

for i in xrange(num_points):
    x1= np.random.normal(0.0, 0.55)
    y1= x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03) 
    z1= x1 * 0.2 + 0.4 + np.random.normal(0.0, 0.10)
    train_set.append([x1, y1, z1])
   


# extract the x and y coordinates of the points

x_train = [v[0] for v in train_set] 
y_train = [v[1] for v in train_set]
z_train = [v[2] for v in train_set] 

printXY(x_train,y_train,'Original data')

" generate a test set"
num_test = 500
test_set = []

for i in xrange(num_test):
    x1= np.random.normal(0.0, 0.55)
    y1= x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    z1= x1 * 0.2 + 0.4 + np.random.normal(0.0, 0.10) 
    test_set.append([x1, y1, z1])

x_test = [v[0] for v in test_set]
y_test = [v[1] for v in test_set]
z_test = [v[2] for v in test_set] 


printXY(x_test,y_test,'Test Data')


# Set up the tensorflow model   placeholders for the batch training data 

x_batch = tf.placeholder(tf.float32)
y_batch = tf.placeholder(tf.float32)
z_batch = tf.placeholder(tf.float32)

# learning parameters
#

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
V = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) 
b = tf.Variable(tf.zeros([1]))

z_train_predicted = W * x_train + V * y_train + b
z_test_predicted = W * x_test + V * y_test + b
z_batch_predicted = W * x_batch + V * y_batch + b

'''
y_train_predicted = W * x_train + b
y_test_predicted = W * x_test + b 
y_batch_predicted = W * x_batch + b
'''

trainloss = tf.reduce_mean(tf.square(z_train_predicted - z_train))
testloss = tf.reduce_mean(tf.square(z_test_predicted - z_test))


train_loss_history = []
test_loss_history = []
steps = []


batchloss = tf.reduce_mean(tf.square(z_batch - z_batch_predicted))
optimizer = tf.train.GradientDescentOptimizer(0.1) 


train = optimizer.minimize(batchloss)


init = tf.global_variables_initializer()
sess = tf.Session() 
sess.run(init)


yC =[]
xC =[]
zC = []

for step in xrange(batch_iter):
    
    print "W, V and b: ",sess.run(W), sess.run(V),sess.run(b)

    del xC[:]
    del yC[:]
    del zC[:]

    if batch_size == num_points:
        xC.extend(x_train)
        yC.extend(y_train)
        zC.extend(z_train)
    elif batch_size == 1:
        xC.append(x_train[step % 1000])
        yC.append(y_train[step % 1000])
        zC.append(z_train[step % 1000])
    else:
        for step2 in xrange(batch_size):
            randOne = np.random.randint(0,num_points)
            xC.append(x_train[randOne])
            yC.append(y_train[randOne])
            zC.append(z_train[randOne]) 

    print '________________________'
    print'Data X: ', xC[0:20]
    print'Data Y: ', yC[0:20]
    print 'Batch Prediction  ', sess.run(z_batch_predicted,feed_dict={x_batch:xC, y_batch: yC})[0:20]
    print 'Batch Target  ', zC[0:20]
    print '________________________'

    sess.run(train,feed_dict={x_batch:xC,y_batch:yC, z_batch: zC})
 
 
    if step % print_how_often  == 0:
        print 'After step #',step, 'W: ', sess.run(W), 'V: ', sess.run(V), 'b: ', sess.run(b),' Loss:', sess.run(trainloss) #stop here 
        print 'Test loss ',sess.run(testloss)
        train_loss_history.append(sess.run(trainloss))
        test_loss_history.append(sess.run(testloss))
        steps.append(step)
    
        plt.plot(x_train,y_train,'ro',label = str(step))
 #   plt.plot(x_test,y_test,'bo',label = str(step))
        plt.plot(x_train,sess.run(W)*x_train+sess.run(b))
        plt.legend()
        plt.show()


plt.plot(x_train,y_train,'ro',label = str(step))
plt.plot(x_train,sess.run(W)*x_train+sess.run(b))
plt.legend()
plt.show()

print train_loss_history
print test_loss_history



plt.plot(steps,train_loss_history,'ro')
plt.plot(steps,test_loss_history,'bo')
plt.show()
