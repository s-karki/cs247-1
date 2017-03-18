import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

print 'Training   Set count ', mnist.train.images.shape
print 'Testing Set count '   , mnist.test.images.shape
print 'Validation Set count ', mnist.validation.images.shape

#the length of each input is determined by the data file

#epocs was 10000
training_size = 1000
epochs = 10000
howOften = 500
batch_size = 10
beta = 0.0001
# default hyperparameters

hiddenLayerSize = 30
learning_rate = 2  # was 0.1
momentum = 0.9

print sys.argv

for o in range(1,len(sys.argv),2):
    print o
    arg = sys.argv[o]
    print arg
    if arg in ['-h', '-hiddens']:
        hiddenLayerSize = int(sys.argv[o+1])
    elif arg in ['-l','-learningRate']:
        learning_rate = float(sys.argv[o+1])
    elif arg in ['-e', '-epochs']:
        epochs = int(sys.argv[o+1])
    elif arg in ['-t', '-trainsize']:
        training_size = int(sys.argv[o+1])
    elif arg in ['-b', '-batchsize']:
        batch_size = int(sys.argv[o+1])

points = mnist.test.images[:training_size,:]
pointsA = mnist.test.labels[:training_size,:] # initially, we had a small training set

train = mnist.train.images[:training_size,:]
trainA = mnist.train.labels[:training_size,:]  #we now move to a larger training set

validation = mnist.validation.images
validationA = mnist.validation.labels

inputLayerSize = len(points[0])
outputLayerSize = 10

print "epochs            ", epochs
print "input layer size  ", inputLayerSize
print "hidden layer size ", hiddenLayerSize
print "learning rate     ", learning_rate
print "training size     ", len(points)
print "validation size   ", len(validation)
print "momentum          ", momentum
print "beta              ", beta

def display_digit(X):
    image = X.reshape([28,28])
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

def plot_loss(valid, training):
    plt.plot(valid, 'ro')
    plt.plot(training, 'bo')
    plt.show()
    


# some number of inputs, each of which is the size of the input layer

x = tf.placeholder(dtype=tf.float32, shape=[None, inputLayerSize], name="inputData")
y = tf.placeholder(dtype=tf.float32, shape=[None, outputLayerSize], name="outputData")

weightsInHid = tf.Variable(tf.random_normal([inputLayerSize, hiddenLayerSize], dtype=tf.float32), name='weightsInHid')
biasesHid = tf.Variable(tf.zeros([hiddenLayerSize]), name='biasesHid')
HidIn = (tf.matmul(x, weightsInHid) + biasesHid)
encoded = tf.nn.sigmoid(HidIn)

weightsHidOut = tf.Variable(tf.random_normal([hiddenLayerSize, outputLayerSize], dtype=tf.float32), name='weightsHidOut')
biasesOut = tf.Variable(tf.zeros([outputLayerSize]), name='biasesOut')
decoded = tf.nn.sigmoid(tf.matmul(encoded, weightsHidOut) + biasesOut)

#l2_term =  tf.nn.l2_loss(weightsHidOut)
l2_term = beta * (tf.nn.l2_loss(weightsInHid) + tf.nn.l2_loss(weightsHidOut))
loss_no_l2 = tf.reduce_mean(tf.square(tf.sub(y, decoded)))

loss =  loss_no_l2 + l2_term
train_op = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss)

num_samples = len(points)
print 'Number of Samples',num_samples

sum1 = tf.summary.scalar("training loss", loss)
sum2 = tf.summary.scalar("total loss", loss)

init = tf.global_variables_initializer()
sess = tf.Session() 
sess.run(init)


test_writer = tf.summary.FileWriter('./test', sess.graph)


def otherIndexProb(arr, i):
    j = 0
    for x in range(len(arr)):
        if j != i and  arr[0][j] > 0.85:
            return True
        j = j + 1
    return False
           
def checkErrors(ins,outs,flag=False):
    errors = 0
    print "Number of Tests ",len(ins)
    for k in range(len(ins)):
        l, d = sess.run([loss ,decoded], feed_dict={x: [ins[k]], y:[outs[k]]}) #acc, loss to accuracy
        
        # A more precise classifier
        # Find the number that algo should predict
        # See if P(N) < 0.85, or if P(~N) > 0.85. In either case
        # we say the program has incorrectly classified

        i = outs[k].tolist().index(1)
        if d[0][i] < 0.75 or otherIndexProb(d, i) :
            errors = errors + 1

        #if (l > 0.05):
        #   errors = errors +1

        if flag:
            print "test number and error", k, l
            print "predicted"
            print d
            print "desired"
            print outs[k]
            
    print "Total probable errors ", errors

validLoss = []
trainingLoss = []

for i in range(epochs):
    trainPoints = []
    trainPointsA = []

    for j in range(batch_size):
        r = np.random.randint(0,num_samples)
        trainPoints.append(train[r])
        trainPointsA.append(trainA[r])
                          
    l, _ = sess.run([loss, train_op], feed_dict={x: trainPoints, y: trainPointsA}) # different training set (actual training set)
    write1 = sess.run(sum1, feed_dict={x: trainPoints,y:trainPointsA}) 
    test_writer.add_summary(write1,i*epochs+j)
    if i % howOften == 0:
        print 'epoch ',i
        big_loss = sess.run(loss_no_l2,feed_dict={x:train,y:trainA})
        valid_loss = sess.run(loss_no_l2,feed_dict={x:validation,y:validationA})
        #l2 = sess.run(l2_term,feed_dict={x:validation,y:validationA})
        
        trainingLoss.append(big_loss)
        validLoss.append(valid_loss)

        print 'Total loss', big_loss
        print 'Validation Set Loss', valid_loss 
        write2 = sess.run(sum2, feed_dict={x:train,y:trainA})
        test_writer.add_summary(write2,i)

        print 'Test Set Errors'
        checkErrors(points,pointsA)
       
        print "Validation errors"
        checkErrors(validation, validationA)
       

checkErrors(points, pointsA)

checkErrors(validation, validationA,False)

plot_loss(validLoss, trainingLoss)

exit()
