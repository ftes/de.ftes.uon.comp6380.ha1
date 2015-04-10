__author__ = 'Fredrik'

#training parameters for neural networks:
learning_rate = 0.05 #set in [0,1]

learning_decay = 0.999 #try 0.999, set in [0.9,1]

momentum = 0.01 # set in [0,0.5]

batch_learning = False #set to learn in batches

validation_proportion = 0. # set in [0,0.5]

hidden_layers = [50] #number of neurons in each hidden layer, make as many layers as you feel like. Try increasing this to 10

iterations = 15 #used only if validaton proportion is 0

#in_file = 'data-with-headers-tabs.txt'
#margins = 0., 1.
#in_file = 'denser-data-with-headers-tabs.txt'
#margins = -1., 1.
in_file = 'data-four.csv'
margins = -1., 1.

from pybrain.datasets            import ClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer, RPropMinusTrainer
from pybrain.structure.modules   import SoftmaxLayer
from pybrain.structure.modules import TanhLayer, LinearLayer


# http://stackoverflow.com/questions/10133386/pybrain-loading-data-with-numpy-loadtxt
import numpy
array = numpy.loadtxt(in_file, delimiter='\t', skiprows=1)
number_of_columns = array.shape[1]
dataset = ClassificationDataSet(number_of_columns - 1, target=1, nb_classes=4)
for row in array:
    dataset.addSample(row[:-1], row[-1:])

dataset._convertToOneOfMany(bounds=[0.,1.])

print "Number of training patterns: ", len(dataset)
print "Input and output dimensions: ", dataset.indim, dataset.outdim

fnn = buildNetwork( dataset.indim, 77, 77, dataset.outdim, hiddenclass=TanhLayer
                    , outclass=LinearLayer
    )


#trainer = BackpropTrainer( fnn, dataset=dataset, momentum=0.5, learningrate=0.005, verbose=True)
trainer = RPropMinusTrainer(fnn, dataset=dataset)

#from scipy import diag, arange, meshgrid, where
import scipy

# graph spiral
step = 0.05
ticks = scipy.arange(margins[0], margins[1] + step, step)
#ticks = arange(-1.05,1.05,0.05)
X, Y = scipy.meshgrid(ticks, ticks)
# need column vectors in dataset, not arrays
griddata = ClassificationDataSet(2,1, nb_classes=2)
for i in xrange(X.size):
    griddata.addSample([X.ravel()[i],Y.ravel()[i]], [0])
griddata._convertToOneOfMany()  # this is still needed to make the fnn feel comfy



error = []

for i in range(2000):
    trainer.train()

    trnresult = percentError( trainer.testOnClassData(),
                              dataset['class'] )
    error.append(trnresult)

    print "epoch: %4d" % trainer.totalepochs, \
          "  train error: %5.2f%%" % trnresult

    if i % 10 == 0:

        out = fnn.activateOnDataset(griddata)
        out = out.argmax(axis=1)  # the highest output activation gives the class
        out = out.reshape(X.shape)


        #from pylab import ion, ioff, figure, draw, contourf, clf, show, hold, plot
        import pylab
        from matplotlib import pyplot

        pyplot.subplot(211)
        pylab.figure(1)
        pylab.ioff()  # interactive graphics off
        pylab.clf()   # clear the plot
        pylab.hold(True) # overplot on
        for c in range(0,4):
            here, _ = scipy.where(dataset['class']==c)
            pylab.plot(dataset['input'][here,0],dataset['input'][here,1],'o')
        if out.max()!=out.min():  # safety check against flat field
            pylab.contourf(X, Y, out)   # plot the contour
        pylab.ion()   # interactive graphics on
        pylab.draw()  # update the plot


        #visualise training error --------------------------------------------------------------------------------------------------------------------------------------------------

        #set up a figure
        pyplot.figure(1)

        #select the first of 2 subplots
        pyplot.subplot(212)

        #graph the error
        err, = pyplot.plot(error,color='r')

        #set the legend for the graph
        legend = [[err],["Error"]]


        #set the X and Y axis labels, and create the legend
        pyplot.ylabel('Error')
        pyplot.xlabel('Training Iterations')
        pyplot.legend(*legend)
        #pyplot.draw()

pylab.ioff()
pylab.show()
pyplot.show()