
import random
import numpy
import matplotlib.pyplot as pyplot
import matplotlib.patches as mpatches

classes = 10
test_samples = 1000

encoder_output = [] #[test_samples, 2]
encoded_sample_labels = []  #[test_samples, classes]
for i in range(0, test_samples):
    sample_label = [0] * classes
    sample_label[random.randint(0, classes-1)] = 1
    encoded_sample_labels.append(sample_label)

    encoded_sample = [random.randint(0, 10000), random.randint(0, 10000)]
    encoder_output.append(encoded_sample)

encoded_sample_labels = numpy.transpose(encoded_sample_labels)  #becomes [classes, test_samples]
numbers = [[] for _ in range(classes)]  #[classes, num_of_digits_for_that_class, 2]

for i in range(0, test_samples):
    numbers[numpy.argmax(encoded_sample_labels[:, i])].append(encoder_output[i])

for i in range(0, classes):
    numbers[i] = numpy.array(numbers[i])    #converting the lists into numpy narray objects

figure = pyplot.figure(1)
color_cycle = ['b', 'r', 'k']
for i in range(0, classes):
    pyplot.subplot(4, 3, (i+1))
    pyplot.plot(numbers[i][:, 0], numbers[i][:, 1], (color_cycle[i%len(color_cycle)]+'.'))
    pyplot.title(s=('Number ' + str(i)))
    pyplot.xlabel('Neuron 1')
    pyplot.ylabel('Neuron 2')

    pyplot.legend(loc=1, handles=[mpatches.Patch(color=color_cycle[i%len(color_cycle)], label=('Number ' + str(i)))]).draggable()
figure.tight_layout()
pyplot.show()



'''
scatterPlots = (int)(classes/2) # scatterPlots per subplot.
for i in range(0, 2):
    startingNum = (i*scatterPlots)

    pyplot.subplot(2, 1, (i+1))
    pyplot.plot(numbers[startingNum+0][:, 0], numbers[startingNum+0][:, 1], 'g.',
                numbers[startingNum+1][:, 0], numbers[startingNum+1][:, 1], 'r.',
                numbers[startingNum+2][:, 0], numbers[startingNum+2][:, 1], 'b.',
                numbers[startingNum+3][:, 0], numbers[startingNum+3][:, 1], 'k.',
                numbers[startingNum+4][:, 0], numbers[startingNum+4][:, 1], 'm.',
                )
    pyplot.title(s=('Numbers in range [' + str(startingNum) + ', ' + (str(startingNum+4)) +']'))
    pyplot.xlabel('Neuron 1')
    pyplot.ylabel('Neuron 2')

    pyplot.legend(loc=1,
                  handles=[mpatches.Patch(color='g', label=('Number ' + str(startingNum+0))), mpatches.Patch(color='r', label=('Number ' + str(startingNum+1))),
                           mpatches.Patch(color='b', label=('Number ' + str(startingNum+2))), mpatches.Patch(color='k', label=('Number ' + str(startingNum+3))),
                           mpatches.Patch(color='m', label=('Number ' + str(startingNum+4)))]
                  ).draggable()
figure.tight_layout()
pyplot.show()
'''

'''
pyplot.subplot(2, 1, 1)
pyplot.plot(numbers[0][:, 0], numbers[0][:, 1], 'g.',
            numbers[1][:, 0], numbers[1][:, 1], 'r.',
            numbers[2][:, 0], numbers[2][:, 1], 'b.',
            numbers[3][:, 0], numbers[3][:, 1], 'k.',
            numbers[4][:, 0], numbers[4][:, 1], 'm.',
            )
pyplot.title('Numbers in range [0, 4]')
pyplot.xlabel('Neuron 1')
pyplot.ylabel('Neuron 2')
pyplot.legend(loc=1, handles=[mpatches.Patch(color='g', label='Number 0'), mpatches.Patch(color='r', label='Number 1'),
                              mpatches.Patch(color='b', label='Number 2'), mpatches.Patch(color='k', label='Number 3'),
                              mpatches.Patch(color='m', label='Number 4')]
              ).draggable()

pyplot.subplot(2, 1, 2)
pyplot.plot(numbers[5][:, 0], numbers[5][:, 1], 'g.',
            numbers[6][:, 0], numbers[6][:, 1], 'r.',
            numbers[7][:, 0], numbers[7][:, 1], 'b.',
            numbers[8][:, 0], numbers[8][:, 1], 'k.',
            numbers[9][:, 0], numbers[9][:, 1], 'm.'
            )
pyplot.title('Numbers in range [5, 9]')
pyplot.xlabel('Neuron 1')
pyplot.ylabel('Neuron 2')
pyplot.legend(loc=1, handles=[mpatches.Patch(color='g', label='Number 5'), mpatches.Patch(color='r', label='Number 6'),
                              mpatches.Patch(color='b', label='Number 7'), mpatches.Patch(color='k', label='Number 8'),
                              mpatches.Patch(color='m', label='Number 9')]
              ).draggable()
'''
