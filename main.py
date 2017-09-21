import numpy
import matplotlib.pyplot as pyplot
import matplotlib.patches as mpatches
from neural_net import AutoEncoder

d_x = 784     #28*28=784
d1 = 128
d2 = 64
d_y = 32
layers = [d_x, d1, d2, d_y, d2, d1, d_x]     #layers[i] = layer's-i layers. layer-0 is input layer.

auto_encoder = AutoEncoder(layers)
auto_encoder.train(total_epochs=200, batch_size=250)
auto_encoder.delete()


'''
encoder_output = auto_encoder.test()

# Plot the reconstructed output as a scatterplot. Each axis is one of the two neurons in the decoder's output layer.
test_labels = numpy.transpose(AutoEncoder.mnist_dataset.test.labels)  # result is [classes, test_samples]
numbers = [[] for _ in range(AutoEncoder.num_of_classes())]  # classifying the encoded digits. Size is [classes, samples_of_digits_for_that_class, 2]
for i in range(0, AutoEncoder.test_samples()):
    numbers[numpy.argmax(test_labels[:, i])].append(encoder_output[:, i])

for i in range(0, AutoEncoder.num_of_classes()):
    numbers[i] = numpy.array(numbers[i])  # converting the lists into numpy narray objects for purposes of multi-axis slicing

figure = pyplot.figure(1)
# Display in two rows. Each row has a graph of 5 numbers.
for i in range(0, 2):
    startingNum = i*(int)(AutoEncoder.num_of_classes() / 2)  # scatterPlots per subplot.

    pyplot.subplot(2, 1, (i + 1))
    pyplot.plot(numbers[startingNum + 0][:, 0], numbers[startingNum + 0][:, 1], 'g.',
                numbers[startingNum + 1][:, 0], numbers[startingNum + 1][:, 1], 'r.',
                numbers[startingNum + 2][:, 0], numbers[startingNum + 2][:, 1], 'b.',
                numbers[startingNum + 3][:, 0], numbers[startingNum + 3][:, 1], 'k.',
                numbers[startingNum + 4][:, 0], numbers[startingNum + 4][:, 1], 'm.',
                )
    pyplot.title(s=('Numbers in range [' + str(startingNum) + ', ' + (str(startingNum + 4)) + ']'))
    pyplot.xlabel('Neuron 1')
    pyplot.ylabel('Neuron 2')

    pyplot.legend(loc=1,
                  handles=[mpatches.Patch(color='g', label=('Number ' + str(startingNum + 0))),
                           mpatches.Patch(color='r', label=('Number ' + str(startingNum + 1))),
                           mpatches.Patch(color='b', label=('Number ' + str(startingNum + 2))),
                           mpatches.Patch(color='k', label=('Number ' + str(startingNum + 3))),
                           mpatches.Patch(color='m', label=('Number ' + str(startingNum + 4)))]
                  ).draggable()

figure.tight_layout()
pyplot.show()
'''

'''
# Display in 4 rows and 3 columns. (10 seperate subplots)
color_cycle = ['b', 'r', 'k']
for i in range(0, AutoEncoder.num_of_classes()):
    cur_color = color_cycle[i % len(color_cycle)]

    pyplot.subplot(4, 3, (i + 1))
    pyplot.plot(numbers[i][:, 0], numbers[i][:, 1], (cur_color + '.'))
    pyplot.title(s=('Number ' + str(i)))
    pyplot.xlabel('Neuron 1')
    pyplot.ylabel('Neuron 2')

    pyplot.legend(loc=1, handles=[mpatches.Patch(color=cur_color, label=('Number ' + str(i)))]).draggable()
figure.tight_layout()
pyplot.show()
'''

