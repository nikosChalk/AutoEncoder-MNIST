
import tensorflow as tf
import mnist
import numpy
import os
from tensorflow.examples.tutorials.mnist import input_data

class AutoEncoder:

    mnsit_dataset = input_data.read_data_sets("MNIST_data", one_hot=True)
        # One-hot=True means that the output of array will have only one 1 value, and the rest will be 0. (Only one active neuron in the output layer)

    def __init__(self, layers, learning_rate):
        """
        Creates an AutoEncoder with the given layers which uses the Stochastic Gradient Descent optimizer. Samples of this encoder
         are assumed to be in range [0, 1]
        :param layers: A list which contains the number of layers of each layer. This list is assumed to be symmetrical and with odd size
        :param learning_rate: A float value
        """
        if(len(layers)%2 == 0): # Checking if size is odd
            raise ValueError('Number of layers must be an odd number')

        first_half = layers[0: (int)(len(layers)/2)]
        second_half = layers[((int)(len(layers)/2)+1) : len(layers)]
        if(first_half[::-1] != second_half):    # Split the list into 2 sublists and compare them. Ignore the middle element.
            raise ValueError('List not symmetrical')

        self._layers = layers
        self._learning_rate = learning_rate
        self._optimizer = tf.train.GradientDescentOptimizer(self._learning_rate)
        self._d_x = layers[0]
        self._d_y = layers[(int)(len(layers)/2)]
        self._weight_matrix = []  # contains Tesnors which represent the weight arrays between layers. w[0] = weight matrix between input layer-0 and layer-1
        self._sess = tf.Session()
        self._writer = tf.summary.FileWriter('TensorBoard_logs/' + str(self._layers).replace(', ', '-') + '_lr=' + str(self._learning_rate))

        for i in range(1, len(layers)): #Define the initialization of the weights
            self._weight_matrix.append(tf.Variable(tf.truncated_normal([layers[i], layers[i - 1]], stddev=0.499), name=('w_matrix_' + str(i-1)) ))

        print('Initialization of Weight Variables...')
        self._sess.run(tf.global_variables_initializer())  #Initializes global variables and starts assessing the computation graph
        self.print_weights(0)
        print('Initializing of Weight Variables Done!\n')

    def print_weights(self, epoch):
        weights = self._sess.run(self._weight_matrix)
        for cur_weight_matrix in range (0, len(weights)):
            f = open(os.path.join('matrices', ('weight_matrix_' + str(cur_weight_matrix) + '_epoch_' + str(epoch) + '.txt')), 'w')
            f.write('Matrix_' + str(cur_weight_matrix) + '\n')
            for i in range (0, len(weights[cur_weight_matrix])):
                for j in range (0, len(weights[cur_weight_matrix][i])):
                    f.write(str(weights[cur_weight_matrix][i][j]) + '\t\t\t')
                f.write('\n')

    def _encode(self, input_data):
        """
        Encodes the given sample
        :param input_data: A Tensor of size [d_x, 1]
        :return: A Tensor of size [d_y, 1]
        """
        if (input_data.get_shape().dims != [self._d_x, 1]):
            raise ValueError('Input Tensor has wrong shape!')

        output = input_data
        for i in range(0, (int)(len(self._weight_matrix) / 2) - 1):
            output = self._fc_layer(self._weight_matrix[i], output, op_name=('hidden_layer_' + str(i+1)))
        output = self._enc_output_layer(self._weight_matrix[(int)(len(self._weight_matrix)/2) - 1], output)
        return output

    def _decode(self, data):
        """
        Decodes the given sample
        :param data: A Tensor of size [d_y, 1]
        :return: A Tensor of size [d_x, 1]
        """
        if (data.get_shape().dims != [self._d_y, 1]):
            raise ValueError('Input Tensor has wrong shape!')

        output = data
        for i in range((int)(len(self._weight_matrix)/2), len(self._weight_matrix) - 1):
            output = self._fc_layer(self._weight_matrix[i], output, op_name=('hidden_layer_' + str(i+1)))
        output = self._dec_output_layer(self._weight_matrix[len(self._weight_matrix) - 1], output)
        return output

    def _model_output(self, data):
        """
        Reconstucts the given sample by encoding it and then decoding it.
        :param data: A Tensor of size [d_x, 1]
        :return: A Tensor
        """
        return self._decode(self._encode(data))

    def _dataset_output(self, dataset):
        """
        Reconstructs all the samples from the given dataset
        :param dataset: A Tensor with d_x rows and any arbitary columns. Each column represents a data point
        :return: A Tensor with the same size as the input consisting of the reconstructed data points.
        """
        samples = dataset.get_shape().dims[1].value  # Tensor ---> TensorShape ---> Dimension list ---> int value
        y_hat = None  # Tensor with a shape same as the dataset
        for i in range(0, samples):  # For each sample in the batch do
            cur_sample = tf.slice(dataset, [0, i], [self._d_x, 1])  # Fetch the sample from column i
            cur_sample_output = self._model_output(cur_sample)  # Calculate the NN's output
            y_hat = (cur_sample_output) if (y_hat is None) else (tf.concat(concat_dim=1, values=[y_hat, cur_sample_output]))
        return y_hat

    def train(self, total_epochs, batch_size):
        """
        Trains the weights of the given Neural Network using the MNIST dataset
        :param total_epochs: The epochs that the NN should run. Must be >0
        :param batch_size: The size of each batch. Must be >0
        :return A numpy array containing the weight matrices
        """
        if(total_epochs<=0 or batch_size <=0):
            raise ValueError('Total epochs and batch size must be >0')

        x = tf.placeholder(dtype=tf.float32, shape=[self._d_x, batch_size], name='nn_input_data')  # [d_x, batch_size]
        tf.summary.image('input_images', tf.reshape(tf.transpose(x), [-1, 28, 28, 1]), max_outputs=4)

        y_hat = self._dataset_output(x)  # [d_x, batch_zie]
        tf.summary.image('output_images', tf.reshape(tf.transpose(y_hat), [-1, 28, 28, 1]), max_outputs=4)

        with tf.name_scope('cost', values=[x, y_hat]):
            cost = tf.mul(tf.constant(1/2, dtype=tf.float32), tf.reduce_sum(tf.square(tf.sub(x, y_hat)), axis=1))  # [d_x, 1]
        tf.summary.histogram('batch_cost', cost)

        gradients = self._optimizer.compute_gradients(cost)
        with tf.name_scope('gradients', values=[gradients]):
            self._optimizer.apply_gradients(gradients)
        for i in range (0, len(gradients)):
            tf.summary.histogram(('weight_matrix_' + str(i)), gradients[i][1])
            tf.summary.histogram(('gradient_for_weight_matrix_' + str(i)), gradients[i][0])
        self._writer.add_graph(graph=self._sess.graph)

        # mean_epoch_cost is defined as: mean(mean(batch_errors==cost, for i=1 up to batch_size), for i=1 up to features==d_x)
        total_batches = (int)(AutoEncoder.training_samples()/batch_size)
        for cur_epoch in range(0, total_epochs):
            mean_epoch_cost = tf.constant(0, dtype=tf.float32, shape=[self._d_x, 1])  # [d_x, 1]
            for cur_batch in range(total_batches):
                batch_x, _ = AutoEncoder.mnsit_dataset.train.next_batch(batch_size)
                batch_x = numpy.transpose(batch_x)
                c = self._sess.run(cost, feed_dict={x: batch_x})
                batch_cost = tf.constant(list(c), shape=[len(c), 1])

                if(cur_batch % 20 == 0): #Do not record all the batches. Just record one batch per 20 encountered batches.
                    merged_summaries = self._sess.run(tf.summary.merge_all(), feed_dict={x: batch_x})
                    self._writer.add_summary(merged_summaries, global_step=(cur_epoch*total_batches + cur_batch))

                mean_epoch_cost = tf.add(mean_epoch_cost, batch_cost)
                print('Current Batch: ', (cur_batch+1), ' completed.')
            mean_epoch_cost = tf.mul(tf.constant(1/total_batches, dtype=tf.float32), mean_epoch_cost)
            mean_epoch_cost = tf.reduce_mean(mean_epoch_cost, axis=0)
            mean_epoch_cost = tf.reshape(mean_epoch_cost, [])  #mean_epoch_cost was assumed to have a shape if (1,). We reshape it to a scalar.

            self._writer.add_summary(tf.summary.scalar(('mean_epoch_' + str(cur_epoch+1) + '_cost'), mean_epoch_cost).eval(session=self._sess))
            self._writer.flush()
            self.print_weights(cur_epoch+1)
        self._writer.close()
        return self._sess.run(self._weight_matrix)

    def test(self):
        """
        Uses the test dataset from the MNIST for testing
        :return: A numpy array whose rows represent the features and whose columns represent a reconstructed data point. Also
        returns a value which represents the mean error of the testing sample.
        """

        test_x = tf.placeholder(dtype=tf.float32, shape=[self._d_x, AutoEncoder.test_samples()])
        y_hat = self._dataset_output(test_x)
        error = tf.sqrt(tf.reduce_sum(tf.square(tf.sub(test_x, y_hat)), axis=0))  # vecotr of shape [1, test_samples] indicating at index i, the error of the i-sample
        mean_error = tf.reduce_mean(error)

        encoder_output = None  # tensor of shape [2, test_samples]
        for i in range(0, AutoEncoder.test_samples()):
            cur_sample = test_x[:, i]  # Fetch the sample from column i
            cur_sample = tf.reshape(cur_sample, shape=[self._d_x, 1])  # Re-shape it into a matrix [d_x, 1] from a vector
            encoded_sample = self._encode(cur_sample)  # [2, 1]. Enconded image
            encoder_output = (encoded_sample) if (encoder_output is None) else (tf.concat(concat_dim=1, values=[encoder_output, encoded_sample]))

        encoder_output, mean_error = self._sess.run([encoder_output, mean_error], feed_dict={test_x: numpy.transpose(AutoEncoder.mnsit_dataset.test.images)})
        return encoder_output, mean_error

    def _fc_layer(self, weight_matrix, layer_input, op_name='fc_layer'):
        """
        Calculates the output of a fully connected layer using the tanh() activation function
        :param weight_matrix: A Tensor which holds the values by which the input is multiplied
        :param layer_input: A Tensor which is the input for the layer
        :param op_name: A name for the output of this operation
        :return: The output of this layer, which is tanh(weight_matrix * input)
        """
        with tf.name_scope(op_name, values=[weight_matrix, layer_input]):
            return tf.nn.tanh(tf.matmul(weight_matrix, layer_input))

    def _enc_output_layer(self, weight_matrix, layer_input, op_name='enc_output_layer'):
        """
        Calculates the output of the encoder, which is a fully connected layer with no activation function.
        :param weight_matrix: A Tensor which holds the values by which the input is multiplied
        :param layer_input: A Tensor which is the input for the layer
        :param op_name: A name for the output of this operation
        :return: The output of this layer, (weight_matrix * input)
        """
        with tf.name_scope(op_name, values=[weight_matrix, layer_input]):
            return tf.matmul(weight_matrix, layer_input)

    def _dec_output_layer(cls, weight_matrix, layer_input, op_name='dec_output_layer'):
        """
        Calculates the output of the decoder, which is a fully connected layer with softmax activation function.
        :param weight_matrix: A Tensor which holds the values by which the input is multiplied
        :param layer_input: A Tensor which is the input for the layer
        :param op_name: A name for the output of this operation
        :return: The output of this layer which is softmax(weight_matrix * input)
        """
        with tf.name_scope(op_name, values=[weight_matrix, layer_input]):
            return tf.nn.softmax(tf.matmul(weight_matrix, layer_input))

    @classmethod
    def training_samples(cls):
        """
        Returns the number of training samples that are being used.
        :return: The number of training samples
        """
        return 55000    #Defined by the MNIST dataset

    @classmethod
    def test_samples(cls):
        """
        Returns the number of test samples that are being used.
        :return: The number of test samples
        """
        return 10000    #Defined by the MNIST dataset

    @classmethod
    def num_of_classes(cls):
        """
        Returns the number of classes for the MNIST dataset
        :return: The number of classes
        """
        return 10    #Defined by the MNIST dataset