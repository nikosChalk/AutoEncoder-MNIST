
import tensorflow as tf
import numpy
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
        self._d_x = layers[0]
        self._d_y = layers[(int)(len(layers)/2)]
        self._encoder_wmatrix = []  # contains Tesnors which represent the weight arrays between the encoder's layers. w[0] = weight matrix between input layer-0 and layer-1
        self._decoder_wmatrix = []  # contains Tesnors which represent the weight arrays between the decoder's layers. w[0] = weight matrix between input layer-0 and layer-1
        self._encoder_bmatrix = []   # contains Tesnors which represent the bias vector for the encoder's layers. b[i] = bias vector for layer (i+1)
        self._decoder_bmatrix = []   # contains Tesnors which represent the bias vector for the decoder's layers. b[i] = bias vector for layer (i+1)
        self._sess = tf.Session()
        self._writer = tf.summary.FileWriter('TensorBoard_logs/' + str(self._layers).replace(', ', '-') + '_lr=' + str(learning_rate))
        self._summary_keys = ['per_batch', 'per_epoch']

        print('Initializing making of Computational Graph...')
        # Making the Computational Graph
        # Define the initialization of the weights and the biases
        for i in range(1, (int)(len(layers)/2)+1):
            self._encoder_wmatrix.append(tf.Variable(tf.truncated_normal([layers[i], layers[i-1]], stddev=0.499), name=('encoder_wmatrix_' + str(i-1)) ))
            self._encoder_bmatrix.append(tf.Variable(tf.truncated_normal([layers[i], 1], stddev=0.499), name=('encoder_bmatrix_' + str(i-1)) ))
        for i in range((int)(len(layers)/2)+1, len(layers)):
            self._decoder_wmatrix.append(tf.Variable(tf.truncated_normal([layers[i], layers[i-1]], stddev=0.499), name=('decoder_wmatrix_' + str(i-(int)(len(layers)/2)-1)) ))
            self._decoder_bmatrix.append(tf.Variable(tf.truncated_normal([layers[i], 1], stddev=0.499), name=('decoder_bmatrix_' + str(i-(int)(len(layers)/2)-1)) ))

        # Adding summaries for the weight and biases matrices
        for i in range (0, len(self._encoder_wmatrix)):
            tf.summary.histogram(self._encoder_wmatrix[i].name, self._encoder_wmatrix[i], collections=[self._summary_keys[0]])
            tf.summary.histogram(self._encoder_bmatrix[i].name, self._encoder_bmatrix[i], collections=[self._summary_keys[0]])

            tf.summary.histogram(self._decoder_wmatrix[i].name, self._decoder_wmatrix[i], collections=[self._summary_keys[0]])
            tf.summary.histogram(self._decoder_bmatrix[i].name, self._decoder_bmatrix[i], collections=[self._summary_keys[0]])

        # Defining NN's input place holder
        self._nn_inp_holder = tf.placeholder(dtype=tf.float32, shape=[self._d_x, None], name='nn_input_data')  # [d_x, batch_size]
        tf.summary.image('input_images', tf.reshape(tf.transpose(self._nn_inp_holder[:, 0:4]), [-1, 28, 28, 1]), max_outputs=4, collections=[self._summary_keys[0]]) # Get 4 images per batch as a sample

        # Defining NN's Output
        self._encoder_op = self._encode(self._nn_inp_holder)    # [2, batch_size]
        self._y_hat = self._decode(self._encoder_op)            # [d_x, batch_zie]
        tf.summary.image('output_images', tf.reshape(tf.transpose(self._y_hat[:, 0:4]), [-1, 28, 28, 1]), max_outputs=4, collections=[self._summary_keys[0]])    # Get 4 images per batch as a sample

        # Defining NN's cost function
        with tf.name_scope('cost', values=[self._nn_inp_holder, self._y_hat]):
            self._cost = tf.reduce_sum(tf.square(tf.sub(self._nn_inp_holder, self._y_hat)), axis=1)  # [d_x, 1]
            self._cost = tf.mul(tf.constant(1/2, dtype=tf.float32), self._cost)
            self._cost = tf.reduce_mean(self._cost, axis=0) # Scalar Value
        tf.summary.scalar('batch_cost', self._cost, collections=[self._summary_keys[0]])

        # Defining NN's optimizing algorithm
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        gradients = optimizer.compute_gradients(self._cost)
        with tf.name_scope('gradients', values=[gradients]):
            self._minimize_op = optimizer.apply_gradients(gradients)
        for i in range (0, len(gradients)): # Adding metrics for the gradient
            tf.summary.histogram(('gradients_for_' + gradients[i][1].name), gradients[i][0], collections=[self._summary_keys[0]])    #Gradients for Variables

        # Defining a metric for the mean epoch cost
        self._batches_cost_holder = tf.placeholder(dtype=tf.float32, shape=[None])
        with tf.name_scope(name='mean_epoch_cost_metrics', values=[self._batches_cost_holder]):
            self._mean_epoch_cost = tf.reduce_mean(self._batches_cost_holder)
            tf.summary.scalar('mean_epoch_cost', self._mean_epoch_cost, collections=[self._summary_keys[1]])

        self._summaries_per_batch = tf.summary.merge_all(key=self._summary_keys[0])
        self._summaries_per_epoch = tf.summary.merge_all(key=self._summary_keys[1])
        self._writer.add_graph(graph=self._sess.graph)  # Adds a visualization graph for displaying the Computation Graph

        print('Initialization of making Computational Graph completed!\n')
        print('Initialization of Weight Variables...')
        self._sess.run(tf.global_variables_initializer())  #Initializes global variables and starts assessing the computation graph
        print('Initializing of Weight Variables Done!\n')

    def delete(self):
        """

        :return:
        """
        self._writer.close()
        self._sess.close()
        tf.reset_default_graph()

    def _encode(self, data):
        """
        Encodes the given sample
        :param data: A Tensor of size [d_x, z], where z can be any number
        :return: A Tensor of size [d_y, z]
        """
        if (data.get_shape().dims[0] != self._d_x):
            raise ValueError('Input Tensor has wrong shape!')

        output = data
        for i in range(0, len(self._encoder_wmatrix)-1):
            output = self._fc_layer(self._encoder_wmatrix[i], output, self._encoder_bmatrix[i], op_name=('encoder_hl_' + str(i+1) + '_output'))
        output = self._enc_output_layer(self._encoder_wmatrix[len(self._encoder_wmatrix)-1], output, self._encoder_bmatrix[len(self._encoder_bmatrix)-1])
        return output

    def _decode(self, data):
        """
        Decodes the given sample
        :param data: A Tensor of size [d_y, z], where z can be any number
        :return: A Tensor of size [d_x, z]
        """
        if (data.get_shape().dims[0] != self._d_y):
            raise ValueError('Input Tensor has wrong shape!')

        output = data
        for i in range(0, len(self._decoder_wmatrix)-1):
            output = self._fc_layer(self._decoder_wmatrix[i], output, self._decoder_bmatrix[i], op_name=('decoder_hl_' + str(i+1) + '_output'))
        output = self._dec_output_layer(self._decoder_wmatrix[len(self._decoder_wmatrix)-1], output, self._decoder_bmatrix[len(self._decoder_bmatrix)-1])
        return output

    def train(self, total_epochs, batch_size):
        """
        Trains the weights of the given Neural Network using the MNIST dataset
        :param total_epochs: The epochs that the NN should run. Must be >0
        :param batch_size: The size of each batch. Must be >0 and <= 3500 and must have 0 modulo with 55000
        :return void
        """
        print('Initializing Training of NN...')
        if( (total_epochs<=0) or (batch_size <=0) or (batch_size>3500) or (AutoEncoder.training_samples() % batch_size != 0)):
            raise ValueError('Total epochs and batch size must be >0')

        # mean_epoch_cost is defined as: mean(batch_errors==cost, for i=1 up to batch_size)
        total_batches = (int)(AutoEncoder.training_samples()/batch_size)

        for cur_epoch in range(0, total_epochs):
            batch_cost_list = []
            for cur_batch in range(total_batches):
                batch_x, _ = AutoEncoder.mnsit_dataset.train.next_batch(batch_size)
                batch_x = numpy.transpose(batch_x)
                c, _ = self._sess.run([self._cost, self._minimize_op], feed_dict={self._nn_inp_holder: batch_x})
                batch_cost_list.append(c)

                if(cur_batch % ((int)(total_batches/15)) == 0): #Do not record all the batches. Just a few of them
                    self._writer.add_summary(self._summaries_per_batch.eval(session=self._sess, feed_dict={self._nn_inp_holder: batch_x}), global_step=(cur_epoch*total_batches + cur_batch))
#                print('Current Batch: ', (cur_batch+1), ' completed.')

            self._writer.add_summary(self._summaries_per_epoch.eval(session=self._sess, feed_dict={self._batches_cost_holder: batch_cost_list}), global_step=(cur_epoch+1))
            self._writer.flush()
            print('Current Epoch: ', (cur_epoch + 1), ' completed.')

        print('Training of NN Done!\n')
        return

    def test(self):
        """
        Uses the test dataset from the MNIST for testing
        :return: A numpy array whose rows represent the features and whose columns represent a reconstructed data point.
        """

        print('Initializing Testing of NN...')
        encoder_output = self._sess.run(self._encoder_op, feed_dict={self._nn_inp_holder: numpy.transpose(AutoEncoder.mnsit_dataset.test.images)})
        print('Testing of NN Done!\n')
        return encoder_output

    def _fc_layer(self, weight_matrix, layer_input, bias_matrix, op_name='fc_layer'):
        """
        Calculates the output of a fully connected layer using the tanh() activation function
        :param weight_matrix: A Tensor which holds the values by which the input is multiplied
        :param layer_input: A Tensor which is the input for the layer
        :param bias_matrix: A Tensor which holds the biases
        :param op_name: A name for the output of this operation
        :return: The output of this layer, which is tanh(weight_matrix * input)
        """
        with tf.name_scope(op_name, values=[weight_matrix, layer_input]):
            mul = tf.matmul(weight_matrix, layer_input)
            add = tf.add(mul, bias_matrix)
            output = tf.nn.sigmoid(add) # Broadcasting is used for performing the add operation
            tf.summary.histogram(op_name, output, collections=[self._summary_keys[0]])
            return output

    def _enc_output_layer(self, weight_matrix, layer_input, bias_matrix, op_name='encoder_output_layer'):
        """
        Calculates the output of the encoder, which is a fully connected layer with no activation function.
        :param weight_matrix: A Tensor which holds the values by which the input is multiplied
        :param layer_input: A Tensor which is the input for the layer
        :param bias_matrix: A Tensor which holds the biases
        :param op_name: A name for the output of this operation
        :return: The output of this layer, (weight_matrix * input)
        """
        with tf.name_scope(op_name, values=[weight_matrix, layer_input]):
            output = tf.add(tf.matmul(weight_matrix, layer_input), bias_matrix)     # Broadcasting is used for performing the add operation
            tf.summary.histogram(op_name, output, collections=[self._summary_keys[0]])
            return output

    def _dec_output_layer(self, weight_matrix, layer_input, bias_matrix, op_name='decoder_output_layer'):
        """
        Calculates the output of the decoder, which is a fully connected layer with softmax activation function.
        :param weight_matrix: A Tensor which holds the values by which the input is multiplied
        :param layer_input: A Tensor which is the input for the layer
        :param bias_matrix: A Tensor which holds the biases
        :param op_name: A name for the output of this operation
        :return: The output of this layer which is softmax(weight_matrix * input)
        """
        with tf.name_scope(op_name, values=[weight_matrix, layer_input]):
            output = tf.nn.sigmoid(tf.add(tf.matmul(weight_matrix, layer_input), bias_matrix))  # Broadcasting is used for performing the add operation
            tf.summary.histogram(op_name, output, collections=[self._summary_keys[0]])
            return output

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