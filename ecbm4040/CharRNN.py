from __future__ import print_function
import tensorflow as tf
import numpy as np
import time
import os

def pick_top_n(preds, vocab_size, top_n=5):
    '''
    choose top_n most possible charactors in predictions
    this can help reduce some noise
    inputs:
    preds
    vocab_size
    top_n
    '''
    # reduce the number of dimensions to minimum necessary
    p = np.squeeze(preds)
    # set all values other that top_n choices to 0
    p[np.argsort(p)[:-top_n]] = 0
    # normalization
    p = p / np.sum(p)
    # randomly choose one of the top_n words
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c


class CharRNN():
    def __init__(self, num_classes, batch_size=64, num_steps=50, cell_type='LSTM',
                 rnn_size=128, num_layers=2, learning_rate=0.001, 
                 grad_clip=5, train_keep_prob=0.5, sampling=False):
        '''
        Initialize the input parameter to define the network
        Inputs:
            :param num_classes: (int) the vocabulary size of your input data
            :param batch_size: (int) number of sequences in one batch
            :param num_steps: (int) length of each seuqence in one batch
            :param cell_type: your rnn cell type, 'LSTM' or 'GRU'
            :param rnn_size: (int) number of units in one rnn layer
            :param num_layers: (int) number of rnn layers
            :param learning_rate: (float)
            :param grad_clip: constraint of gradient to avoid gradient explosion
            :param train_keep_prob: (float) dropout probability for rnn cell training
            :param sampling: (boolean) whether train mode or sample mode
        '''
        # if not training
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps
        
        # rebuild graph if necessary
        tf.reset_default_graph()
        
        # constants
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.cell_type = cell_type
        self.rnn_size = rnn_size
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.train_keep_prob = train_keep_prob
        
        # network objects
        self.inputs_layer()
        self.rnn_layer()
        self.outputs_layer()
        self.my_loss()
        self.my_optimizer()
        self.saver = tf.train.Saver()
    
    
    def inputs_layer(self):
        '''
        Build the input layer
        Sized to match the batch_size
        '''
        #X and Y
        self.inputs = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps), name='inputs')
        self.targets = tf.placeholder(tf.int32, shape=(self.batch_size, self.num_steps), name='targets')
        
        # add keep_prob to determine drop out
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # one_hot encodes rnn_inputs
        self.rnn_inputs = tf.one_hot(self.inputs, self.num_classes)
    
    
    def rnn_layer(self):
        '''
        Build rnn_cell layer
        Inputs:
            self.cell_type, self.rnn_size, self.keep_prob, self.num_layers,
            self.batch_size, self.rnn_inputs
        we have to define:
            self.rnn_outputs, self.final_state for later use
        '''
        
        #initialize variables
        rnn_inputs = self.rnn_inputs
        batch_size = self.batch_size
        cell_type  = self.cell_type
        rnn_size  = self.rnn_size 
        num_layers = self.num_layers
        train_keep_prob = self.train_keep_prob 
        rnn_list = []
        
        #make LSTM or GRU cells
        if cell_type == "LSTM":
            for i in range(num_layers):
                lstm = tf.nn.rnn_cell.LSTMCell(num_units = rnn_size)
                lstm = tf.nn.rnn_cell.DropoutWrapper(cell = lstm, output_keep_prob = train_keep_prob)
                rnn_list.append(lstm)
        elif cell_type == "GRU":
            for i in range(num_layers):
                gru = tf.nn.rnn_cell.GRUCell(num_units = rnn_size)
                gru = tf.nn.rnn_cell.DropoutWrapper(cell = gru, output_keep_prob = train_keep_prob)
                rnn_list.append(gru)    
        else:
            print("u wot m8")
        
        #makelist of chain of cells to stack layers of cells
        rnn = tf.nn.rnn_cell.MultiRNNCell(rnn_list)
        
        #initialize states to 0 if no initial state exists
        try:
            self.initial_state
        except:
            self.initial_state = rnn.zero_state(batch_size, dtype = tf.float32)
        
        #unwrap cells for rnn. Essentially this creates the network based on the parameters cells
        # the outputs of final_state are the parameters of each cell
        self.rnn_outputs, self.final_state = tf.nn.dynamic_rnn(
            cell = rnn, inputs = rnn_inputs, initial_state=self.initial_state, dtype=tf.float32)
    
    def outputs_layer(self):
        ''' 
        Building the output layer. We essentially are calculating the probability of which characters will appear next.
            We combine the outputs of the rnn_cells and resize it 
            By resizing we can get an x that matches rnn_size outputs
            The we input into the softmax layer for our predictions
        Inputs:
            rnn_size, rnn_outputs
        Outputs:
            prob_pred
        '''
        # concatenate the outputs of rnn_cell，example: [[1,2,3],[4,5,6]] -> [1,2,3,4,5,6]
        seq_output = tf.concat(self.rnn_outputs, axis=1) 
        # reshape
        x = tf.reshape(seq_output, [-1, self.rnn_size])
        
        # define softmax layer variables:
        with tf.variable_scope('softmax'):
            softmax_w = tf.Variable(tf.truncated_normal([self.rnn_size, self.num_classes], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(self.num_classes))
        
        # calculate logits
        self.logits = tf.matmul(x, softmax_w) + softmax_b
        
        # softmax generate probability predictions
        self.prob_pred = tf.nn.softmax(self.logits, name='predictions')
        
        
    def my_loss(self):
        '''
        We then calculate loss according to logits and targets
        We calculate the error between true target(ex. [0, 0, .., 1, 0]) and softmax layer
        '''
        # One-hot coding
        y_one_hot = tf.one_hot(self.targets, self.num_classes)
        y_reshaped = tf.reshape(y_one_hot, self.logits.get_shape())
        
        # Softmax cross entropy loss
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=y_reshaped)
        self.loss = tf.reduce_mean(loss)
        
        
    def my_optimizer(self):
        '''
        Build our optimizer
        Unlike gradient vanishing problem, rnn cells are at risk of an "exploding gradient" effect where the value keeps growing.
        We use the gradient clipping to address this. Whenever the gradients are updated, 
        they are "clipped" to some reasonable range (like -5 to 5) so they will never get out of this range.
        Inputs:
            self.loss, self.grad_clip, self.learning_rate
        We have to define:
            self.optimizer for later use
        '''
        # intialize parameters
        learning_rate = self.learning_rate
        grad_clip = self.grad_clip
        loss = self.loss
        
        #Still using Adam
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        #replace minimize loss with an intermediate step
        gvs = optimizer.compute_gradients(loss)
        #clip values to +-grad_clip
        clipped_gvs = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var) for grad, var in gvs]
        #new optimzed values
        self.optimizer = optimizer.apply_gradients(clipped_gvs)
        
    def train(self, batches, max_count, save_every_n):
        self.session = tf.Session()
        with self.session as sess:
            sess.run(tf.global_variables_initializer())
            counter = 0
            new_state = sess.run(self.initial_state)
            # Train network
            for x, y in batches:
                counter += 1
                start = time.time()
                #print(x.shape, y.shape)
                feed = {self.inputs: x,
                        self.targets: y,
                        self.keep_prob: self.train_keep_prob,
                        self.initial_state: new_state}
                batch_loss, new_state, _ = sess.run([self.loss, 
                                                     self.final_state, 
                                                     self.optimizer], 
                                                     feed_dict=feed)
                    
                end = time.time()
                if counter % 200 == 0:
                    print('step: {} '.format(counter),
                          'loss: {:.4f} '.format(batch_loss),
                          '{:.4f} sec/batch'.format((end-start)))
                    
                if (counter % save_every_n == 0):
                    self.saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, self.rnn_size))
                    
                if counter >= max_count:
                    break
            
            self.saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, self.rnn_size))
               
        
    def sample(self, checkpoint, n_samples, vocab_size, vocab_to_ind, ind_to_vocab, prime='You \n'):
        '''
        Generates new text given the prime word
        Inputs:
            :n_samples: (int) number of characters you want to generate
            :vocab_size: (int) number of vocabulary size of your input data
            :vocab_to_ind, ind_to_vocab: mapping from unique characters to indices
            :prime: (str) you new text starting word
        Outputs:
            :a string of generated characters
        '''
        # change text into character list
        samples = np.array([vocab_to_ind[c] for c in prime],  dtype=np.int32)

        #initialize session
        self.session = tf.Session()
        with self.session as sess:
            #restore previous session and initialize new_state to initial state
            self.saver.restore(sess, checkpoint)
            new_state = sess.run(self.initial_state)
            
            #convert values of prime first
            for i in samples:
                #check to make sure i is printing correctly
                feed = {self.inputs: [[i]], 
                        self.initial_state: new_state, 
                        self.keep_prob: self.train_keep_prob,}
                
                #run to get predicted values of prime and new h state
                prediction, new_state = sess.run(
                    [self.prob_pred, self.final_state], feed_dict = feed)
                
            top_picks = pick_top_n(prediction, vocab_size, top_n = 5)
            prime = prime + ind_to_vocab[top_picks]
            
            #Generate new values based off of prime
            for i in range(n_samples):
                feed = {self.inputs: [[top_picks]],
                        self.initial_state: new_state, 
                        self.keep_prob: self.train_keep_prob,}
                
                #run to produce new values
                prediction, new_state = sess.run(
                    [self.prob_pred, self.final_state], feed_dict = feed)
                
                top_picks = pick_top_n(prediction, vocab_size, top_n = 5)
                prime = prime + ind_to_vocab[top_picks]
            return prime
            
                           
    