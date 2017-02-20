# Important NOTE: original code of sketch-rnn is based on TensorFlow version less than 0.10
# updated to Tensorflow version 1.0 based on the change list: https://tensorflow.blog/2016/12/22/tensorflow-api-changes/
# modified by bhkim
# last update: 2017-02-19

import tensorflow as tf
import numpy as np
import random

#from layers import *
import supercell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell, core_rnn_cell_impl

import inspect

# RNN + MDN model. MDN = Mixture Density Networks
class Model():
    def __init__(self, args, infer=False):
        if infer:
            args.batch_size = 1
            args.seq_length = 1
        self.args = args

        self.unitcell_state_is_tuple=False
        
        if args.model == 'gru':   # currently, this gru does not work well owing to bugs maybe
            cell_fn = supercell.GRUCell #tf.contrib.rnn.GRUCell
        elif args.model == 'lstm':
            cell_fn = supercell.LSTMCell #tf.nn.rnn_cell.BasicLSTMCell #(state_is_tuple=True)
            self.unitcell_state_is_tuple = True
        elif args.model == 'hyperlstm':
            cell_fn = supercell.HyperLSTMCell #HyperLnLSTMCell # HyperLSTMCell 
            self.unitcell_state_is_tuple = True
        else:
            raise Exception("model type not supported: {}".format(args.model))

        self.state_is_tuple=True  # should not be False
        cell = cell_fn(args.rnn_size)

        # we use stacked RNN with skip connections (currently, input-to-hidden only)
        if args.skipconn:
            cell = supercell.MultiRNNCellWithSkipConn(
                [cell] * args.num_layers, 
                state_is_tuple=self.state_is_tuple)  # state_is_tuple=True for LSTMs
        else:    
            cell = core_rnn_cell_impl.MultiRNNCell(
                [cell] * args.num_layers, 
                state_is_tuple=self.state_is_tuple)

        if (infer == False and args.keep_prob < 1): # training mode
            cell = core_rnn_cell.DropoutWrapper(cell, output_keep_prob = args.keep_prob)

        self.cell = cell

        self.input_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, 5])
        self.target_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, 5])
        
        ###
        self.initial_state = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)
        
        #print('## initial state: {}\n'.format(self.initial_state))
        
        self.num_mixture = args.num_mixture
        NOUT = 3 + self.num_mixture * 6 # [end_of_stroke + end_of_char, continue_with_stroke] + prob + 2*(mu + sig) + corr

        with tf.variable_scope('rnn_mdn'):
            if args.skipconn:
                #output_w = [ tf.get_variable("output_w{}".format(i), [args.rnn_size, NOUT]) for i in xrange(args.num_layers) ]
                output_w = tf.get_variable("output_w", [args.rnn_size * args.num_layers, NOUT])
            else:    
                output_w = tf.get_variable("output_w", [args.rnn_size, NOUT])
            output_b = tf.get_variable("output_b", [NOUT])
        
        inputs = tf.split(self.input_data, args.seq_length, 1)  
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        self.initial_input = np.zeros((args.batch_size, 5), dtype=np.float32)
        self.initial_input[:,4] = 1.0 # initially, the pen is down.
        self.initial_input = tf.constant(self.initial_input)

        def tfrepeat(a, repeats):
            num_row = a.get_shape()[0].value
            num_col = a.get_shape()[1].value
            assert(num_col == 1)
            result = [a for i in range(repeats)]
            result = tf.concat(result, 0)
            result = tf.reshape(result, [repeats, num_row])
            result = tf.transpose(result)
            return result
        
        def custom_rnn_autodecoder(decoder_inputs, initial_input, initial_state, cell, scope=None):
            # customized rnn_decoder for the task of dealing with the end of character
            with tf.variable_scope(scope or "rnn_decoder"):
                states = [initial_state]
                outputs = []
                prev = None

                for i in xrange(len(decoder_inputs)):  # for each time step in mini-batch
                    inp = decoder_inputs[i]
                    if i > 0:
                        tf.get_variable_scope().reuse_variables()
                    #output, new_state = cell(inp, states[-1])  # this line is for single RNN cell
                    _, new_states = cell(inp, states[-1])  # this line is for MultiRNNCell. The first return value is inp                    
                    #print('## new_states: {}, \n new_states[0]: {}\n'.format(new_states, new_states[0]))
                    
                    if self.state_is_tuple:                        
                        if self.unitcell_state_is_tuple:
                            num_state = new_states[0][0].get_shape()[1].value
                            if args.skipconn:
                                output = new_states[0][1]
                                for i in xrange(1, self.args.num_layers) :
                                    output = tf.concat([output, new_states[i][1]], 1)
                            else:    
                                output = new_states[-1][1]
                        else:
                            num_state = new_states[0].get_shape()[1].value
                            if args.skipconn:
                                output = new_states[0]
                                for i in xrange(1, self.args.num_layers) :
                                    output = tf.concat([output, new_states[i]], 1)
                            else:    
                                output = new_states[-1]  # get the top hidden states as the output
                    else: # should not be reached
                        num_state = int(new_states.get_shape()[1].value / self.args.rnn_size)
                        if self.unitcell_state_is_tuple:
                            output = new_states[-self.args.rnn_size:]  # ??
                        else:    
                            output = new_states[-self.args.rnn_size:] 
                    #print('## output: {}\n'.format(output))
                    #print('## n_states: {}'.format(num_state))
                    
                    # if the input has an end-of-character signal, have to zero out the state
                    #to do by hardmaru: test this code.
                    num_batches = self.args.batch_size
                    eoc_detection = inp[:,3]
                    #eoc_detection = tf.reshape(eoc_detection, [num_batches, 1])
                    #eoc_detection_state = tfrepeat(eoc_detection, num_state)
                    #eoc_detection_state = tf.greater(eoc_detection_state, tf.zeros_like(eoc_detection_state, dtype=tf.float32)) # make it a binary tensor

                    # if the eoc detected, new state should be reset to zeros (initial state)
                    #new_state = tf.select(eoc_detection_state, initial_state, new_state)   # tf.select(condition, t, e, name=None). Selects elements from t or e , depending on condition
                    #new_states = tf.where(eoc_detection_state, initial_state, new_states)
                    
                    for i in xrange(num_batches):
                        if eoc_detection[i] == 1:
                            for j in self.args.num_layers:
                                if args.model == 'gru':
                                    new_states[j][i] = initial_state[j][i]
                                elif args.model == 'lstm':
                                    new_states[j][0][i] = initial_state[j][0][i]
                                    new_states[j][1][i] = initial_state[j][1][i]

                    outputs.append(output)
                    states.append(new_states)
                                        
            return outputs, states

        outputs, states = custom_rnn_autodecoder(inputs, self.initial_input, self.initial_state, cell, scope='rnn_mdn')
        
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size * args.num_layers])        
        
        #output = tf.nn.xw_plus_b(output, output_w[-1], output_b)
        output = tf.matmul(output, output_w) + output_b

        self.final_state = states[-1]

        # reshape target data so that it is compatible with prediction shape
        flat_target_data = tf.reshape(self.target_data,[-1, 5])
        [x1_data, x2_data, eos_data, eoc_data, cont_data] = tf.split(flat_target_data, 5, 1)
        pen_data = tf.concat([eos_data, eoc_data, cont_data], 1)

        # long method:
        #flat_target_data = tf.split(1, args.seq_length, self.target_data)
        #flat_target_data = [tf.squeeze(flat_target_data_, [1]) for flat_target_data_ in flat_target_data]
        #flat_target_data = tf.reshape(tf.concat(1, flat_target_data), [-1, 3])

        def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
            # eq # 24 and 25 of http://arxiv.org/abs/1308.0850
            norm1 = tf.subtract(x1, mu1)
            norm2 = tf.subtract(x2, mu2)
            s1s2 = tf.multiply(s1, s2)
            z = tf.square(tf.divide(norm1, s1))+tf.square(tf.divide(norm2, s2))-2*tf.divide(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2)
            negRho = 1-tf.square(rho)
            result = tf.exp(tf.divide(-z,2*negRho))
            denom = 2*np.pi*tf.multiply(s1s2, tf.sqrt(negRho))
            result = tf.divide(result, denom)
            return result

        def get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen, x1_data, x2_data, pen_data):
            result0 = tf_2d_normal(x1_data, x2_data, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr)
            # implementing eq # 26 of http://arxiv.org/abs/1308.0850
            epsilon = 1e-20
            result1 = tf.multiply(result0, z_pi)
            result1 = tf.reduce_sum(result1, 1, keep_dims=True)
            result1 = -tf.log(tf.maximum(result1, 1e-20)) # at the beginning, some errors are exactly zero.
            result_shape = tf.reduce_mean(result1)

            result2 = tf.nn.softmax_cross_entropy_with_logits(labels=pen_data, logits = z_pen)
            #pen_data_weighting = pen_data[:, 2]+np.sqrt(self.args.stroke_importance_factor)*pen_data[:, 0]+self.args.stroke_importance_factor*pen_data[:, 1]
            pen_data_weighting = pen_data[:, 2] + \
                        np.sqrt(self.args.stroke_importance_factor)*pen_data[:, 0] + \
                        self.args.stroke_importance_factor*pen_data[:, 1]
            result2 = tf.multiply(result2, pen_data_weighting)
            result_pen = tf.reduce_mean(result2)

            result = result_shape + result_pen
            return result, result_shape, result_pen,

        # below is where we need to do MDN splitting of distribution params
        def get_mixture_coef(output):
            # returns the tf slices containing mdn dist params
            # ie, eq 18 -> 23 of http://arxiv.org/abs/1308.0850
            z = output
            z_pen = z[:, 0:3] # end of stroke, end of character/content, continue w/ stroke
            z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr = tf.split(z[:, 3:], 6, 1)

            # process output z's into MDN paramters

            # softmax all the pi's:
            max_pi = tf.reduce_max(z_pi, 1, keep_dims=True)
            z_pi = tf.subtract(z_pi, max_pi)
            z_pi = tf.exp(z_pi)
            normalize_pi = tf.reciprocal(tf.reduce_sum(z_pi, 1, keep_dims=True))  # inv (api 0.10) --> reciprocal (api 0.12, name changed)
            z_pi = tf.multiply(normalize_pi, z_pi)

            # exponentiate the sigmas and also make corr between -1 and 1.
            z_sigma1 = tf.exp(z_sigma1)
            z_sigma2 = tf.exp(z_sigma2)
            z_corr = tf.tanh(z_corr)

            return [z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_pen]

        [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen] = get_mixture_coef(output)

        self.pi = o_pi
        self.mu1 = o_mu1
        self.mu2 = o_mu2
        self.sigma1 = o_sigma1
        self.sigma2 = o_sigma2
        self.corr = o_corr
        self.pen = o_pen # state of the pen

        [lossfunc, loss_shape, loss_pen] = get_lossfunc(o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, x1_data, x2_data, pen_data)
        self.cost = lossfunc
        self.cost_shape = loss_shape
        self.cost_pen = loss_pen

        self.lr = tf.Variable(0.0001, trainable=False) # tf.Variable(0.01, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr, epsilon=0.001)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))


    def sample(self, sess, num=30, temp_mixture=1.0, temp_pen=1.0, stop_if_eoc = False):

        def get_pi_idx(x, pdf):
            N = pdf.size
            accumulate = 0
            for i in range(0, N):
                accumulate += pdf[i]
                if (accumulate >= x):
                    return i
            print 'error with sampling ensemble'
            return -1

        def sample_gaussian_2d(mu1, mu2, s1, s2, rho):
            mean = [mu1, mu2]
            cov = [[s1*s1, rho*s1*s2], [rho*s1*s2, s2*s2]]
            x = np.random.multivariate_normal(mean, cov, 1)
            return x[0][0], x[0][1]

        prev_x = np.zeros((1, 1, 5), dtype=np.float32)
        #prev_x[0, 0, 2] = 1 # initially, we want to see beginning of new stroke
        #prev_x[0, 0, 3] = 1 # initially, we want to see beginning of new character/content
        prev_state = sess.run(self.cell.zero_state(self.args.batch_size, tf.float32))

        strokes = np.zeros((num, 5), dtype=np.float32)
        mixture_params = []

        for i in xrange(num):

            feed = {self.input_data: prev_x, self.initial_state:prev_state}

            [o_pi, o_mu1, o_mu2, o_sigma1, o_sigma2, o_corr, o_pen, next_state] = sess.run([self.pi, self.mu1, self.mu2, self.sigma1, self.sigma2, self.corr, self.pen, self.final_state],feed)

            pi_pdf = o_pi[0]
            if i > 1:
                pi_pdf = np.log(pi_pdf) / temp_mixture
                pi_pdf -= pi_pdf.max()
                pi_pdf = np.exp(pi_pdf)
            pi_pdf /= pi_pdf.sum()

            idx = get_pi_idx(random.random(), pi_pdf)

            pen_pdf = o_pen[0]
            if i > 1:
                pi_pdf /= temp_pen # softmax convert to prob
            pen_pdf -= pen_pdf.max()
            pen_pdf = np.exp(pen_pdf)
            pen_pdf /= pen_pdf.sum()

            pen_idx = get_pi_idx(random.random(), pen_pdf)
            eos = 0
            eoc = 0
            cont_state = 0

            if pen_idx == 0:
                eos = 1
            elif pen_idx == 1:
                eoc = 1
            else:
                cont_state = 1

            next_x1, next_x2 = sample_gaussian_2d(o_mu1[0][idx], o_mu2[0][idx], o_sigma1[0][idx], o_sigma2[0][idx], o_corr[0][idx])

            strokes[i,:] = [next_x1, next_x2, eos, eoc, cont_state]

            params = [pi_pdf, o_mu1[0], o_mu2[0], o_sigma1[0], o_sigma2[0], o_corr[0], pen_pdf]
            mixture_params.append(params)

            # early stopping condition
            if (stop_if_eoc and eoc == 1):
                strokes = strokes[0:i+1, :]
                break

            prev_x = np.zeros((1, 1, 5), dtype=np.float32)
            prev_x[0][0] = np.array([next_x1, next_x2, eos, eoc, cont_state], dtype=np.float32)
            prev_state = next_state

        strokes[:,0:2] *= self.args.data_scale
        return strokes, mixture_params


