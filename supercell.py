'''
supercell
https://github.com/hardmaru/supercell/
inspired by http://supercell.jp/

# modified by bhkim
- adding a deep lstm with skip connections as in (Graves, 2013). code from https://github.com/carpedm20/attentive-reader-tensorflow/blob/master/model/cells.py
'''

import tensorflow as tf
import numpy as np

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell

from tensorflow.python.ops import math_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
#from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

from tensorflow.python.util import nest

# Orthogonal Initializer from
# https://github.com/OlavHN/bnlstm
def orthogonal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)

def orthogonal_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return tf.constant(orthogonal(shape), dtype)
    return _initializer

def lstm_ortho_initializer(scale=1.0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        size_x = shape[0]
        size_h = shape[1]/4 # assumes lstm.
        t = np.zeros(shape)
        t[:, :size_h] = orthogonal([size_x, size_h])*scale
        t[:, size_h:size_h*2] = orthogonal([size_x, size_h])*scale #np.identity(size_h) * scale  #bhkim-170219: following https://github.com/OlavHN/bnlstm  # 
        t[:, size_h*2:size_h*3] = orthogonal([size_x, size_h])*scale
        t[:, size_h*3:] = orthogonal([size_x, size_h])*scale
        return tf.constant(t, dtype)
    return _initializer

def layer_norm_all(h, batch_size, base, num_units, scope="layer_norm", reuse=False, gamma_start=1.0, epsilon = 1e-3, use_bias=True):
    # Layer Norm (faster version, but not using defun)
    #
    # Performas layer norm on multiple base at once (ie, i, g, j, o for lstm)
    #
    # Reshapes h in to perform layer norm in parallel
    h_reshape = tf.reshape(h, [batch_size, base, num_units])
    mean = tf.reduce_mean(h_reshape, [2], keep_dims=True)
    var = tf.reduce_mean(tf.square(h_reshape - mean), [2], keep_dims=True)
    epsilon = tf.constant(epsilon)
    rstd = tf.rsqrt(var + epsilon)
    h_reshape = (h_reshape - mean) * rstd
    # reshape back to original
    h = tf.reshape(h_reshape, [batch_size, base * num_units])
    with tf.variable_scope(scope):
        if reuse == True:
            tf.get_variable_scope().reuse_variables()
        gamma = tf.get_variable('ln_gamma', [4*num_units], initializer=tf.constant_initializer(gamma_start))
        if use_bias:
            beta = tf.get_variable('ln_beta', [4*num_units], initializer=tf.constant_initializer(0.0))
    if use_bias:
        return gamma*h + beta
    return gamma * h

def layer_norm(x, num_units, scope="layer_norm", reuse=False, gamma_start=1.0, epsilon = 1e-3, use_bias=True):
    axes = [1]
    mean = tf.reduce_mean(x, axes, keep_dims=True)
    x_shifted = x-mean
    var = tf.reduce_mean(tf.square(x_shifted), axes, keep_dims=True)
    inv_std = tf.rsqrt(var + epsilon)
    with tf.variable_scope(scope):
        if reuse == True:
            tf.get_variable_scope().reuse_variables()
        gamma = tf.get_variable('ln_gamma', [num_units], initializer=tf.constant_initializer(gamma_start))
        if use_bias:
            beta = tf.get_variable('ln_beta', [num_units], initializer=tf.constant_initializer(0.0))
    output = gamma*(x_shifted)*inv_std
    if use_bias:
        output = output + beta
    return output

def super_linear(x, output_size, scope=None, reuse=False,
    init_w="ortho", weight_start=0.0, use_bias=True, bias_start=0.0, input_size=None):
    # support function doing linear operation.    uses ortho initializer defined earlier.
    shape = x.get_shape().as_list()
    with tf.variable_scope(scope or "linear"):
        if reuse == True:
            tf.get_variable_scope().reuse_variables()

        w_init = None # uniform
        if input_size == None:
            x_size = shape[1]
        else:
            x_size = input_size
        h_size = output_size
        if init_w == "zeros":
            w_init=tf.constant_initializer(0.0)
        elif init_w == "constant":
            w_init=tf.constant_initializer(weight_start)
        elif init_w == "gaussian":
            w_init=tf.random_normal_initializer(stddev=weight_start)
        elif init_w == "ortho":
            w_init=lstm_ortho_initializer(1.0)

        w = tf.get_variable("super_linear_w",
            [x_size, output_size], tf.float32, initializer=w_init)
        if use_bias:
            b = tf.get_variable("super_linear_b", [output_size], tf.float32,
                initializer=tf.constant_initializer(bias_start))
            return tf.matmul(x, w) + b
        return tf.matmul(x, w)

def hyper_norm(layer, hyper_output, embedding_size, num_units,
                             scope="hyper", use_bias=True):
    '''
    HyperNetwork norm operator
    
    provides context-dependent weights
    layer: layer to apply operation on
    hyper_output: output of the hypernetwork cell at time t
    embedding_size: embedding size of the output vector (see paper)
    num_units: number of hidden units in main rnn
    '''
    # recurrent batch norm init trick (https://arxiv.org/abs/1603.09025).
    init_gamma = 0.10 # cooijmans' da man.
    with tf.variable_scope(scope):
        zw = super_linear(hyper_output, embedding_size, init_w="constant",
            weight_start=0.00, use_bias=True, bias_start=1.0, scope="zw")
        alpha = super_linear(zw, num_units, init_w="constant",
            weight_start=init_gamma / embedding_size, use_bias=False, scope="alpha")
        result = tf.multiply(alpha, layer)
    return result

def hyper_bias(layer, hyper_output, embedding_size, num_units,
                             scope="hyper"):
    '''
    HyperNetwork norm operator
    
    provides context-dependent bias
    layer: layer to apply operation on
    hyper_output: output of the hypernetwork cell at time t
    embedding_size: embedding size of the output vector (see paper)
    num_units: number of hidden units in main rnn
    '''

    with tf.variable_scope(scope):
        zb = super_linear(hyper_output, embedding_size, init_w="gaussian",
            weight_start=0.01, use_bias=False, bias_start=0.0, scope="zb")
        beta = super_linear(zb, num_units, init_w="constant",
            weight_start=0.00, use_bias=False, scope="beta")
    return layer + beta
    
class LSTMCell(tf.contrib.rnn.RNNCell): 
    """
    Layer-Norm, with Ortho Initialization and
    Recurrent Dropout without Memory Loss.
    https://arxiv.org/abs/1607.06450 - Layer Norm
    https://arxiv.org/abs/1603.05118 - Recurrent Dropout without Memory Loss
    derived from
    https://github.com/OlavHN/bnlstm
    https://github.com/LeavesBreathe/tensorflow_with_latest_papers
    """

    def __init__(self, num_units, forget_bias=1.0, use_layer_norm=True,
        use_recurrent_dropout=True, dropout_keep_prob=0.90, state_is_tuple=True):
        """Initialize the Layer Norm LSTM cell.
        Args:
            num_units: int, The number of units in the LSTM cell.
            forget_bias: float, The bias added to forget gates (default 1.0).
            use_recurrent_dropout: float, Whether to use Recurrent Dropout (default False)
            dropout_keep_prob: float, dropout keep probability (default 0.90)
        """
        self.num_units = num_units
        self.forget_bias = forget_bias
        self.use_layer_norm = use_layer_norm
        self.use_recurrent_dropout = use_recurrent_dropout
        self.dropout_keep_prob = dropout_keep_prob
        self._state_is_tuple = state_is_tuple
    
    @property
    def input_size(self):
        return self.num_units
    
    @input_size.setter
    def input_size(self, value):
        self._input_size = value    
        
    @property
    def output_size(self):
        return self.num_units

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self.num_units, self.num_units)

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):    # "BasicLSTMCell"
            c, h = state  # assuming 'state_is_tuple=True' 

            h_size = self.num_units
            
            batch_size = x.get_shape().as_list()[0]
            x_size = x.get_shape().as_list()[1]
            
            self.input_size = x_size
            
            w_init=orthogonal_initializer()  #None # uniform
            h_init=lstm_ortho_initializer()

            W_xh = tf.get_variable('W_xh',
                [x_size, 4 * self.num_units], initializer=w_init)

            W_hh = tf.get_variable('W_hh_i',
                [self.num_units, 4*self.num_units], initializer=h_init)

            W_full = tf.concat([W_xh, W_hh], 0)

            bias = tf.get_variable('bias',
                [4 * self.num_units], initializer=tf.constant_initializer(0.0))

            concat = tf.concat([x, h], 1) # concat for speed.
            concat = tf.matmul(concat, W_full) + bias
            
            # new way of doing layer norm (faster)
            if self.use_layer_norm:
                concat = layer_norm_all(concat, batch_size, 4, self.num_units, 'ln')

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(concat, 4, 1)

            if self.use_recurrent_dropout:
                g = tf.nn.dropout(tf.tanh(j), self.dropout_keep_prob)
            else:
                g = tf.tanh(j) 

            new_c = c*tf.sigmoid(f+self.forget_bias) + tf.sigmoid(i)*g
            if self.use_layer_norm:
                new_h = tf.tanh(layer_norm(new_c, self.num_units, 'ln_c')) * tf.sigmoid(o)
            else:
                new_h = tf.tanh(new_c) * tf.sigmoid(o)
        
        return new_h, tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

class HyperLSTMCell(tf.contrib.rnn.RNNCell):
    '''
    HyperLSTM, with Ortho Initialization,
    Layer Norm and Recurrent Dropout without Memory Loss.
    
    https://arxiv.org/abs/1609.09106
    '''

    def __init__(self, num_units, forget_bias=1.0,
        use_recurrent_dropout=False, dropout_keep_prob=0.90, use_layer_norm=True,
        hyper_num_units=128, hyper_embedding_size=16,
        hyper_use_recurrent_dropout=False):
        '''Initialize the Layer Norm HyperLSTM cell.
        Args:
            num_units: int, The number of units in the LSTM cell.
            forget_bias: float, The bias added to forget gates (default 1.0).
            use_recurrent_dropout: float, Whether to use Recurrent Dropout (default False)
            dropout_keep_prob: float, dropout keep probability (default 0.90)
            use_layer_norm: boolean. (default True)
                Controls whether we use LayerNorm layers in main LSTM and HyperLSTM cell.
            hyper_num_units: int, number of units in HyperLSTM cell.
                (default is 128, recommend experimenting with 256 for larger tasks)
            hyper_embedding_size: int, size of signals emitted from HyperLSTM cell.
                (default is 4, recommend trying larger values but larger is not always better)
            hyper_use_recurrent_dropout: boolean. (default False)
                Controls whether HyperLSTM cell also uses recurrent dropout. (Not in Paper.)
                Recommend turning this on only if hyper_num_units becomes very large (>= 512)
        '''
        self.num_units = num_units
        self.forget_bias = forget_bias
        self.use_recurrent_dropout = use_recurrent_dropout
        self.dropout_keep_prob = dropout_keep_prob
        self.use_layer_norm = use_layer_norm
        self.hyper_num_units = hyper_num_units
        self.hyper_embedding_size = hyper_embedding_size
        self.hyper_use_recurrent_dropout = hyper_use_recurrent_dropout

        self.total_num_units = self.num_units + self.hyper_num_units

        self.hyper_cell=LSTMCell(hyper_num_units,
                                 use_recurrent_dropout=hyper_use_recurrent_dropout,
                                 use_layer_norm=use_layer_norm,
                                 dropout_keep_prob=dropout_keep_prob)

    @property
    def output_size(self):
        return self.num_units

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self.num_units+self.hyper_num_units,
                                                                                 self.num_units+self.hyper_num_units)

    def __call__(self, x, state, timestep = 0, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            total_c, total_h = state
            c = total_c[:, 0:self.num_units]
            h = total_h[:, 0:self.num_units]
            hyper_state = tf.contrib.rnn.LSTMStateTuple(total_c[:,self.num_units:],                                                                                                    total_h[:,self.num_units:])

            w_init=None # uniform

            h_init=lstm_ortho_initializer(1.0)
            
            x_size = x.get_shape().as_list()[1]
            embedding_size = self.hyper_embedding_size
            num_units = self.num_units
            batch_size = x.get_shape().as_list()[0]

            W_xh = tf.get_variable('W_xh',
                [x_size, 4*num_units], initializer=w_init)
            W_hh = tf.get_variable('W_hh',
                [num_units, 4*num_units], initializer=h_init)
            bias = tf.get_variable('bias',
                [4*num_units], initializer=tf.constant_initializer(0.0))

            # concatenate the input and hidden states for hyperlstm input
            hyper_input = tf.concat([x, h], 1)
            hyper_output, hyper_new_state = self.hyper_cell(hyper_input, hyper_state)

            xh = tf.matmul(x, W_xh)
            hh = tf.matmul(h, W_hh)

            # split Wxh contributions
            ix, jx, fx, ox = tf.split(xh, 4, 1)
            ix = hyper_norm(ix, hyper_output, embedding_size, num_units, 'hyper_ix')
            jx = hyper_norm(jx, hyper_output, embedding_size, num_units, 'hyper_jx')
            fx = hyper_norm(fx, hyper_output, embedding_size, num_units, 'hyper_fx')
            ox = hyper_norm(ox, hyper_output, embedding_size, num_units, 'hyper_ox')

            # split Whh contributions
            ih, jh, fh, oh = tf.split(hh, 4, 1)
            ih = hyper_norm(ih, hyper_output, embedding_size, num_units, 'hyper_ih')
            jh = hyper_norm(jh, hyper_output, embedding_size, num_units, 'hyper_jh')
            fh = hyper_norm(fh, hyper_output, embedding_size, num_units, 'hyper_fh')
            oh = hyper_norm(oh, hyper_output, embedding_size, num_units, 'hyper_oh')

            # split bias
            ib, jb, fb, ob = tf.split(bias, 4, 0) # bias is to be broadcasted.
            ib = hyper_bias(ib, hyper_output, embedding_size, num_units, 'hyper_ib')
            jb = hyper_bias(jb, hyper_output, embedding_size, num_units, 'hyper_jb')
            fb = hyper_bias(fb, hyper_output, embedding_size, num_units, 'hyper_fb')
            ob = hyper_bias(ob, hyper_output, embedding_size, num_units, 'hyper_ob')

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i = ix + ih + ib
            j = jx + jh + jb
            f = fx + fh + fb
            o = ox + oh + ob

            if self.use_layer_norm:
                concat = tf.concat([i, j, f, o], 1)
                concat = layer_norm_all(concat, batch_size, 4, num_units, 'ln_all')
                i, j, f, o = tf.split(concat, 4, 1)

            if self.use_recurrent_dropout:
                g = tf.nn.dropout(tf.tanh(j), self.dropout_keep_prob)
            else:
                g = tf.tanh(j) 

            new_c = c*tf.sigmoid(f+self.forget_bias) + tf.sigmoid(i)*g
            if self.use_layer_norm:
                new_h = tf.tanh(layer_norm(new_c, num_units, 'ln_c')) * tf.sigmoid(o)
            else:
                new_h = tf.tanh(new_c) * tf.sigmoid(o)
        
            hyper_c, hyper_h = hyper_new_state
            new_total_c = tf.concat([new_c, hyper_c], 1)
            new_total_h = tf.concat([new_h, hyper_h], 1)

        return new_h, tf.contrib.rnn.LSTMStateTuple(new_total_c, new_total_h)

######################################################
# new code by bhkim
class GRUCell(tf.contrib.rnn.RNNCell): 
    """
    Layer-Norm, with Ortho Initialization and
    https://arxiv.org/abs/1607.06450 - Layer Norm
    derived from
    https://github.com/OlavHN/bnlstm
    """

    def __init__(self, num_units, use_layer_norm=True):
        """Initialize the Layer Norm GRU cell.
        Args:
            num_units: int, The number of units in the GRU cell.
        """
        self.num_units = num_units
        self.use_layer_norm = use_layer_norm
    
    @property
    def input_size(self):
        return self.num_units
    
    @input_size.setter
    def input_size(self, value):
        self._input_size = value    
        
    @property
    def output_size(self):
        return self.num_units

    @property
    def state_size(self):
        return self.num_units
        
    def __call__(self, x, state, scope=None):
    # reference code from Keras: https://github.com/fchollet/keras/blob/master/keras/layers/recurrent.py#L404 - class GRU(Recurrent): step()
        with tf.variable_scope(scope or type(self).__name__):
            h_tm1 = state
            h_size = self.num_units
            
            batch_size = x.get_shape().as_list()[0]
            x_size = x.get_shape().as_list()[1]
            
            self.input_size = x_size
            
            ## build() ##  consume_less = mem' mode. Beware: notion of U and W is reversed
            u_init=orthogonal_initializer()   # None # uniform
            w_init=orthogonal_initializer()

            U_xh_z = tf.get_variable('U_xh_z',
                [x_size, self.num_units], initializer=u_init)
            W_hh_z = tf.get_variable('W_hh_z',
                [self.num_units, self.num_units], initializer=w_init)
            bias_z = tf.get_variable('bias_z',
                self.num_units, initializer=tf.constant_initializer(0.0))
            U_xh_r = tf.get_variable('U_xh_r',
                [x_size, self.num_units], initializer=u_init)
            W_hh_r = tf.get_variable('W_hh_r',
                [self.num_units, self.num_units], initializer=w_init)
            bias_r = tf.get_variable('bias_r',
                self.num_units, initializer=tf.constant_initializer(0.0))            
            U_xh_h = tf.get_variable('U_xh_h',
                [x_size, self.num_units], initializer=u_init)
            W_hh_h = tf.get_variable('W_hh_h',
                [self.num_units, self.num_units], initializer=w_init)
            bias_h = tf.get_variable('bias_h',
                self.num_units, initializer=tf.constant_initializer(0.0))
            
            self.U = tf.concat([U_xh_z, U_xh_r, U_xh_h],1)
            self.W = tf.concat([W_hh_z, W_hh_r, W_hh_h],1)
            self.b = [bias_z, bias_r, bias_h]
            ## end of build ##

            ## step() ##  consume_less = mem' mode. Beware: notion of U and W is reversed
            x_z = tf.matmul(x, U_xh_z) + bias_z
            x_r = tf.matmul(x, U_xh_r) + bias_r
            x_h = tf.matmul(x, U_xh_h) + bias_h
            
            z = x_z + tf.matmul(h_tm1, W_hh_z) 
            r = x_r + tf.matmul(h_tm1, W_hh_r) 
            if self.use_layer_norm:
                z = layer_norm(z, self.num_units, 'ln_z')
                r = layer_norm(r, self.num_units, 'ln_r')
            r, z = tf.sigmoid(r), tf.sigmoid(z)
            
            hh = x_h + tf.matmul(r * h_tm1, W_hh_h) 
            if self.use_layer_norm:
                hh = layer_norm(hh, self.num_units, 'ln_hh')
            hh = tf.tanh(hh)
                                        
            new_h = z * h_tm1 + (1 - z) * hh
            
        return new_h, new_h
    
#####################################################
# code from https://github.com/carpedm20/attentive-reader-tensorflow/blob/master/model/cells.py (2017-02-17)
#   -- bug correction: cur_inp, new_state = cell(tf.concat([inputs, first_layer_input], 1), cur_state) ==> cur_inp, new_state = cell(tf.concat([cur_inp, first_layer_input], 1), cur_state) (2017-03-05)
class MultiRNNCellWithAdditionalConn(core_rnn_cell.RNNCell):    # from tensorflow.models.rnn.rnn_cell import RNNCell
    """Almost same with tf.models.rnn.rnn_cell.MultiRnnCell except adding
    a skip connection from the input of current time t to every hidden layers and using 
    _num_units not state size because LSTMCell returns only [h] not [c, h].
    NOTE: original code of this function is from https://github.com/carpedm20/attentive-reader-tensorflow/blob/master/model/cells.py
    """

    def __init__(self, cells, state_is_tuple=True, add_skip_conn=False, add_resid_conn=False):
        """Create a RNN cell composed sequentially of a number of RNNCells.
        Args:
          cells: list of RNNCells that will be composed in this order.
        Raises:
          ValueError: if cells is empty (not allowed) or if their sizes don't match.
        """
        if not cells:
            raise ValueError("Must specify at least one cell for MultiRNNCell.")
        if not nest.is_sequence(cells):
            raise TypeError(
                    "cells must be a list or tuple, but saw: %s." % cells)

        self._cells = cells
        self._state_is_tuple = state_is_tuple
        self._add_skip_conn = add_skip_conn
        self._add_resid_conn = add_resid_conn
        if not state_is_tuple:
            if any(nest.is_sequence(c.state_size) for c in self._cells):
                raise ValueError("Some cells return tuples of states, but the flag "
                                                 "state_is_tuple is not set.    State sizes are: %s"
                                                 % str([c.state_size for c in self._cells]))

    @property
    def input_size(self):
        return self._cells[0].input_size
    
    @property
    def state_is_tuple(self):
        return self._state_is_tuple

    @property
    def output_size(self):
        if self._add_skip_conn:
            out_size = self._cells[-1].output_size * len(cells) 
        else:    
            out_size = self._cells[-1].output_size
        return out_size

    @property
    def state_size(self):  # modified according to MultiRNNCell in https://github.com/tensorflow/tensorflow/blob/66d5d1fa0c192ca4c9b75cde216866805eb160f2/tensorflow/contrib/rnn/python/ops/core_rnn_cell_impl.py
        if self._state_is_tuple:
            return tuple(cell.state_size for cell in self._cells)
        else:
            return sum([cell.state_size for cell in self._cells])
        
    def __call__(self, inputs, state, scope=None):
        """Run this multi-layer cell on inputs, starting from state."""
        #with tf.variable_scope("MultiRNNCellWithConn"):
        with vs.variable_scope(scope or "multi_rnn_cell_with_skipconn"):    
            cur_state_pos = 0
            first_layer_input = cur_inp = inputs
            new_states = []                        

            for i, cell in enumerate(self._cells):
                with vs.variable_scope("cell_%d" % i):
                    if self._state_is_tuple:
                        if not nest.is_sequence(state):
                            raise ValueError(
                                    "Expected state to be a tuple of length %d, but received: %s"
                                    % (len(self.state_size), state))
                        cur_state = state[i]
                    else:
                        cur_state = array_ops.slice(
                                state, [0, cur_state_pos], [-1, cell.state_size])
                        cur_state_pos += cell.state_size
                                            
                    if self._add_skip_conn:  # input-to-state skip connection is added for each hidden layer
                        if i != 0:
                            first_layer_input = first_layer_input
                        else:
                            first_layer_input = tf.zeros_like(first_layer_input)

                        cell_out, new_state = cell(tf.concat([cur_inp, first_layer_input], 1), cur_state)
                    else:
                        cell_out, new_state = cell(cur_inp, cur_state)
                                         
                    if self._add_resid_conn:    
                        if i != 0:
                            cur_inp = cur_inp + cell_out   # F(x) = x + residual 
                        else:    
                            cur_inp = cell_out
                    else:    
                        cur_inp = cell_out
                        
                    new_states.append(new_state)
        new_states = (tuple(new_states) if self._state_is_tuple else
                                    array_ops.concat(new_states, 1))
        
        return cur_inp, new_states # tf.concat(1, new_states)
    
    
#################################################################
# https://github.com/tensorflow/tensorflow/blob/a0d784bdd31b27e013a7eac58a86ba62e86db299/tensorflow/contrib/rnn/python/ops/rnn_cell.py (accesed 2017-02-15)
class LayerNormBasicLSTMCell(core_rnn_cell.RNNCell):
    """LSTM unit with layer normalization and recurrent dropout.
    This class adds layer normalization and recurrent dropout to a
    basic LSTM unit. Layer normalization implementation is based on:
        https://arxiv.org/abs/1607.06450.
    "Layer Normalization"
    Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
    and is applied before the internal nonlinearities.
    Recurrent dropout is base on:
        https://arxiv.org/abs/1603.05118
    "Recurrent Dropout without Memory Loss"
    Stanislau Semeniuta, Aliaksei Severyn, Erhardt Barth.
    """

    def __init__(self, num_units, forget_bias=1.0,
                             input_size=None, activation=math_ops.tanh,
                             layer_norm=True, norm_gain=1.0, norm_shift=0.0,
                             dropout_keep_prob=1.0, dropout_prob_seed=None):
        """Initializes the basic LSTM cell.
        Args:
            num_units: int, The number of units in the LSTM cell.
            forget_bias: float, The bias added to forget gates (see above).
            input_size: Deprecated and unused.
            activation: Activation function of the inner states.
            layer_norm: If `True`, layer normalization will be applied.
            norm_gain: float, The layer normalization gain initial value. If
                `layer_norm` has been set to `False`, this argument will be ignored.
            norm_shift: float, The layer normalization shift initial value. If
                `layer_norm` has been set to `False`, this argument will be ignored.
            dropout_keep_prob: unit Tensor or float between 0 and 1 representing the
                recurrent dropout probability value. If float and 1.0, no dropout will
                be applied.
            dropout_prob_seed: (optional) integer, the randomness seed.
        """

        if input_size is not None:
            logging.warn("%s: The input_size parameter is deprecated.", self)

        self._num_units = num_units
        self._activation = activation
        self._forget_bias = forget_bias
        self._keep_prob = dropout_keep_prob
        self._seed = dropout_prob_seed
        self._layer_norm = layer_norm
        self._g = norm_gain
        self._b = norm_shift

    @property
    def state_size(self):
        return core_rnn_cell.LSTMStateTuple(self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def _norm(self, inp, scope):
        shape = inp.get_shape()[-1:]
        gamma_init = init_ops.constant_initializer(self._g)
        beta_init = init_ops.constant_initializer(self._b)
        with vs.variable_scope(scope):
            # Initialize beta and gamma for use by layer_norm.
            vs.get_variable("gamma", shape=shape, initializer=gamma_init)
            vs.get_variable("beta", shape=shape, initializer=beta_init)
        normalized = layers.layer_norm(inp, reuse=True, scope=scope)
        return normalized

    def _linear(self, args):
        out_size = 4 * self._num_units
        proj_size = args.get_shape()[-1]
        weights = vs.get_variable("weights", [proj_size, out_size])
        out = math_ops.matmul(args, weights)
        if not self._layer_norm:
            bias = vs.get_variable("biases", [out_size])
            out = nn_ops.bias_add(out, bias)
        return out

    def __call__(self, inputs, state, scope=None):
        """LSTM cell with layer normalization and recurrent dropout."""

        with vs.variable_scope(scope or "layer_norm_basic_lstm_cell"):
            c, h = state
            args = array_ops.concat([inputs, h], 1)
            concat = self._linear(args)

            i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)
            if self._layer_norm:
                i = self._norm(i, "input")
                j = self._norm(j, "transform")
                f = self._norm(f, "forget")
                o = self._norm(o, "output")

            g = self._activation(j)
            if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
                g = nn_ops.dropout(g, self._keep_prob, seed=self._seed)

            new_c = (c * math_ops.sigmoid(f + self._forget_bias)
                             + math_ops.sigmoid(i) * g)
            if self._layer_norm:
                new_c = self._norm(new_c, "state")
            new_h = self._activation(new_c) * math_ops.sigmoid(o)

            new_state = core_rnn_cell.LSTMStateTuple(new_c, new_h)
            return new_h, new_state

        
####################################################################        
# codes from http://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html

# replacement of the zero_state method in base RNNCell class
def get_initial_cell_state(cell, initializer, batch_size, dtype):
    """Return state tensor(s), initialized with initializer.
    Args:
      cell: RNNCell.
      batch_size: int, float, or unit Tensor representing the batch size.
      initializer: must be a function with four arguments: shape and dtype, a la tf.zeros, and additionally batch_size and index, which are introduced to play nice with variables.
      dtype: the data type to use for the state.
    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size x state_size]` initialized
      according to the initializer.
      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
    the shapes `[batch_size x s]` for each s in `state_size`.
    """
    state_size = cell.state_size
    if nest.is_sequence(state_size):
        state_size_flat = nest.flatten(state_size)
        init_state_flat = [
            initializer(_state_size_with_prefix(s), batch_size, dtype, i)
                for i, s in enumerate(state_size_flat)]
        init_state = nest.pack_sequence_as(structure=state_size,
                                    flat_sequence=init_state_flat)
    else:
        init_state_size = _state_size_with_prefix(state_size)
        init_state = initializer(init_state_size, batch_size, dtype, None)

    return init_state       

# calling get_initial_cell_state(cell, zero_state_initializer, batch_size, tf.float32) does the same thing as calling zero_state(cell, batch_size, tf.float32).
def zero_state_initializer(shape, batch_size, dtype, index):
    z = tf.zeros(tf.pack(_state_size_with_prefix(shape, [batch_size])), dtype)
    z.set_shape(_state_size_with_prefix(shape, prefix=[None]))
    return z

def make_variable_state_initializer(**kwargs):
    def variable_state_initializer(shape, batch_size, dtype, index):
        args = kwargs.copy()

        if args.get('name'):
            args['name'] = args['name'] + '_' + str(index)
        else:
            args['name'] = 'init_state_' + str(index)

        args['shape'] = shape
        args['dtype'] = dtype

        var = tf.get_variable(**args)
        var = tf.expand_dims(var, 0)
        var = tf.tile(var, tf.pack([batch_size] + [1] * len(shape)))
        var.set_shape(_state_size_with_prefix(shape, prefix=[None]))
        return var

    return variable_state_initializer