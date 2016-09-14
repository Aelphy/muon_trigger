import os
import theano
import lasagne
import theano.tensor as T
import numpy as np
from lasagne.layers import DenseLayer,\
                           Conv2DLayer,\
                           MaxPool2DLayer,\
                           InputLayer,\
                           GlobalPoolLayer,\
                           DropoutLayer,\
                           Upscale2DLayer
from lasagne.nonlinearities import elu, sigmoid, rectify

class Base():
    def __init__(self, img_shape=(200, 200), learning_rate=1e-2, c=1.0):
        self.c = c
        self.input_X = T.tensor4('inputs')
        self.targets = T.tensor4('targets')
        self.output_layer = self.build_network(img_shape)
        self.output = lasagne.layers.get_output(self.output_layer, self.input_X)
        self.train = self.compile_trainer(learning_rate)
        self.evaluate = self.compile_evaluator()
        self.predict = self.compile_forward_pass()

    def get_obj(self):             
        a = self.output.ravel()
        t = self.targets.ravel()

        return -(t * T.log(a) + self.c * (1.0 - t) * T.log(1.0 - a)).mean()
    
    def get_recall(self):   
        a = self.output.ravel()
        t = self.targets.ravel()
        
        return ((a > 0.5) * t).sum() / t.sum()
    
    def get_accuracy(self):                      
        return lasagne.objectives.binary_accuracy(self.output, self.targets).mean()

    def compile_trainer(self, learning_rate):
        obj = self.get_obj()
        params = lasagne.layers.get_all_params(self.output_layer, trainable=True)
        updates = lasagne.updates.adagrad(obj, params, learning_rate=learning_rate)
        
        return theano.function([self.input_X, self.targets], 
                               [obj,
                                self.get_recall(),
                                self.get_accuracy()],
                               updates=updates)

    # TODO: could be modified
    def build_network(self, img_shape):
        input_layer = InputLayer((None, 1) + tuple(img_shape), self.input_X)

        conv0 = Conv2DLayer(input_layer,
                            num_filters=1,
                            filter_size=3,
                            pad='same',
                            nonlinearity=elu)
        mp01 = MaxPool2DLayer(conv0, pool_size=5)
        conv1 = Conv2DLayer(mp01,
                            num_filters=3,
                            filter_size=3,
                            pad='same',
                            nonlinearity=elu)
        mp12 = MaxPool2DLayer(conv1, pool_size=4)
        out = Conv2DLayer(mp12,
                          num_filters=1,
                          filter_size=1,
                          pad='same',
                          nonlinearity=sigmoid)

        return out

    def compile_forward_pass(self):
        return theano.function([self.input_X], self.output)

    def compile_evaluator(self):        
        return theano.function([self.input_X, self.targets], [self.get_recall(), self.get_accuracy()])
    
    def save(self, path, name):
        np.savez(os.path.join(path, name, *lasagne.layers.get_all_param_values(self.output_layer)))
    
    def load(self, path, name):
        with np.load(os.path.join(path, name + '.npz')) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
            lasagne.layers.set_all_param_values(self.output_layer, param_values)
