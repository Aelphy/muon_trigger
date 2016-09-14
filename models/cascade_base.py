import lasagne
import theano
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


class CascadeBase(object):
    def __init__(self,
                 img_shape=(640, 480),
                 learning_rate=1e-3,
                 c=1.0,
                 c_complexity=1e-3,
                 c_sub_objs=[1e-3]):
        self.c_sub_objs = c_sub_objs
        self.c_complexity = c_complexity
        self.c = c
        
        self.input_X = T.tensor4('inputs')
        self.targets = T.tensor4('targets')
        
        self.output_layers = self.build_network(img_shape)
        
        assert(len(self.output_layers) - 1 == len(self.c_sub_objs))
        
        self.output = self.build_output()
        self.train = self.compile_trainer(learning_rate)
        self.evaluate = self.compile_evaluator()
        self.predict = self.compile_forward_pass()
        
    def build_output(self):
        answers = 1

        for output_layer in self.output_layers:
            answers *= lasagne.layers.get_output(output_layer, self.input_X)
            
        return answers
    
    def compute_loss(self, a, t, c):
        return -(t * T.log(a) + c * (1.0 - t) * T.log(1.0 - a)).mean()
        
    # TODO: here we can change the value of self.c with individual value for every cascade
    def get_sub_loss(self):
        sub_obj = 0
        
        for i, output_layer in enumerate(self.output_layers[1:]):
            sub_answer = lasagne.layers.get_output(output_layer, self.input_X)
            sub_obj += self.compute_loss(sub_answer.ravel(),
                                         self.targets.ravel(),
                                         self.c) * self.c_sub_objs[i]
            
        return sub_obj
    
    def get_complexity(self):
        complexity = 0
        cumprod = 1
        
        for output_layer in self.output_layers[1:]:
            cumprod *= lasagne.layers.get_output(output_layer, self.input_X)
            complexity += (cumprod * (1 - self.targets)).sum()
            
        return complexity
    
    def get_loss(self):
        return self.compute_loss(self.output.ravel(), self.targets.ravel(), self.c)
    
    def get_obj(self):                    
        return self.get_loss() + self.get_sub_loss() + self.get_complexity()
    
    def get_precision(self):   
        a = self.output.ravel()
        t = self.targets.ravel()
        
        return ((a > 0.5) * t).sum() / (a > 0.5).sum()
    
    def get_recall(self):   
        a = self.output.ravel()
        t = self.targets.ravel()
        
        return ((a > 0.5) * t).sum() / t.sum()
    
    def get_accuracy(self):      
        a = self.output.ravel()
        t = self.targets.ravel()
        
        return lasagne.objectives.binary_accuracy(a, t).mean()

    # TODO: could be modified
    def build_network(self, img_shape, pool_sizes=[5, 4], depths=[1, 3]):
        input_layer = InputLayer((None, 1) + tuple(img_shape),
                                 self.input_X,
                                 name='network input')

        conv0 = Conv2DLayer(input_layer,
                         nonlinearity=elu,
                         num_filters=depths[0],
                         filter_size=3,
                         pad='same',
                         name='l1 conv')
        mp01 = MaxPool2DLayer(conv0, pool_size=pool_sizes[0], name='l01 max_pool')
        conv1 = Conv2DLayer(mp01,
                            nonlinearity=elu,
                            num_filters=depths[1],
                            filter_size=3,
                            pad='same',
                            name='l1 conv')
        mp12 = MaxPool2DLayer(conv1, pool_size=pool_sizes[1], name='l12 max_pool')
        out = Conv2DLayer(mp12,
                          nonlinearity=sigmoid,
                          num_filters=1,
                          filter_size=1,
                          pad='same',
                          name='prediction layer')

        out1 = MaxPool2DLayer(mp01, pool_size=pool_sizes[1])
        out1 = Conv2DLayer(out1,
                           num_filters=1,
                           filter_size=1,
                           nonlinearity=sigmoid,
                           name='decide network1 output')

        return [out, out1]

    def compile_forward_pass(self):
        return theano.function([self.input_X], self.output)

    def compile_evaluator(self):        
        return theano.function([self.input_X, self.targets], [self.get_recall(),
                                                              self.get_accuracy(),
                                                              self.get_loss(),
                                                              self.get_sub_loss(),
                                                              self.get_complexity()
                                                             ])
    
    def compile_trainer(self, learning_rate):
        obj = self.get_obj()

        params = []
        for layer in self.output_layers:
            params += lasagne.layers.get_all_params(layer, trainable=True)

        updates = lasagne.updates.adamax(obj,
                                         params,
                                         learning_rate=learning_rate)
        return theano.function([self.input_X, self.targets], 
                               [obj,
                                self.get_recall(),
                                self.get_accuracy(),
                                self.get_loss(),
                                self.get_sub_loss(),
                                self.get_complexity()
                               ],
                               updates=updates)
