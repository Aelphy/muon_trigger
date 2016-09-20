import os
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
                 c_sub_objs=[1e-3],
                 c_sub_obj_cs=[1e-3],
                 mul=True,
                 pool_sizes=[2, 2, 5],
                 num_filters=[1, 1, 3],
                 filter_sizes=[1, 3, 3]
                ):
        self.img_shape = img_shape
        self.pool_sizes = pool_sizes
        
        self.c_sub_objs = c_sub_objs
        self.c_sub_obj_cs = c_sub_obj_cs
        self.c_complexity = c_complexity
        self.c = c
        
        self.input_X = T.tensor4('inputs')
        self.targets = T.tensor4('targets')
        
        assert(len(pool_sizes) == len(num_filters))
        self.num_cascades = len(pool_sizes) - 1
        
        self.output_layer, self.downscaled_activation_layers, self.branches = self.build_network(num_filters, filter_sizes)
        
        assert(len(self.decide_layers) == len(self.c_sub_objs))
        assert(len(self.decide_layers) == len(self.c_sub_obj_cs))
        self.output = self.build_output(mul)
        self.train = self.compile_trainer(learning_rate)
        self.evaluate = self.compile_evaluator()
        self.predict = self.compile_forward_pass()
        
    def build_output(self, mul=True):
        if mul:
            ai_layer = MaxPool2DLayer(self.downscaled_activation_layers[-1], pool_size=self.pool_sizes[-1])
            answers = lasagne.layers.get_output(ai_layer, ai_next)
        else:
            answers = 1
            
        return answers * lasagne.layers.get_output(self.output_layer, self.input_X), downscaled_activation_layers
    
    def compute_loss(self, a, t, c):
        return -(t * T.log(a) + c * (1.0 - t) * T.log(1.0 - a)).mean()

    def get_sub_loss(self):
        sub_obj = 0
        
        target_transform_input_layer = InputLayer(self.img_shape, self.targets, name='target transform input layer')
        
        for i, output_layer in enumerate(self.downscaled_activation_layers):
            sub_answer = lasagne.layers.get_output(output_layer, self.input_X)
            
            downsample_target_layer = MaxPool2DLayer(target_transform_input_layer, pool_size=self.pool_sizes[i])
            targets = lasagne.layers.get_output(downsample_target_layer, self.targets)
            
            sub_obj += self.compute_loss(sub_answer.ravel(),
                                         targets.ravel(),
                                         self.c_sub_obj_cs[i]) * self.c_sub_objs[i]
            
        return sub_obj
    
    def get_complexity(self):
        complexity = 0
        cumprod = 1
        max_complexity = 0
        
        for output_layer in self.output_layers[1:]:
            cumprod *= lasagne.layers.get_output(output_layer, self.input_X)
            complexity += (cumprod * (1 - self.targets)).sum()
            max_complexity += (1 - self.targets).sum()
            
        return complexity * self.c_complexity / max_complexity
    
    def get_loss(self):
        return self.compute_loss(self.output.ravel(), self.targets.ravel(), self.c)
    
    def get_obj(self):                    
        return self.get_loss() + self.get_sub_loss() + self.c_complexity * self.get_complexity()
    
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

    def build_network(self, num_filters, filter_sizes):
        net = InputLayer((None, 1) + tuple(self.img_shape),
                         self.input_X,
                         name='network input')

        # Build network
        for i in range(self.num_cascades + 1):
            net = Conv2DLayer(net,
                              nonlinearity=elu,
                              num_filters=num_filters[i],
                              filter_size=filter_sizes[i],
                              pad='same',
                              name='conv {}'.format(i + 1))
            net = MaxPool2DLayer(net,
                                 pool_size=self.pool_sizes[i],
                                 name='Max Pool {} {}'.format(i + 1, i + 2))

        
        out = Conv2DLayer(net,
                          nonlinearity=sigmoid,
                          num_filters=1,
                          filter_size=1,
                          pad='same',
                          name='prediction layer')
        
        branches = [None] * self.num_cascades
        
        layers = lasagne.layers.get_all_layers(out)
        
        # Build branches
        for i in range(self.num_cascades):
            branches[i] = Conv2DLayer(layers[(i + 1) * 2],
                                      num_filters=1,
                                      filter_size=1,
                                      nonlinearity=sigmoid,
                                      name='decide network {} output'.format(i + 1))
        
        # TODO(Aelphy): use here new layer in order to perform automatic graph image building
        downscaled_activation_layers = []
        ai_next = lasagne.layers.get_output(branches[0], self.input_X)

        for i in range(self.num_cascades - 1):
            downscaled_activation_layers.append(InputLayer(ai_next.shape, ai_next))
            ai_layer = MaxPool2DLayer(downscaled_activation_layers[-1], pool_size=self.pool_sizes[i + 1])
            ai = lasagne.layers.get_output(ai_layer, ai_next)
            ai_next = ai * lasagne.layers.get_output(branches[i + 1], self.input_X)

        downscaled_activation_layers.append(InputLayer(ai_next.shape, ai_next))
        
        return out, downscaled_activation_layers, branches

    def compile_forward_pass(self):
        return theano.function([self.input_X], self.output)

    def compile_evaluator(self):        
        return theano.function([self.input_X, self.targets], [self.get_obj(),
                                                              self.get_recall(),
                                                              self.get_precision(),
                                                              self.get_accuracy(),
                                                              self.get_loss(),
                                                              self.get_sub_loss(),
                                                              self.get_complexity()
                                                             ])
    
    def compile_trainer(self, learning_rate):
        obj = self.get_obj()

        params = lasagne.layers.get_all_params(layer, trainable=True)

        for branch in self.branches:
            params += lasagne.layers.get_all_params(branch, trainable=True)

        updates = lasagne.updates.adamax(obj,
                                         params,
                                         learning_rate=learning_rate)
        return theano.function([self.input_X, self.targets], 
                               [obj,
                                self.get_recall(),
                                self.get_precision(),
                                self.get_accuracy(),
                                self.get_loss(),
                                self.get_sub_loss(),
                                self.get_complexity()
                               ],
                               updates=updates)
    
    def save(self, path, name):
        for i, output_layer in enumerate(self.output_layers):
            np.savez(os.path.join(path, name + str(i)), *lasagne.layers.get_all_param_values(output_layer))
    
    def load(self, path, name):
        for j, output_layer in enumerate(self.output_layers):
            with np.load(os.path.join(path, name + str(j) + '.npz')) as f:
                param_values = [f['arr_%d' % i] for i in range(len(f.files))]
                lasagne.layers.set_all_param_values(output_layer, param_values)
