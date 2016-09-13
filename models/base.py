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


class Base(object):
    def __init__(self, img_shape=(640, 480), learning_rate=1e-3):
        self.output_layers = self.build_network(img_shape)
        self.train_network = self.compile_trainer(learning_rate)
        self.eval_network = self.compile_evaluator()
        self.forward_pass = self.compile_forward_pass()


    # TODO: could be modified
    # add cost as sum nonzeros elements in the middle layers
    def get_obj(self, targets, answers, c=1e-3):
        obj = (targets * T.log(targets * answers)).sum() +\
          c * ((1 - targets) * T.log(1 - (1 - targets) * answers)).sum()

        return obj

    # TODO: could be modified
    def compile_trainer(self, learning_rate):
        targets = T.tensor4('targets')
        input_X = T.tensor4('answers')

        answers = 1

        for output_layer in self.output_layers:
            answers *= lasagne.layers.get_output(output_layer, input_X)

        obj = self.get_obj(targets, answers)

        params = []
        for layer in self.output_layers:
            params += lasagne.layers.get_all_params(layer, trainable=True)

        updates = lasagne.updates.adamax(obj,
                                         params,
                                         learning_rate=learning_rate)
        return theano.function([targets, input_X], obj, updates=updates)

    # TODO: could be modified
    def build_network(self, img_shape):
        input_layer = InputLayer((None, 1) + tuple(img_shape),
                                  name='network input')

        l1 = Conv2DLayer(input_layer,
                         nonlinearity=rectify,
                         num_filters=6,
                         filter_size=(3, 3),
                         pad='same',
                         name='l1 conv')
        mp12 = MaxPool2DLayer(l1, pool_size=2, name='l12 max_pool')
        l2 = Conv2DLayer(mp12,
                         nonlinearity=rectify,
                         num_filters=12,
                         filter_size=(3, 3),
                         pad='same',
                         name='l2 conv')
        mp23 = MaxPool2DLayer(l2, pool_size=2, name='l23 max_pool')
        l3 = Conv2DLayer(mp23,
                         nonlinearity=rectify,
                         num_filters=24,
                         filter_size=(3, 3),
                         pad='same',
                         name='l3 conv')

        out1 = InputLayer((None, 6) + tuple(img_shape),
                          name='decide network1 input')
        #out1 = MaxPool2DLayer(out1, pool_size=)
        out1 = Conv2DLayer(out1,
                           num_filters=1,
                           filter_size=(1, 1),
                           nonlinearity=sigmoid,
                           name='decide network1 output')

        out2 = InputLayer((None, 12, img_shape[0] / 2, img_shape[1] / 2),
                           name='decide network2 input')
        out2 = Upscale2DLayer(out2,
                              scale_factor=2,
                              name='decide network2 upscale')
        out2 = Conv2DLayer(out2,
                           num_filters=1,
                           nonlinearity=sigmoid,
                           filter_size=(1, 1),
                           name='decide network2 output')
        #out2 = MaxPool2DLayer(out2, pool_size=)

        out3 = InputLayer((None, 24, img_shape[0] / 4, img_shape[1] / 4),
                           name='decide network3 input')
        out3 = Upscale2DLayer(out3,
                              scale_factor=4,
                              name='decide network3 upscale')
        out3 = Conv2DLayer(out3,
                           num_filters=1,
                           nonlinearity=sigmoid,
                           filter_size=(1, 1),
                           name='decide network3 output')
        #out3 = MaxPool2DLayer(out3, pool_size=)

        return [l3, out1, out2, out3]

    # TODO: could be modified
    def compile_forward_pass(self):
        input_X =  T.tensor4('input images')

        res = 1
        for output_layer in self.output_layers:
            res *= lasagne.layers.get_output(output_layer, input_X)

        return theano.function([input_X], res)

    def compile_evaluator(self):
        targets = T.tensor4('targets')
        answers = T.tensor4('answers')

        obj = self.get_obj(targets, answers)

        return theano.function([targets, answers], obj)
