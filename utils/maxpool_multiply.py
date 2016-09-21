import lasagne

from lasagne.layers import MaxPool2DLayer
from lasagne.layers.base import MergeLayer
from collections import OrderedDict

class MaxPoolMultiplyLayer(MergeLayer):
    def __init__(self, activation_layer_small, activation_layer_big, pool_size, name=None):
        self.input_shapes = [activation_layer_small.output_shape, activation_layer_big.output_shape]
        self.max_pool_layer = MaxPool2DLayer(activation_layer_big, pool_size=pool_size)
        self.input_layers = [activation_layer_small, activation_layer_big]
        self.name = name
        self.params = OrderedDict()
        self.get_output_kwargs = []
        
    def get_output_shape_for(self, input_shapes):
        return self.input_shapes[0]
    
    def get_output_for(self, inputs, **kwargs):
        downsampled_big_activation = lasagne.layers.get_output(self.max_pool_layer)
        return downsampled_big_activation * lasagne.layers.get_output(self.input_layers[0])
        