"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import re

def freeze(model):
    """ Set all layers in a model to non-trainable.

    The weights for these layers will not be updated during training.

    This function modifies the given model in-place,
    but it also returns the modified model to allow easy chaining with other functions.
    """
    for layer in model.layers:
        layer.trainable = False
    return model

def freeze_resnet_blocks(model, layers, layer_name_base='res', inverse=False):
    """ Set the given block layers in a resnet model to non-trainable.

    The weights for these layers will not be updated during training.

    This function modifies the given model in-place,
    but it also returns the modified model to allow easy chaining with other functions.
    parameters: 
    model: the resnet model to modify 
    layers: list of integers corresponding to resnet blocks 
            eg: [1,2] correspond to blocks 1 and 2. Expressions of res1 and res2 will be created 
            and be matched on all layer names. The matched layers are set to non-trainable.
    inverse: if True, layer will be set to trainable instead of non-trainable. Useful in case of freezeing all layers and unfreezing last layers
    """
    all_layers = [(layer.name, layer) for layer in model.layers]
    expressions = []
    layers_to_freeze = []
    for layer_num in layers:
        base = layer_name_base + str(layer_num)
        expression = '^' + base + '.*'
        regex = re.compile(expression)
        layer_matches = filter(lambda layer: regex.match(layer[0]), all_layers)
        layers_to_freeze += layer_matches
    
    for layer_name, layer in layers_to_freeze:
        if inverse:
            layer.trainable = True 
        else:
            layer.trainable = False
    return model  

def freeze_resnet_up_to(model, end_layer):
    """ Set all layers up to given layer in a resnet model to non-trainable.

    The weights for these layers will not be updated during training.

    This function modifies the given model in-place,
    but it also returns the modified model to allow easy chaining with other functions.
    parameters: 
    model: the resnet model to modify 
    end_layer: (int) layer number up to which to set layers as non-trainable. 
    """
    initial_layers = ['conv1', 'conv1_relu', 'pool1', 'padding_conv1']
    count = 0
    for layer in model.layers:
        if layer.name == initial_layers[count]:
            layer.trainable = False 
            count += 1
        if count == len(initial_layers): break 
    return freeze_resnet_blocks(model, list(range(1, end_layer + 1))) 