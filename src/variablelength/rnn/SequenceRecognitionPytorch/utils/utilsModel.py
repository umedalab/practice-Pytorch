import torch

class UtilsModel():
    """ Collection of utilities for a model
    """

    @staticmethod
    def network_inputsize(net):
        ''' Get the input network size
            https://stackoverflow.com/questions/63131273/how-to-get-input-tensor-shape-of-an-unknown-pytorch-model
        '''
        shape_of_first_layer = list(net.parameters())[0].shape #shape_of_first_layer
        N,C = shape_of_first_layer[:2]
        return N, C

