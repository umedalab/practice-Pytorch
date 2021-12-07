# check https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py

import os
import numpy as np
import math
from collections import defaultdict
from collections import OrderedDict

class ObjectLabels:
    @staticmethod
    def read_label(fname):
        """ it reads a file in the format
            in out label in out label ...

            It returns a container with the relative information
        """
        with open(fname) as f:
            content = f.readlines()
        labels = []
        for x in content:
            #print('x:{}'.format(x))
            # minimum action size 3 numbers 2 spaces
            if len(x) < 5: continue
            words = x.strip().split(' ')
            print('words:{}'.format(words))
            if len(words) > 0:
                s = len(words)
                for i in range(0, s, 3):
                    frame_in = int(words[i])
                    frame_out = int(words[i + 1])
                    label = int(words[i + 2])
                    labels.append((frame_in, frame_out, label))

        print(labels)
        return labels

