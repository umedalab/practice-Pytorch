# check https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py

import os
import numpy as np
import math
from collections import defaultdict
from collections import OrderedDict
from loader.ObjectLabelsFile import ObjectLabels

class ObjectFeatures:

    @staticmethod
    def which_label(labels, num_frame):
        """ it returns the action corresponding to a selected frame number
        """
        valid = False
        key = 0
        for x in labels:
            if x[0] <= num_frame <= x[1]:
                #print('x:{} num_frame:{}'.format(x, num_frame))
                key = x[2]
                valid = True
        return valid, key

    @staticmethod
    def which_objectlabel2id_norm(objectlabel2id, label):
        if label in objectlabel2id:
            return objectlabel2id[label]
        return 0
        for i in range(0, len(objectlabel2id)):
            if objectlabel2id[i] == label:
                return float(i) / len(objectlabel2id)
        return 0

    @staticmethod
    def load_keypoints_labels_from_measurementpy_id(fname, labels_in, filters, objectlabel2id):
        """ This function read a file in a format
            frame id pos0x pos0y pos1x pos1y ...
            It expects that the file contains multiple actions and id
            it collects only the information from a specific id and inside a specific range of frames
            
        """
        print('[i] open:{}'.format(fname))

        lines = []
        with open(fname) as f:
            lines = f.readlines()

        labels = defaultdict(list)
        features = defaultdict(list)
        s = len(lines)
        for i in range(0, s):
            #print('lines[{}]:{}'.format(i, lines[i]))
            #continue
            # split the index
            words = lines[i].split()
            nline = int(words[0])
            objectlabel2id_norm = ObjectFeatures.which_objectlabel2id_norm(objectlabel2id, words[1])
            #print('nline:{} index:{}'.format(nline, index))
            # split the pose 2D
            feat = []
            for k in range(3, len(words)):
                feat.append(float(words[k]))

            label = ObjectFeatures.which_label(labels_in, nline)
            #print('nline:{} {} {}'.format(nline, objectlabel2id_norm, feat))
            if label[0] == True:
                if label[1] not in filters:
                    features[nline].append(objectlabel2id_norm)
                    for f in feat:
                        features[nline].append(f)
                    labels[nline] = label[1]

        features_out = []
        # this solution just pick the latest valid for each id
        for k in features.items():
            #print('k:{}'.format(k))
            # create a vector of n elements (15)
            n = np.zeros(12)
            #print(n)
            # for each key select the current points and sum
            for i in range(0, len(k[1]), 5):
                idx = round(k[1][i] / 0.333) * 4 - 4
                #print(idx)
                #print(k[1][i])
                #n[idx] = k[1][i]
                for j in range(1, 5):
                    n[idx + j - 1] = k[1][i + j]
            #print('n:{}'.format(n))
            # add the elements in the right position
            # normalize
            features_out.append(n)

        labels_out = []
        for k, v in labels.items():
            #print('k:{} v:{}'.format(k, v))
            labels_out.append(v)

        return np.array(features_out), np.array(labels_out)

    @staticmethod
    def get_featurelabel_sequence(features, labels, start_idx, num_observations):
        #print('f:{}'.format(features))
        #print('l:{}'.format(labels))
        if len(features) < start_idx + num_observations:
            return None, None

        all_features = []
        all_labels = []

        for i in range(start_idx, start_idx + num_observations):
            # get the features from the pose
            all_features.append(features[i])
            if labels is not None:
                all_labels.append(labels[i])

        # https://stackoverflow.com/questions/6252280/find-the-most-frequent-number-in-a-numpy-array
        label = np.array([])
        if len(all_labels) > 0:
            label = np.append(label, np.bincount(all_labels).argmax())
        else:
            label = np.append(label, 0)

        # flatten the features
        # https://stackoverflow.com/questions/27516849/how-to-convert-list-of-numpy-arrays-into-single-numpy-array    <--
        # https://stackoverflow.com/questions/952914/how-to-make-a-flat-list-out-of-a-list-of-lists
        return np.array(all_features), label        


def main(args):
    """ Test the dataloader
    """

    filters = [] # it filters the falling action (?)
    objectlabel2id = dict()
    objectlabel2id['person'] = 0.333
    objectlabel2id['personsleep'] = 0.333
    objectlabel2id['bed'] = 0.666
    objectlabel2id['wheelchair'] = 0.999

    labels = ObjectLabels.read_label('labels0.txt')
    feat, lab = ObjectFeatures.load_keypoints_labels_from_measurementpy_id('result_essential0.txt', labels, filters, objectlabel2id)
    print('labels:{}'.format(labels))    
    print('feat:{}'.format(feat))
    print('lab:{}'.format(lab))
    
if __name__ == "__main__":
   
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    args = parser.parse_args()

    main(args)    
