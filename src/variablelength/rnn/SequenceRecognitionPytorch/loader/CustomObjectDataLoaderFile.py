from torch.utils.data import Dataset, DataLoader
import numpy as np

from loader.ObjectFeaturesFile import ObjectFeatures
from loader.ObjectLabelsFile import ObjectLabels

class CustomObjectDataLoader(Dataset):

    def __init__(self, observed_length, transforms=None):
        ''' out_item_mode: It defines the loader mode (0: pose converted in feature, 1: pose returned as sequence as is it)
        '''
        self.transforms = transforms
        self.observed_length = observed_length # 5
        self.clear()

    def __getitem__(self, index):
        # Get the features in a custom made mode (direction)
        all_features, label = ObjectFeatures.get_featurelabel_sequence(
                self.feat, self.lab, index, self.observed_length)
        return {'feature': all_features, 'label': label}

    def __len__(self):
        return self.length

    def clear(self):
        self.feat = None
        self.lab = None

    def read_data_pose_from_measurementpy_id(self, fname, labels_in, filters, objectlabel2id):
        """ The function read the configuration file and pose information
            It is expected to get a file with a single skeleton information.
        """
        feat, lab = ObjectFeatures.load_keypoints_labels_from_measurementpy_id(
            fname, labels_in, filters, objectlabel2id)
        print('new>>2d{} {}'.format(type(feat), feat.shape))
        print('WARNING: CHECK IF THE DATA IS EMPTY')
        # print('>>3d{} {}'.format(type(poses3d_tmp), poses3d_tmp.shape))
        # self.data_set_path + '/' + fname_pose)
        if (self.feat is None):
            self.feat = feat.copy(order='k')
            # self.poses3d = poses3d_tmp.copy(order='k')
            self.lab = lab.copy(order='k')
        else:
            print('before??2d{} {}'.format(type(self.feat), self.feat.shape))
            # print('??3d{} {}'.format(type(self.poses3d), self.poses3d.shape))
            self.feat = np.append(self.feat, feat, axis=0)
            # self.poses3d = np.append(self.poses3d, poses3d_tmp, axis=0)
            self.lab = np.append(self.lab, lab, axis=0)

        print('after]]{} {}'.format(type(self.feat), self.feat.shape))
        # print(']]{} {}'.format(type(self.poses3d), self.poses3d.shape))
        #print('poses2d:{}'.format(self.poses2d))

        self.length = len(self.feat) - (self.observed_length - 1)
        print('</read_data_pose_mylabel>')
        
        
def main(args):
    """ Test the dataloader
    """

    filters = [] # it filters the falling action (?)
    objectlabel2id = dict()
    objectlabel2id['person'] = 0.333
    objectlabel2id['personsleep'] = 0.333
    objectlabel2id['bed'] = 0.666
    objectlabel2id['wheelchair'] = 0.999

    # read the label file
    labels = ObjectLabels.read_label('labels0.txt')
    print('labels:{}'.format(labels))    
    # Data Loaders
    expected_observed_length = 2
    data_loader = CustomObjectDataLoader(observed_length=expected_observed_length)
    data_loader.read_data_pose_from_measurementpy_id('result_essential0.txt', labels, filters, objectlabel2id)   
    print('len:{}'.format(data_loader.__len__()))
    for i in range(0, data_loader.__len__()):
        print(data_loader.__getitem__(i))
    
if __name__ == "__main__":
   
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    args = parser.parse_args()

    main(args)    
