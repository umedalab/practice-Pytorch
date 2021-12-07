from torch.utils.data import Dataset, DataLoader
import numpy as np

from loader.SkeletonFeaturesFile import SkeletonFeatures

class CustomPoseDataLoader(Dataset):

    def __init__(self, data_set_path, fname_bodydescriptor, observed_length, out_item_mode, transforms=None):
        ''' out_item_mode: It defines the loader mode (0: pose converted in feature, 1: pose returned as sequence as is it)
        '''
        self.data_set_path = data_set_path
        self.transforms = transforms
        self.observed_length = observed_length # 5
        self.clear()
        self.read_data_connectivity(fname_bodydescriptor)
        self.out_item_mode = out_item_mode
        #self.read_data_pose(fname_pose)

    def __getitem__(self, index):
        # Get the features in a custom made mode (direction)
        if self.out_item_mode == 0:
            all_features, label = SkeletonFeatures.get_pose_featurelabel(
                self.connectivity, self.max_keypoints, self.poses2d, self.labels, index, self.observed_length)
        elif self.out_item_mode == 1:
            all_features, label = SkeletonFeatures.get_pose_sequence(
                self.connectivity, self.max_keypoints, self.poses2d, self.labels, index, self.observed_length, -1, -1)
 
        #if self.transforms is not None:

        return {'feature': all_features, 'label': label}

    def __len__(self):
        return self.length

    def clear(self):
        self.num_line = None
        self.poses2d = None
        self.poses3d = None
        self.labels = None

    """ Dataset loader for images and associated labels
        The loader produces the threshold necessary to binarize the source 
        image.
    """
    def read_data_connectivity(self, fname_bodydescriptor):
        """ The function read the configuration file and pose information
        """
        self.connectivity, self.max_keypoints = SkeletonFeatures.load_connectivity(
            self.data_set_path + '/' + fname_bodydescriptor)
        #print('connectivity:{}'.format(connectivity))
        print('</read_data_connectivity>')

    """ Dataset loader for images and associated labels
        The loader produces the threshold necessary to binarize the source 
        image.
    """
    def read_data_pose(self, fname_pose):
        """ The function read the configuration file and pose information
        """
        num_line_tmp, poses2d_tmp, poses3d_tmp, labels_tmp = SkeletonFeatures.load_keypoints_labels(
            fname_pose)
        print('new>>2d{} {}'.format(type(poses2d_tmp), poses2d_tmp.shape))
        #print('>>3d{} {}'.format(type(poses3d_tmp), poses3d_tmp.shape))
        #self.data_set_path + '/' + fname_pose)
        if (self.num_line is None):
            self.num_line = num_line_tmp.copy(order='k')
            self.poses2d = poses2d_tmp.copy(order='k')
            #self.poses3d = poses3d_tmp.copy(order='k')
            self.labels = labels_tmp.copy(order='k')
        else:
            print('before??2d{} {}'.format(type(self.poses2d), self.poses2d.shape))
            #print('??3d{} {}'.format(type(self.poses3d), self.poses3d.shape))
            self.num_line = np.append(self.num_line, num_line_tmp, axis=0)
            self.poses2d = np.append(self.poses2d, poses2d_tmp, axis=0)
            #self.poses3d = np.append(self.poses3d, poses3d_tmp, axis=0)
            self.labels = np.append(self.labels, labels_tmp, axis=0)

        print('after]]{} {}'.format(type(self.poses2d), self.poses2d.shape))
        #print(']]{} {}'.format(type(self.poses3d), self.poses3d.shape))
        #print('poses2d:{}'.format(self.poses2d))

        self.length = len(self.poses2d) - self.observed_length
        print('</read_data_pose>')

    def read_data_pose_from_measurementpy(self, fname_pose, label_id):
        """ The function read the configuration file and pose information
            It is expected to get a file with a single skeleton information.
        """
        num_line_tmp, poses2d_tmp, poses3d_tmp, labels_tmp = SkeletonFeatures.load_keypoints_labels_from_measurementpy(
            fname_pose, label_id)
        #print('num_line_tmp[#{}]:{}'.format(len(num_line_tmp), num_line_tmp))
        #print('poses2d_tmp:{}'.format(poses2d_tmp))
        #print('poses3d_tmp:{}'.format(poses3d_tmp))
        #print('labels_tmp:{}'.format(labels_tmp))
        print('new>>2d{} {}'.format(type(poses2d_tmp), poses2d_tmp.shape))
        # print('>>3d{} {}'.format(type(poses3d_tmp), poses3d_tmp.shape))
        # self.data_set_path + '/' + fname_pose)
        if (self.num_line is None):
            self.num_line = num_line_tmp.copy(order='k')
            self.poses2d = poses2d_tmp.copy(order='k')
            # self.poses3d = poses3d_tmp.copy(order='k')
            self.labels = labels_tmp.copy(order='k')
        else:
            print('before??2d{} {}'.format(type(self.poses2d), self.poses2d.shape))
            # print('??3d{} {}'.format(type(self.poses3d), self.poses3d.shape))
            self.num_line = np.append(self.num_line, num_line_tmp, axis=0)
            self.poses2d = np.append(self.poses2d, poses2d_tmp, axis=0)
            # self.poses3d = np.append(self.poses3d, poses3d_tmp, axis=0)
            self.labels = np.append(self.labels, labels_tmp, axis=0)

        print('after]]{} {}'.format(type(self.poses2d), self.poses2d.shape))
        # print(']]{} {}'.format(type(self.poses3d), self.poses3d.shape))
        #print('poses2d:{}'.format(self.poses2d))

        self.length = len(self.poses2d) - self.observed_length
        print('</read_data_pose_mylabel>')

    def read_data_pose_from_measurementpy_multiactionid(self, fname_pose, skeleton_id, frame_in, frame_out, label_id):
        """ The function read the configuration file and pose information
            It is expected to get a file with a single skeleton information.
        """
        num_line_tmp, poses2d_tmp, poses3d_tmp, labels_tmp = SkeletonFeatures.load_keypoints_labels_from_measurementpy_multiactionid(
            fname_pose, skeleton_id, frame_in, frame_out, label_id)
        #print('num_line_tmp[#{}]:{}'.format(len(num_line_tmp), num_line_tmp))
        #print('poses2d_tmp:{}'.format(poses2d_tmp))
        #print('poses3d_tmp:{}'.format(poses3d_tmp))
        #print('labels_tmp:{}'.format(labels_tmp))
        print('new>>2d{} {}'.format(type(poses2d_tmp), poses2d_tmp.shape))
        # print('>>3d{} {}'.format(type(poses3d_tmp), poses3d_tmp.shape))
        # self.data_set_path + '/' + fname_pose)
        if (self.num_line is None):
            self.num_line = num_line_tmp.copy(order='k')
            self.poses2d = poses2d_tmp.copy(order='k')
            # self.poses3d = poses3d_tmp.copy(order='k')
            self.labels = labels_tmp.copy(order='k')
        else:
            print('before??2d{} {}'.format(type(self.poses2d), self.poses2d.shape))
            # print('??3d{} {}'.format(type(self.poses3d), self.poses3d.shape))
            self.num_line = np.append(self.num_line, num_line_tmp, axis=0)
            self.poses2d = np.append(self.poses2d, poses2d_tmp, axis=0)
            # self.poses3d = np.append(self.poses3d, poses3d_tmp, axis=0)
            self.labels = np.append(self.labels, labels_tmp, axis=0)

        print('after]]{} {}'.format(type(self.poses2d), self.poses2d.shape))
        # print(']]{} {}'.format(type(self.poses3d), self.poses3d.shape))
        #print('poses2d:{}'.format(self.poses2d))

        self.length = len(self.poses2d) - self.observed_length
        print('</read_data_pose_mylabel>')


    def read_data_pose_from_measurementpy_id(self, fname_pose, skeleton_id, labels, filters, w, h):
        """ The function read the configuration file and pose information
            It is expected to get a file with a single skeleton information.
        """
        num_line_tmp, poses2d_tmp, poses3d_tmp, labels_tmp = SkeletonFeatures.load_keypoints_labels_from_measurementpy_id(
            fname_pose, skeleton_id, labels, filters, w, h)
        #print('num_line_tmp[#{}]:{}'.format(len(num_line_tmp), num_line_tmp))
        #print('poses2d_tmp:{}'.format(poses2d_tmp))
        #print('poses3d_tmp:{}'.format(poses3d_tmp))
        #print('labels_tmp:{}'.format(labels_tmp))
        print('new>>2d{} {}'.format(type(poses2d_tmp), poses2d_tmp.shape))
        print('WARNING: CHECK IF THE DATA IS EMPTY')
        # print('>>3d{} {}'.format(type(poses3d_tmp), poses3d_tmp.shape))
        # self.data_set_path + '/' + fname_pose)
        if (self.num_line is None):
            self.num_line = num_line_tmp.copy(order='k')
            self.poses2d = poses2d_tmp.copy(order='k')
            # self.poses3d = poses3d_tmp.copy(order='k')
            self.labels = labels_tmp.copy(order='k')
        else:
            print('before??2d{} {}'.format(type(self.poses2d), self.poses2d.shape))
            # print('??3d{} {}'.format(type(self.poses3d), self.poses3d.shape))
            self.num_line = np.append(self.num_line, num_line_tmp, axis=0)
            self.poses2d = np.append(self.poses2d, poses2d_tmp, axis=0)
            # self.poses3d = np.append(self.poses3d, poses3d_tmp, axis=0)
            self.labels = np.append(self.labels, labels_tmp, axis=0)

        print('after]]{} {}'.format(type(self.poses2d), self.poses2d.shape))
        # print(']]{} {}'.format(type(self.poses3d), self.poses3d.shape))
        #print('poses2d:{}'.format(self.poses2d))

        self.length = len(self.poses2d) - self.observed_length
        print('</read_data_pose_mylabel>')


    def len_connectivity(self):
        return len(self.connectivity)
        
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
        
        
def main(args):
    """ Test the dataloader
    """
    # read the label file
    labels = CustomPoseDataLoader.read_label('/home/moro/workspace/work/RitecsTaira/ActionAnomaly/data/datasets/actions/customerlabels/025.txt')
    filters = [7]
    print('labels:{}'.format(labels))    
    # Data Loaders
    expected_observed_length = 20
    data_loader = CustomPoseDataLoader(observed_length=expected_observed_length, data_set_path='data',
                                       fname_bodydescriptor='connectivityCOCO18.txt')
    #data_loader.read_data_pose_from_measurementpy(
    #    '../../../data/datasets/actions/sitting/scene001.txt', label_id=2)
    data_loader.read_data_pose_from_measurementpy_id(
        '/home/moro/workspace/work/RitecsTaira/ActionAnomaly/data/datasets/actions/customer/025.txt', 0, labels, filters)
    #data_loader.read_data_pose_from_measurementpy_multiactionid(
    #    '/home/moro/workspace/work/RitecsTaira/ActionAnomaly/data/datasets/actions/customer/025.txt', 1, 0, 100, label_id=1)
        
        
        
    #print('data_loader item0:{}'.format(data_loader.__getitem__(2)))
   
def test_filters():
    filters = [4, 3, 1]
    val = [4]
    res = False
    for x in val: 
        if x in filters: 
            res = True
    print('is:{} {} {}'.format(filters, val, res))
   
    
if __name__ == "__main__":
   
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    args = parser.parse_args()

    main(args)    
