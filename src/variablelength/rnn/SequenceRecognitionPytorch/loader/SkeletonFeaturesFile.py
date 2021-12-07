# check https://github.com/pytorch/benchmark/blob/master/rnns/fastrnns/custom_lstms.py

import os
import numpy as np
import math
from collections import defaultdict

class SkeletonFeatures:

    @staticmethod
    def create_connectivity():

        v_connectivity = []
        
        v_connectivity.append([16, 14])
        v_connectivity.append([14, 0])
        v_connectivity.append([15, 0])
        v_connectivity.append([17, 15])
        v_connectivity.append([0, 1])
        v_connectivity.append([4, 3])
        v_connectivity.append([3, 2])
        v_connectivity.append([2, 1])
        v_connectivity.append([7, 6])
        v_connectivity.append([6, 5])
        v_connectivity.append([5, 1])
        v_connectivity.append([10, 9])
        v_connectivity.append([9, 8])
        v_connectivity.append([8, 1])
        v_connectivity.append([13, 12])
        v_connectivity.append([12, 11])
        v_connectivity.append([11, 1])

        return np.array(v_connectivity)

    @staticmethod
    def angle2dvectors(a, b, offset):
        return math.atan2(a[1] - b[1], a[0] - b[0]) + offset;

    @staticmethod
    def skeleton2feature(connectivity, max_keypoints, p2d):
        features = []
        if len(p2d) < max_keypoints:
            return False, features
        kInvalidKeypoint2D = -1
        #print('skeleton2feature #:{}'.format(len(connectivity)))
        for i in range(0, len(connectivity)):
            idxA = connectivity[i][0]
            idxB = connectivity[i][1]
            #print('idxA={} idxB={}'.format(idxA, idxB))
            if ((p2d[idxA][0] != kInvalidKeypoint2D or p2d[idxA][1] != kInvalidKeypoint2D) and (p2d[idxB][0] != kInvalidKeypoint2D or p2d[idxB][1] != kInvalidKeypoint2D)):
                v = SkeletonFeatures.angle2dvectors(p2d[idxA], p2d[idxB], math.pi) / (math.pi * 2)
                features.append(v)
            else: 
                features.append(0);
        return True, np.array(features)

    @staticmethod
    def skeleton2sequence_position(connectivity, max_keypoints, p2d):
        features = []
        if len(p2d) < max_keypoints:
            return False, features
        kInvalidKeypoint2D = -1
        #print('skeleton2feature #:{}'.format(len(connectivity)))
        for i in range(0, len(p2d)):
            x = p2d[i][0]
            y = p2d[i][1]
            features.append([x, y])
        return True, np.array(features)

    @staticmethod
    def skeleton2sequence_position_norm(connectivity, max_keypoints, p2d, w, h):
        features = []
        if len(p2d) < max_keypoints:
            return False, features
        kInvalidKeypoint2D = -1
        #print('skeleton2feature #:{}'.format(len(connectivity)))
        for i in range(0, len(p2d)):
            x = p2d[i][0] / w * 2.0 - 1.0
            y = p2d[i][1] / h * 2.0 - 1.0
            features.append([x, y])
        return True, np.array(features)


    @staticmethod
    def words2array(words):
        vals = []
        for w in words:
            vals.append(float(w))
        return vals

    @staticmethod
    def array2pose(arr):
        vals = []
        for i in range(0, len(arr) // 2):
            vals.append([arr[i * 2], arr[i * 2 + 1]])
        return vals

    @staticmethod
    def load_keypoints(fname):
        lines = []
        with open(fname) as f:
            lines = f.readlines()

        poses = []
        count = 0
        for line in lines:
           count += 1
           #print(f'line {count}: {line}') 
           words = line.split()
           vals = SkeletonFeatures.words2array(words)
           pose = SkeletonFeatures.array2pose(vals)
           #print('pose:{}'.format(pose)) 
           poses.append(pose)

        return np.array(poses)

    @staticmethod
    def load_connectivity(fname):
        lines = []
        with open(fname) as f:
            lines = f.readlines()

        connectivity = []
        count = 0
        max_keypoints = 0
        for line in lines:
           count += 1
           #print(f'line {count}: {line}') 
           words = line.split()
           if len(words) >= 2:
               conn = [int(words[0]), int(words[1])]
               max_keypoints = max(max_keypoints, int(words[0]))
               max_keypoints = max(max_keypoints, int(words[1]))
               connectivity.append(conn)
        return np.array(connectivity), max_keypoints + 1

    @staticmethod
    def load_keypoints_labels(fname):
        lines = []
        with open(fname) as f:
            lines = f.readlines()

        num_line = []
        labels = []
        poses2d = []
        poses3d = []
        s = len(lines)
        for i in range(0, s, 3):
            # split the index
            words = lines[i].split()
            index = int(words[0])
            # split the pose 2D
            words = lines[i + 1].split()
            pose2d = []
            if len(words) > 0:
                vals = SkeletonFeatures.words2array(words)
                pose2d = SkeletonFeatures.array2pose(vals)
            # split the pose 3D
            words = lines[i + 2].split()
            pose3d = []
            if len(words) > 0:
                vals = SkeletonFeatures.words2array(words)
                pose3d = SkeletonFeatures.array2pose(vals)
            if index >= 0:
                num_line.append(i / 3)
                labels.append(index)
                poses2d.append(pose2d)
                poses3d.append(pose3d)
        return np.array(num_line), np.array(poses2d), np.array(poses3d), np.array(labels)

    @staticmethod
    def load_keypoints_labels_from_measurementpy(fname, label_id):
        """ This function read a file in a format
            frame id pos0x pos0y pos1x pos1y ...
            It combines all the skeleton information together and it assign the label id
        """
        print('[i] open:{}'.format(fname))
        lines = []
        with open(fname) as f:
            lines = f.readlines()

        num_line = defaultdict(list)
        labels = defaultdict(list)
        poses2d = defaultdict(list)
        poses3d = defaultdict(list)
        s = len(lines)
        for i in range(0, s):
            #print('lines[{}]:{}'.format(i, lines[i]))
            # split the index
            words = lines[i].split()
            nline = int(words[0])
            index = int(words[1])
            #print('nline:{} index:{}'.format(nline, index))
            # split the pose 2D
            pose2d = []
            for k in range(2, len(words), 2):
                pose2d.append([int(words[k]), int(words[k + 1])])

            #print('nline:{}'.format(nline))
            #print('index:{}'.format(index))
            #print('pose2d:{}'.format(pose2d))
            num_line[index].append(nline)
            poses2d[index].append(pose2d)
            labels[index].append(label_id)
        #print('num_line[#{}]:{}'.format(len(num_line), num_line))
        num_line_out = []
        for k, v in num_line.items():
            #print('k:{} v:{}'.format(k, v))
            for v0 in v:
                num_line_out.append(v0)
        poses2d_out = []
        for k, v in poses2d.items():
            #print('k:{} v:{}'.format(k, v))
            for v0 in v:
                #print('v0:{}'.format(v0))
                poses2d_out.append(v0)
        labels_out = []
        for k, v in labels.items():
            #print('k:{} v:{}'.format(k, v))
            for v0 in v:
                labels_out.append(v0)
        return np.array(num_line_out), np.array(poses2d_out), np.array(poses3d), np.array(labels_out)


    @staticmethod
    def load_keypoints_labels_from_measurementpy_multiactionid(fname, skeleton_id, frame_in, frame_out, label_id):
        """ This function read a file in a format
            frame id pos0x pos0y pos1x pos1y ...
            It expects that the file contains multiple actions and id
            it collects only the information from a specific id and inside a specific range of frames
        """
        print('[i] open:{}'.format(fname))
        lines = []
        with open(fname) as f:
            lines = f.readlines()

        num_line = defaultdict(list)
        labels = defaultdict(list)
        poses2d = defaultdict(list)
        poses3d = defaultdict(list)
        s = len(lines)
        for i in range(0, s):
            #print('lines[{}]:{}'.format(i, lines[i]))
            # split the index
            words = lines[i].split()
            nline = int(words[0])
            index = int(words[1])
            #print('nline:{} index:{}'.format(nline, index))
            # split the pose 2D
            pose2d = []
            for k in range(2, len(words), 2):
                pose2d.append([int(words[k]), int(words[k + 1])])

            #print('nline:{}'.format(nline))
            #print('index:{}'.format(index))
            #print('pose2d:{}'.format(pose2d))
            if index == skeleton_id and nline >= frame_in and nline <= frame_out:
                num_line[index].append(nline)
                poses2d[index].append(pose2d)
                labels[index].append(label_id)
        #print('num_line[#{}]:{}'.format(len(num_line), num_line))
        num_line_out = []
        for k, v in num_line.items():
            #print('k:{} v:{}'.format(k, v))
            for v0 in v:
                num_line_out.append(v0)
        poses2d_out = []
        for k, v in poses2d.items():

            #print('k:{} v:{}'.format(k, v))
            for v0 in v:
                #print('v0:{}'.format(v0))
                poses2d_out.append(v0)
        labels_out = []
        for k, v in labels.items():
            #print('k:{} v:{}'.format(k, v))
            for v0 in v:
                labels_out.append(v0)
        return np.array(num_line_out), np.array(poses2d_out), np.array(poses3d), np.array(labels_out)


    @staticmethod
    def which_action(labels, num_frame):
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
    def load_keypoints_labels_from_measurementpy_id(fname, skeleton_id, labels_in, filters, w, h):
        """ This function read a file in a format
            frame id pos0x pos0y pos1x pos1y ...
            It expects that the file contains multiple actions and id
            it collects only the information from a specific id and inside a specific range of frames
            
        """
        print('[i] open:{}'.format(fname))

        lines = []
        with open(fname) as f:
            lines = f.readlines()

        num_line = defaultdict(list)
        labels = defaultdict(list)
        poses2d = defaultdict(list)
        poses3d = defaultdict(list)
        s = len(lines)
        for i in range(0, s):
            #print('lines[{}]:{}'.format(i, lines[i]))
            # split the index
            words = lines[i].split()
            nline = int(words[0])
            index = int(words[1])
            #print('nline:{} index:{}'.format(nline, index))
            # split the pose 2D
            pose2d = []
            for k in range(2, len(words), 2):
                x = int(words[k])
                y = int(words[k + 1])
                if w > 0 and h > 0:
                    x = x / w * 2.0 - 1.0
                    y = y / h * 2.0 - 1.0
                pose2d.append([x, y])

            #print('nline:{}'.format(nline))
            #print('index:{}'.format(index))
            #print('pose2d:{}'.format(pose2d))
            if index == skeleton_id:
                actions = SkeletonFeatures.which_action(labels_in, nline)
                #print('nline:{} {} {}'.format(nline, labels_in, actions))
                if actions[0] == True:
                    if actions[1] not in filters:
                        num_line[index].append(nline)
                        poses2d[index].append(pose2d)
                        labels[index].append(actions[1])
        #print('num_line[#{}]:{}'.format(len(num_line), num_line))
        num_line_out = []
        for k, v in num_line.items():
            #print('k:{} v:{}'.format(k, v))
            for v0 in v:
                num_line_out.append(v0)
        poses2d_out = []
        for k, v in poses2d.items():
            #print('k:{} v:{}'.format(k, v))
            for v0 in v:
                #print('v0:{}'.format(v0))
                poses2d_out.append(v0)
        labels_out = []
        for k, v in labels.items():
            #print('k:{} v:{}'.format(k, v))
            for v0 in v:
                labels_out.append(v0)
        return np.array(num_line_out), np.array(poses2d_out), np.array(poses3d), np.array(labels_out)


    @staticmethod
    def get_pose_featurelabel(connectivity, max_keypoints, poses, labels, start_idx, num_observations):
        if len(poses) < start_idx + num_observations:
            return None, None

        all_features = []
        all_labels = []

        for i in range(start_idx, start_idx + num_observations):
            # get the features from the pose
            res, features = SkeletonFeatures.skeleton2feature(connectivity, max_keypoints, poses[i])
            if res == False:
                return None, None
            all_features.append(features)
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
        return np.concatenate(all_features, axis = 0), label
        
    @staticmethod
    def get_pose_sequence(connectivity, max_keypoints, poses, labels, start_idx, num_observations, w, h):
        if len(poses) < start_idx + num_observations:
            return None, None

        all_features = []
        all_labels = []

        for i in range(start_idx, start_idx + num_observations):
            # get the features from the pose
            if w > 0 and h > 0:
                res, features = SkeletonFeatures.skeleton2sequence_position_norm(connectivity, max_keypoints, poses[i], w, h)
            else:
                res, features = SkeletonFeatures.skeleton2sequence_position(connectivity, max_keypoints, poses[i])
            if res == False:
                return None, None
            all_features.append(features)
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
