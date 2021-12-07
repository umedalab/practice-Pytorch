import sys
import os
import datetime

import numpy as np
import collections


class TrackObjectInfo:
    """ Class to collect tracker information
    """
    def __init__(self, max_len):
        self.t_start = 0
        self.t_stop = 0
        self.d = collections.deque(maxlen=max_len)
        
    def create(self, t, pose):
        self.t_start = t
        self.t_stop = t
        self.d.append(pose)

    def update(self, t, pose):
        self.t_stop = t
        self.d.append(pose)
         
class CollectTracker:
    """ Class to collect all the tracker information
    """
    def __init__(self, max_len):
        self.container = []
        # The dictionary is defined as follow
        # 0: 0 time found, 1 time last
        # 1: 0 objectA, 1 objectB pair of objects x0y0x1y1ID
        # 2: 0 xyzA, 1 xyzB pair of objects
        # 3: bool if recorded (default false)
        self.my_dictionary = dict()
        self.max_len = max_len

    def add(self, index, pose):
        """ Add an object and collect. Timestamp information of the add call is calculated internally
        """
        t = datetime.datetime.now()
        self.add_time_bbox(t, index, pose)

    def add_time_bbox(self, t, key, pose):
        """ Add an object and collect with the timestamp information
        """
        # if the key is not found, it adds the result
        # for the dictionary description, check above
        if key not in self.my_dictionary.keys():
            obj = TrackObjectInfo(max_len = self.max_len)
            obj.create(t, pose)
            self.my_dictionary[key] = obj
        else:
            #self.my_dictionary[key][0][1] = t
            # The position and object information is updated until the image is not saved
            #self.my_dictionary[key][1][0] = objA
            #self.my_dictionary[key][1][1] = objB
            #self.my_dictionary[key][2][0] = xyzA
            #self.my_dictionary[key][2][1] = xyzB
            self.my_dictionary[key].update(t, pose)


    def get_poses_len(self, key):
        if key in self.my_dictionary.keys():
           return len(self.my_dictionary[key].d)
        return 0

    def get_poses(self, key):
        if key in self.my_dictionary.keys():
           return self.my_dictionary[key].d
        return None

    def show(self):
        print('self.my_dictionary:{}'.format(self.my_dictionary))

    def timeout_sec(self, threshold):
        """ If the difference between the last observed and current time is 
            over the threshold, delete the element from container
        """
        t = datetime.datetime.now()
        delete = []
        for dic in self.my_dictionary:
            #print('dic:{}'.format(dic))
            tdiff = (t - self.my_dictionary[dic].t_stop).total_seconds()
            #print('val:{} | {} = {}'.format(self.my_dictionary[dic], threshold, tdiff))
            if tdiff > threshold:
                delete.append(dic)
        #print('delete_timeout:{}'.format(delete))
        # https://stackoverflow.com/questions/11277432/how-to-remove-a-key-from-a-python-dictionary
        for key in delete:
            self.my_dictionary.pop(key, None)


