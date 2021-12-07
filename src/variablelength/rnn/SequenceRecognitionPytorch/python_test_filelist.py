import sys
import os
from os import walk


class TrainingData:

    def get_directories(self, mypath):

        f = []
        for (dirpath, dirnames, filenames) in walk(mypath):
            f.extend(dirnames)
            break
        #print('f:{}'.format(f))
        return f

    def get_files_fullpath(self, mypath):

        f = []
        for (dirpath, dirnames, filenames) in walk(mypath):
            f.extend(filenames)
            break
        #print('f:{}'.format(f))
        return f

    def get_pair(self, mypath):
        # get the container with label
        directories = self.get_directories(mypath)
        for d in directories:
            if d == 'data':
                data = self.get_files_fullpath(mypath + '/' + d)
            if d == 'labels':
                label = self.get_files_fullpath(mypath + '/' + d)
        res = []
        for i in range(0, len(data)):
            for j in range(0, len(label)):
                #print('pair?:{} {}'.format(data[i], label[j]))
                if data[i] == label[j]:
                    #print('pair:{}'.format(data[i]))
                    res.append([mypath + '/data/' + data[i], mypath + '/labels/' + label[j]])
        #print('res:{}'.format(res))
        return res


def get_files(path, f):

    for (dirpath, dirnames, filenames) in walk(path):
        print('dirpath:{}'.format(dirpath))
        print('dirnames:{}'.format(dirnames))
        print('filenames:{}'.format(filenames))
        for d in dirnames:
            g = get_files(path + '/' + d, f)
            for v in g:
                f.append(v)
        break
    print('f:{}'.format(f))


def main(args):

    mypath = '/home/moro/workspace/work/RitecsTaira/ActionAnomaly/data/datasets/actions/training/scene001'
    
    tdata = TrainingData()
    res = tdata.get_pair(mypath)
    print(res)
    return

    f = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        print('dirpath:{}'.format(dirpath))
        print('dirnames:{}'.format(dirnames))
        print('filenames:{}'.format(filenames))
        f.extend(dirnames)
        break
    print('f:{}'.format(f))

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    args = parser.parse_args()

    main(args)
