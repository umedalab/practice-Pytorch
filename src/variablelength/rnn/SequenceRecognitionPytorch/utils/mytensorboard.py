# ### OUR CODE ###
import os.path as osp
import tensorboardX

# create a new class inheriting from tensorboardX.SummaryWriter
class SummaryWriter(tensorboardX.SummaryWriter):
    def __init__(self, log_dir=None, comment="", **kwargs):
        super().__init__(log_dir, comment, **kwargs)
    # create a new function that will take dictionary as input and uses built-in add_scalar() function
    # that function combines all plots into one subgroup by a tag
    def add_scalar_dict(self, dictionary, global_step, tag=None):
        for name, val in dictionary.items():
            #print('name:{} val:{}',format(name, val))
            if tag is not None:
                name = osp.join(tag, name)
            self.add_scalar(name, val.item(), global_step)

    # create a new function that will show the model
    def add_model(self, model, indata):
        print(model)
        print(indata)
        self.add_graph(model, indata)



# define variables as globals to have an access everywhere
args = None  # will be replaced to argparse dictionary
total_epochs = 0  # init total number of epochs
global_iter = 0  # init total number of iterations

name = "exp-000"  # init experiment name
log_dir = "../../../experiments"  # init path where tensorboard logs will be stored
# (if log_dir is not specified writer object will automatically generate filename)
# Log files will be saved in 'experiments/exp-000'
# create our custom logger
logger = SummaryWriter(log_dir=osp.join(log_dir, name))

# ### END OF OUR CODE ###
