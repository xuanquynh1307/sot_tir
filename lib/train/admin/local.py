class EnvironmentSettings:
    def __init__(self):
        # Base directory for saving network checkpoints.
        self.workspace_dir = 'C:/Users/PC/Documents/DC/single_object_tracking/sot_tir'
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.pretrained_networks = self.workspace_dir + '/pretrained_networks/'
        self.lsotb_dir = 'C:/Users/PC/Documents/DC/01-Datasets/LSOTB-TIR-2023'
