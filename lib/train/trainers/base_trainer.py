import os
import glob
import torch
import traceback
from lib.train.admin import multigpu
from torch.utils.data.distributed import DistributedSampler
import time
from softadapt import SoftAdapt, NormalizedSoftAdapt, LossWeightedSoftAdapt
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def create_exp_dir(path, visual_folder=False):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        if visual_folder is True:
            os.makedirs(path + '/visual', exist_ok=True)  # for visual results
    else:
        print("DIR already existed.")
    print('Experiment dir : {}'.format(path))


class BaseTrainer:
    """Base trainer class. Contains functions for training and saving/loading checkpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders

        self.update_settings(settings)

        self.epoch = 0
        self.stats = {}

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.actor.to(self.device)
        self.settings = settings
        self.best_model_indicator = float('inf')
        self.not_improved_epochs = 0
        self.early_stop = 20

        # define the adaptive weight loss function
        self.softadapt_object = LossWeightedSoftAdapt(beta=0.1)
        self.epochs_to_make_updates = 5

    def update_settings(self, settings=None):
        """Updates the trainer settings. Must be called to update internal settings."""
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
            '''2021.1.4 New function: specify checkpoint dir'''
            if self.settings.save_dir is None:
                self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            else:
                self._checkpoint_dir = os.path.join(self.settings.save_dir, 'checkpoints')
            print("checkpoints will be saved to %s" % self._checkpoint_dir)

            if self.settings.local_rank in [-1, 0]:
                if not os.path.exists(self._checkpoint_dir):
                    print("Training with multiple GPUs. checkpoints directory doesn't exist. "
                          "Create checkpoints directory")
                    os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None

    def train(self, max_epochs, load_latest=False, fail_safe=True, load_previous_ckpt=False, distill=False):
        """Do training for the given number of epochs.
        args:
            max_epochs - Max number of training epochs,
            load_latest - Bool indicating whether to resume from latest epoch.
            fail_safe - Bool indicating whether the training to automatically restart in case of any crashes.
        """
        path = '{}/{}/{}'.format(self._checkpoint_dir, self.settings.project_path, time.strftime("%Y%m%d-%H%M%S"))
        create_exp_dir(path, visual_folder=False)
        epoch = -1
        num_tries = 1
        for i in range(num_tries):
            try:
                checkpoint_path = 'ODTrack_ep0300.pth.tar'
                checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
                self.actor.net.load_state_dict(checkpoint_dict['net'], strict=False)

                giou_loss_list = []
                l1_loss_list = []
                focal_loss_list = []
                cls_loss_list = []

                for epoch in range(self.epoch + 1, max_epochs + 1):
                    self.epoch = epoch
                    loss_dict = self.train_epoch()
                    giou_loss_list.append(np.mean(loss_dict['train']["giou_loss"]))
                    l1_loss_list.append(np.mean(loss_dict['train']['l1_loss']))
                    focal_loss_list.append(np.mean(loss_dict['train']['focal_loss']))
                    cls_loss_list.append(np.mean(loss_dict['train']['cls_loss']))

                    if epoch % self.epochs_to_make_updates == 0 and epoch != 0:
                        adapt_weights = self.softadapt_object.get_component_weights(torch.tensor(giou_loss_list),
                                                                                    torch.tensor(l1_loss_list),
                                                                                    torch.tensor(focal_loss_list),
                                                                                    torch.tensor(cls_loss_list),
                                                                                    verbose=False,
                                                                                    )
                        print('Update loss weight:')

                        # Resetting the lists to start fresh (this part is optional)
                        self.actor.loss_weight = {'giou': adapt_weights[0].item(),
                                                  'l1': adapt_weights[1].item(),
                                                  'focal': adapt_weights[2].item(),
                                                  'cls': adapt_weights[3].item()}

                        print(self.actor.loss_weight)
                        giou_loss_list = []
                        l1_loss_list = []
                        focal_loss_list = []
                        cls_loss_list = []

                    if self.lr_scheduler is not None:
                        if self.settings.scheduler_type != 'cosine':
                            self.lr_scheduler.step()
                        else:
                            self.lr_scheduler.step(epoch - 1)
                    self.save_checkpoint(path)
                    if self.stats['val']['Loss/total'].history[-1] < self.best_model_indicator:
                        print(f'Model exceeds prev best score'
                              f'({self.stats["val"]["Loss/total"].history[-1]:.4f} < {self.best_model_indicator:.4f}). '
                              f'Saving it now.')
                        self.best_model_indicator = self.stats['val']['Loss/total'].history[-1]
                        # Save the model
                        self.save_checkpoint(path, best_model=True)
                        self.not_improved_epochs = 0  # reset counter
                    else:
                        if self.not_improved_epochs > self.early_stop:  # early stopping
                            print(f"Stopping training early since not improved for {self.early_stop} epochs.")
                            break
                        else:
                            self.not_improved_epochs = self.not_improved_epochs + 1
            except:
                print('Training crashed at epoch {}'.format(epoch))
                if fail_safe:
                    self.epoch -= 1
                    load_latest = True
                    print('Traceback for the error!')
                    print(traceback.format_exc())
                    print('Restarting training from last epoch ...')
                else:
                    raise

        print('Finished training!')

    def train_epoch(self):
        raise NotImplementedError

    def save_checkpoint(self, path, best_model=False):
        """Saves a checkpoint of the network and other variables."""

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': net.state_dict(),
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'settings': self.settings
        }

        # First save as a tmp file
        if not best_model:
            file_path = '{}/{}_last_model.pth.tar'.format(path, net_type, self.epoch)
        else:
            file_path = '{}/{}_best_model.pth.tar'.format(path, net_type)
        torch.save(state, file_path)

    def load_checkpoint(self, checkpoint=None, fields=None, ignore_fields=None, load_constructor=False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted(glob.glob('{}/{}/{}_ep*.pth.tar'.format(self._checkpoint_dir,
                                                                             self.settings.project_path, net_type)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_dir, self.settings.project_path,
                                                                 net_type, checkpoint)
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

            # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                net.load_state_dict(checkpoint_dict[key])
            elif key == 'optimizer':
                self.optimizer.load_state_dict(checkpoint_dict[key])
            else:
                setattr(self, key, checkpoint_dict[key])

        # Set the net info
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        # Update the epoch in lr scheduler
        if 'epoch' in fields:
            self.lr_scheduler.last_epoch = self.epoch
            # 2021.1.10 Update the epoch in data_samplers
            for loader in self.loaders:
                if isinstance(loader.sampler, DistributedSampler):
                    loader.sampler.set_epoch(self.epoch)
        return True

    def load_state_dict(self, checkpoint=None, distill=False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """
        if distill:
            net = self.actor.net_teacher.module if multigpu.is_multi_gpu(self.actor.net_teacher) \
                else self.actor.net_teacher
        else:
            net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        net_type = type(net).__name__

        if isinstance(checkpoint, str):
            # checkpoint is the path
            if os.path.isdir(checkpoint):
                checkpoint_list = sorted(glob.glob('{}/*_ep*.pth.tar'.format(checkpoint)))
                if checkpoint_list:
                    checkpoint_path = checkpoint_list[-1]
                else:
                    raise Exception('No checkpoint found')
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        print("Loading pretrained model from ", checkpoint_path)
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        missing_k, unexpected_k = net.load_state_dict(checkpoint_dict["net"], strict=False)
        print("previous checkpoint is loaded.")
        print("missing keys: ", missing_k)
        print("unexpected keys:", unexpected_k)

        return True
