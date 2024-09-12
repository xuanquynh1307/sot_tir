import os
import os.path
import numpy as np
import torch
import csv
import pandas
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class LSOTB(BaseVideoDataset):
    def __init__(self, root=None, image_loader=jpeg4py_loader, split='train'):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'val'
        """
        root = env_settings().lsotb_dir if root is None else root
        root = root + '/' + split
        super().__init__('LSOTB', root, image_loader)
        # all folders inside the root
        self.sequence_list = self._get_sequence_list()
        # get order of sequences
        if split is not None:
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'lsotb_train.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'data_specs', 'lsotb_val.txt')
            else:
                raise ValueError('Unknown split name.')
            seq_ids = pandas.read_csv(file_path, header=None, dtype=np.int64).squeeze('columns').values.tolist()

        else:
            seq_ids = list(range(0, len(self.sequence_list)))
        self.sequence_list = [self.sequence_list[i] for i in seq_ids]
        self.sequence_meta_info = self._load_meta_info()

    def get_name(self):
        return 'lsotb'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return True

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def _read_meta(self, seq_path):
        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})
        return object_meta

    def _get_sequence_list(self):
        with open(os.path.join(self.root, 'list.txt')) as f:
            dir_list = list(csv.reader(f))
        dir_list = [dir_name[0] for dir_name in dir_list]
        return dir_list

    def _read_bb_anno(self, seq_path):
        bb_anno_file = os.path.join(seq_path, "groundtruth_rect.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion
        occlusion_file = os.path.join(seq_path, "occlusion.txt")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        target_visible = ~occlusion

        return target_visible

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, 'seq', self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = self._read_target_visible(seq_path)

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, 'img', '{:08}.jpg'.format(frame_id+1))    # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]

        return frame_list, anno_frames, obj_meta
