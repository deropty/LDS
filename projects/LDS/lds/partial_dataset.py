# encoding: utf-8

"""
@author:  lingxiao he
@contact: helingxiao3@jd.com
"""

import glob
import os
import os.path as osp
import re

from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.datasets.bases import ImageDataset

__all__ = ['PartialREID', 'PartialiLIDS', 'OccludedREID', 'P_DukeMTMC_ReID', 'P_ETHZ', 'Occluded_DukeMTMC']


def process_test(query_path, gallery_path):
    query_img_paths = glob.glob(os.path.join(query_path, '*.jpg'))
    gallery_img_paths = glob.glob(os.path.join(gallery_path, '*.jpg'))
    query_paths = []
    pattern = re.compile(r'([-\d]+)_(\d*)')
    for img_path in query_img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        query_paths.append([img_path, pid, camid])
    gallery_paths = []
    for img_path in gallery_img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        gallery_paths.append([img_path, pid, camid])
    return query_paths, gallery_paths

def process_dir(dir_path, relabel=False, is_query=True):
    types = ('*.jpg', '*.png')
    img_paths = []
    for i in types:
        img_paths.extend(glob.glob(osp.join(dir_path, '*', i)))
    if is_query:
        camid = 0
    else:
        camid = 1
    pid_container = set()
    for img_path in img_paths:
        img_name = img_path.split('/')[-1]
        pid = int(img_name.split('_')[0])
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}

    data = []
    for img_path in img_paths:
        img_name = img_path.split('/')[-1]
        pid = int(img_name.split('_')[0])
        if relabel:
            pid = pid2label[pid]
        data.append((img_path, pid, camid))
    return data

def process_train_dir(dir_path, relabel=True):
    img_paths = glob.glob(osp.join(dir_path,'whole_body_images','*','*.jpg'))
    camid=1
    pid_container = set()
    for img_path in img_paths:
        img_name = img_path.split('/')[-1]
        pid = int(img_name.split('_')[0])
        pid_container.add(pid)
    pid2label = {pid:label for label, pid in enumerate(pid_container)}
    data = []
    for img_path in img_paths:
        img_name = img_path.split('/')[-1]
        pid = int(img_name.split('_')[0])
        if relabel:
            pid = pid2label[pid]
        data.append((img_path, pid, camid))
    img_paths = glob.glob(osp.join(dir_path,'occluded_body_images','*','*.jpg'))
    camid=0
    for img_path in img_paths:
        img_name = img_path.split('/')[-1]
        pid = int(img_name.split('_')[0])
        if relabel:
            pid = pid2label[pid]
        data.append((img_path, pid, camid))
    return data

@DATASET_REGISTRY.register()
class Occluded_DukeMTMC(ImageDataset):

    dataset_name = "occludeddukemtmc"

    def __init__(self, root='datasets',):
        self.root = root

        self.dataset_dir = osp.join(self.root, 'Occluded_Duke')
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self.list_train_path = osp.join(self.root, 'Occluded_Duke/train.list')
        self.list_query_path = osp.join(self.root, 'Occluded_Duke/query.list')
        self.list_gallery_path = osp.join(self.root, 'Occluded_Duke/gallery.list')

        train = self.process_dir(self.train_dir, self.list_train_path)
        query = self.process_dir(self.query_dir, self.list_query_path, is_train=False)
        gallery = self.process_dir(self.gallery_dir, self.list_gallery_path, is_train=False)

        ImageDataset.__init__(self, train, query, gallery)

    def process_dir(self, dir_path, list_path, is_train=True):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        data = []

        for img_idx, img_info in enumerate(lines):
            pid, camid, _ = img_info.split('_')
            pid = int(pid)  # no need to relabel
            camid = int(camid[1]) - 1  # index starts from 0
            img_path = osp.join(dir_path, img_info[:-1])
            if is_train:
                pid = 'occluded' + "_" + str(pid)
                camid = 'occluded' + "_" + str(camid)
            data.append((img_path, pid, camid))

        return data

@DATASET_REGISTRY.register()
class PartialREID(ImageDataset):

    dataset_name = "partialreid"

    def __init__(self, root='datasets',):
        self.root = root

        self.query_dir = osp.join(self.root, 'Partial_REID/partial_body_images')
        self.gallery_dir = osp.join(self.root, 'Partial_REID/whole_body_images')
        query, gallery = process_test(self.query_dir, self.gallery_dir)

        ImageDataset.__init__(self, [], query, gallery)


@DATASET_REGISTRY.register()
class PartialiLIDS(ImageDataset):
    dataset_name = "partialilids"

    def __init__(self, root='datasets',):
        self.root = root

        self.query_dir = osp.join(self.root, 'PartialiLIDS/query')
        self.gallery_dir = osp.join(self.root, 'PartialiLIDS/gallery')
        query, gallery = process_test(self.query_dir, self.gallery_dir)

        ImageDataset.__init__(self, [], query, gallery)


@DATASET_REGISTRY.register()
class OccludedREID(ImageDataset):
    dataset_name = "occludereid"

    def __init__(self, root='datasets',):
        self.root = root

        self.query_dir = osp.join(self.root, 'OccludedREID/query')
        self.gallery_dir = osp.join(self.root, 'OccludedREID/gallery')
        query, gallery = process_test(self.query_dir, self.gallery_dir)

        ImageDataset.__init__(self, [], query, gallery)

@DATASET_REGISTRY.register()
class P_DukeMTMC_ReID(ImageDataset):
    dataset_name = "pdukemtmcreid"

    def __init__(self, root='datasets', **kwargs):
        self.root = root

        self.train_dir = osp.join(self.root, 'P-DukeMTMC-reid/train')
        self.query_dir = osp.join(self.root, 'P-DukeMTMC-reid/test/occluded_body_images')
        self.gallery_dir = osp.join(self.root, 'P-DukeMTMC-reid/test/whole_body_images')
        train = process_train_dir(self.train_dir, relabel=True)
        query = process_dir(self.query_dir, relabel=False, is_query=True)
        gallery = process_dir(self.gallery_dir, relabel=False, is_query=False)

        ImageDataset.__init__(self, train, query, gallery)

@DATASET_REGISTRY.register()
class P_ETHZ(ImageDataset):
    dataset_name = "pethz"

    def __init__(self, root='datasets',):
        self.root = root

        self.query_dir = osp.join(self.root, 'P_ETHZ/occluded_body_images')
        self.gallery_dir = osp.join(self.root, 'P_ETHZ/whole_body_images')
        query = process_dir(self.query_dir, relabel=False, is_query=True)
        gallery = process_dir(self.gallery_dir, relabel=False, is_query=False)

        ImageDataset.__init__(self, [], query, gallery)