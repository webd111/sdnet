# From original author
import json
import os.path

# built-in
import random

import cv2

# 3rd party
import numpy as np
import scipy.io as sio
import torch
import torch.utils.data
from datasets.utils import np_flip, np_rotate
from tqdm import tqdm

from torchvision.datasets.utils import download_url


class HPatches(torch.utils.data.Dataset):
    urls = {
        "data": [
            "http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-release.tar.gz",
            "hpatches-release.tar.gz",
            "0ab830d37fceb2b4c86cb1cc6cc79a61",
        ],
        "splits": [
            "https://raw.githubusercontent.com/hpatches/hpatches-benchmark/master/tasks/splits/splits.json",
            "splits.json",
        ],
    }
    patch_types = [
        "ref",
        "e1",
        "e2",
        "e3",
        "e4",
        "e5",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "t1",
        "t2",
        "t3",
        "t4",
        "t5",
    ]
    mean = {"full": 0.421789139509}
    std = {"full": 0.226070001721}
    length = {"full": 2211472}

    def __init__(self, root, num_types=16, transform=None, train=True, output_index=False,
                 data_aug=False, download=False, split_name="a"):
        super().__init__()
        self.root = root
        self.split_file = os.path.join(self.root, self.urls["splits"][1])
        self.data_dir = os.path.join(self.root, "hpatches-release")
        self.split_name = split_name
        self.num_types = min(num_types, 16)
        self.transform = transform
        self.train = train
        self.data_aug = data_aug
        self._output_index = output_index

        if self.train:
            self.data_file = os.path.join(self.root, "data_" + split_name + "_train.pt")
        else:
            self.data_file = os.path.join(self.root, "data_" + split_name + "_test.pt")

        # if os.path.exists(self.data_file):
        #     self._read_image_file()

        if download:
            self.download()

        if not self._check_datafile_exists():
            raise RuntimeError("Dataset not found." + " You can use download=True to download it")

        # load the serialized data
        checkpoint = torch.load(self.data_file)
        self.data = checkpoint["data"]  # patch_type * label * patch_size * patch_size, 16 * 59532 * 65 * 65
        self.labels = checkpoint["labels"]  # label * patch_type, 952512 = 59532 * 16
        self.data_sequence_name = checkpoint["sequence_name"]
        self.data_point_index = checkpoint["point_index"]
        self.data_patch_type = checkpoint["patch_type"]
        self.sequence_len = checkpoint["sequence_len"]

    # Use hpatches_eval.py to test. Descriptors are stored in following forms
    # desc: dict {"sequence names": hpatches_descr} {"distance": l2} {"dim": 128}
    # hpatches_descr:   N: number
    #                   dim: dimension
    #                   name: sequence name
    #                   itr: split list
    #                   e1 - t5, ref: N * dim descriptors
    def __getitem__(self, index):
        # TODO: split easy, hard and tough examples
        if self.train:
            p_idx1 = index
            t_idx1 = random.randint(0, self.num_types - 1)  # 0 - 15
            patch_a = self.data[t_idx1, p_idx1, ...]

            t_idx2 = t_idx1
            while t_idx2 == t_idx1:
                t_idx2 = random.randint(0, self.num_types - 1)
            patch_p = self.data[t_idx2, p_idx1, ...]

            idx_rot = 0
            idx_flip = 0
            if self.data_aug:
                idx_rot = random.randint(0, 3)
                idx_flip = random.randint(0, 2)
                patch_a, patch_p = np_rotate([patch_a, patch_p], idx_rot)
                patch_a, patch_p = np_flip([patch_a, patch_p], idx_flip)

            if self.transform is not None:
                patch_a = self.transform(patch_a)
                patch_p = self.transform(patch_p)

            if self._output_index:
                return patch_a, patch_p, p_idx1, t_idx1, t_idx2, (idx_rot, idx_flip)
            else:
                return patch_a, patch_p
        else:
            # load all types of one patch
            patches_list = []
            sequence_name = self.data_sequence_name[index]
            point_label = self.data_point_index[index]

            for i in range(16):
                patch = self.data[i, index, ...]
                if self.transform is not None:
                    patch = self.transform(patch)
                patches_list.append(patch)
            patches = torch.cat(patches_list, dim=0)

            return patches, sequence_name, point_label

    def __len__(self):
        return self.data.shape[1]

    def _read_image_file(self):
        # sequence, idx, patch_type
        with open(self.split_file) as f:
            splits = json.load(f)
        if self.train:
            sequences = splits[self.split_name]["train"]
        else:
            sequences = splits[self.split_name]["test"]

        all_patches = []
        sequence_name = []
        point_label = []
        patch_type = []
        sequence_len = {}
        for t in tqdm(self.patch_types):
            for seq in tqdm(sequences, disable=True):
                im_p = os.path.join(self.data_dir, seq, t + ".png")
                im = cv2.imread(im_p, 0)
                N = im.shape[0] / 65
                patches = np.split(im, N)
                all_patches += patches
                for i in np.arange(N):
                    sequence_name.append(seq)
                    patch_type.append(t)
                point_label += np.arange(N).astype(np.int32).tolist()
                sequence_len[seq] = int(N)

        # labels
        labels = np.arange(len(all_patches) / len(self.patch_types))
        labels = np.tile(labels, len(self.patch_types))

        return np.array(all_patches).reshape(len(self.patch_types), -1, 65, 65), labels, sequence_name, \
               np.array(point_label), patch_type, sequence_len

    def _check_datafile_exists(self):
        return os.path.exists(self.data_file)

    def _check_downloaded(self):
        return os.path.exists(self.data_dir)

    def download(self):
        if self._check_datafile_exists():
            print("# Found cached data {}".format(self.data_file))
            return

        if not self._check_downloaded():
            # download raw data
            fpath = os.path.join(self.root, self.urls["data"][1])
            download_url(self.urls["data"][0], self.root, filename=self.urls["data"][1], md5=self.urls["data"][2], )

            print("# Extracting data {}\n".format(fpath))

            import tarfile

            with tarfile.open(fpath, "r:gz") as t:
                t.extractall()

            os.unlink(fpath)

            # download splits.json
            download_url(self.urls["splits"][0], self.root, filename=self.urls["splits"][1])

        # process and save as torch files
        print("# Caching data {}".format(self.data_file))

        dataset, labels, sequence_name, point_index, patch_type, sequence_len = self._read_image_file()

        with open(self.data_file, "wb") as f:
            torch.save({"data": dataset, "labels": labels, "sequence_name": sequence_name,
                        "point_index": point_index, "patch_type": patch_type, "sequence_len": sequence_len}, f, pickle_protocol=4)


if __name__ == "__main__":
    dset = HPatches(root="./data/hpatches", download=True)
    print(len(dset))
