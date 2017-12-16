import numpy as np
import glob
import code_utils as cu
from PIL import Image


class ImageUtils():
    def __init__(self):
        self.test_data = glob.glob('test-images/*.jpeg')
        self.test_label = np.array(
            [cu.in_transition(self.test_data[index].split('/')[-1].split('.')[0].split('_')[0]) for index in
             range(len(self.test_data))])

        self.train_data = glob.glob('train-images/*.jpeg')
        self.train_label = np.array(
            [cu.in_transition(self.train_data[index].split('/')[-1].split('.')[0].split('_')[0]) for index in
             range(len(self.train_data))])

    @staticmethod
    def sample(capacity, batch_size, datas, labels):
        sample_index = np.random.choice(capacity, batch_size)
        _datas = np.array([np.array(Image.open(datas[index]).convert("1"))[:, :, np.newaxis] for index in sample_index])
        _labels = labels[sample_index, :]
        return _datas, _labels

    @staticmethod
    def trainstion_data(datas, sample_index=None, start=0, end=1000):
        if sample_index != None:
            return np.array(
                [np.array(Image.open(datas[index]).convert("1"))[:, :, np.newaxis] for index in sample_index])
        return np.array(
            [np.array(Image.open(datas[index]).convert("1"))[:, :, np.newaxis] for index in range(start, end)])
