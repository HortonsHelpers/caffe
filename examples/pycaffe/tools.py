import numpy as np


class SimpleTransformer:

    """
    SimpleTransformer is a simple class for preprocessing and deprocessing
    images for caffe.
    """

    def __init__(self, mean=[128, 128, 128]):
        self.mean = np.array(mean, dtype=np.float32)
        self.scale = 1.0

    def set_mean(self, mean):
        """
        Set the mean to subtract for centering the data.
        """
        self.mean = mean

    def set_scale(self, scale):
        """
        Set the data scaling.
        """
        self.scale = scale

    def preprocess(self, im):
        """
        preprocess() emulate the pre-processing occurring in the vgg16 caffe
        prototxt.
        """

        im = np.float32(im)
        im = im[:, :, ::-1]  # change to BGR
        im -= self.mean
        im *= self.scale
        return im.transpose((2, 0, 1))

    def deprocess(self, im):
        """
        inverse of preprocess()
        """
        im = im.transpose(1, 2, 0)
        im /= self.scale
        im += self.mean
        im = im[:, :, ::-1]  # change to RGB

        return np.uint8(im)


class CaffeSolver:

    """
    Caffesolver is a class for creating a solver.prototxt file. It sets default
    values and can export a solver parameter file.
    Note that all parameters are stored as strings. Strings variables are
    stored as strings in strings.
    """

    def __init__(self, testnet_prototxt_path="testnet.prototxt",
                 trainnet_prototxt_path="trainnet.prototxt", debug=False):

        self.sp = {
            'base_lr': '0.001',
            'momentum': '0.9',
            'test_iter': '100',
            'test_interval': '250',
            'display': '25',
            'snapshot': '2500',
            'snapshot_prefix': '"snapshot"',
            'lr_policy': '"fixed"',
            'gamma': '0.1',
            'weight_decay': '0.0005',
            'train_net': f'"{trainnet_prototxt_path}"',
            'test_net': f'"{testnet_prototxt_path}"',
            'max_iter': '100000',
            'test_initialization': 'false',
            'average_loss': '25',
            'iter_size': '1',
        }

        if (debug):
            self.sp['max_iter'] = '12'
            self.sp['test_iter'] = '1'
            self.sp['test_interval'] = '4'
            self.sp['display'] = '1'

    def add_from_file(self, filepath):
        """
        Reads a caffe solver prototxt file and updates the Caffesolver
        instance parameters.
        """
        with open(filepath, 'r') as f:
            for line in f:
                if line[0] == '#':
                    continue
                splitLine = line.split(':')
                self.sp[splitLine[0].strip()] = splitLine[1].strip()

    def write(self, filepath):
        """
        Export solver parameters to INPUT "filepath". Sorted alphabetically.
        """
        f = open(filepath, 'w')
        for key, value in sorted(self.sp.items()):
            if type(value) is not str:
                raise TypeError('All solver parameters must be strings')
            f.write('%s: %s\n' % (key, value))
