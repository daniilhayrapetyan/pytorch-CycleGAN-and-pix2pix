import os.path

from data.base_dataset import BaseDataset, get_params
from data.image_folder import make_dataset
import torchvision.transforms as transforms
import glob

from PIL import Image


class TextureDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        assert opt.data_class_a is not None, "--data-class-a is required"
        assert opt.data_class_b is not None, "--data-class-b is required"

        self.data_dir_a = opt.data_class_a
        self.data_dir_b = opt.data_class_b

        self.data_paths_a = glob.glob(os.path.join(opt.data_class_a, '*'))
        self.data_paths_b = glob.glob(os.path.join(opt.data_class_b, '*'))

        # assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths
        """
        # read a image given a random integer index
        a_path = self.data_paths_a[index]
        b_path = self.data_paths_b[index]

        a_image = Image.open(a_path).convert('RGB')
        b_image = Image.open(b_path).convert('RGB')

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, a_image.size)

        a_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        b_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        a = a_transform(a_image)
        b = b_transform(b_image)

        return {'A': a, 'B': b, 'A_paths': a_path, 'B_paths': b_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return min(len(self.data_paths_a), len(self.data_paths_b))


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list.append(transforms.Lambda(lambda img: __crop(img)))

    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))

    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    return transforms.Compose(transform_list)


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __crop(img, texture_size=512):
    s = texture_size * 400 / 512
    return img.crop((texture_size/2 - s/2, texture_size - s, texture_size/2 + s/2, texture_size))
