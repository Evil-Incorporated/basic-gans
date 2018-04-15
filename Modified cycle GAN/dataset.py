import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='train'):
        if transforms_ is not None:
            self.transform = transforms.Compose(transforms_)
        else:
            self.transform = None
        self.unaligned = unaligned

        # print(os.path.join(root, '%sA' % mode) + '/*.*')
        # print(sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*')))

        self.files_A = sorted(glob.glob(os.path.join(root, '%sA' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%sB' % mode) + '/*.*'))

    def __getitem__(self, index):
        if self.transform is not None:
            item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]))
        else:
            item_A = Image.open(self.files_A[index % len(self.files_A)])

        if self.unaligned:
            if self.transform is not None:
                item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]))
            else:
                item_B = Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)])
        else:
            if self.transform is not None:
                item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]))
            else:
                item_B = Image.open(self.files_B[index % len(self.files_B)])

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

# img_dataset = ImageDataset('./sketch', [transforms.Grayscale(), transforms.Resize(int(128 * 1.12), Image.BICUBIC),
#                                         transforms.CenterCrop(128)])
# img_dataset[1]['A'].show()
#img_dataset[1]['A'].show()
#img_dataset[1]['B'].show()
