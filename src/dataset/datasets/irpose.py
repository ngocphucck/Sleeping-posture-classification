from torch.utils.data import Dataset
import cv2

from src.utils.utils_io import read_json


class IRPoseDataset(Dataset):
    def __init__(self, transforms, json_path):
        super(IRPoseDataset, self).__init__()
        self.transforms = transforms
        self.data = list(read_json(json_path).items())

    def __getitem__(self, item):
        image_path, label = self.data[item]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = self.transforms(image)

        return image, label

    def __len__(self):

        return len(self.data)
