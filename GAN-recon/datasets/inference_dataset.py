import os

from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils


class InferenceDataset:
	def __init__(self, root, transform=None, preprocess=None, opts=None):
		self.root = root
		self.transform = transform
		self.preprocess = preprocess
		self.img_paths = []  # 初始化一个空列表用于存储图像路径

		# 加载图像路径
		for img_name in os.listdir(root):
			self.img_paths.append(os.path.join(root, img_name))

	# 加载图像或其他初始化

	def __len__(self):
		return len(self.img_paths)

	def __getitem__(self, index):
		from_path = self.img_paths[index]
		if self.preprocess is not None:
			from_im = self.preprocess(from_path)
		else:
			from_im = Image.open(from_path).convert('RGB')
		if self.transform:
			from_im = self.transform(from_im)
		return from_im
