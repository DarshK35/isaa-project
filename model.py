from fastai.vision.all import *

data = ImageDataLoaders.from_folder("..\\Dataset\\", "Train", "Validation", classes = ["Malware", "Benign"])

for d in data.loaders:
	print(d)