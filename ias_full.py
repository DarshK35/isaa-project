import numpy as np
from PIL import Image
import sys
import glob
import errno
import os

result=[]
temp1=np.zeros((256,256))
image = np.zeros((256,256))
i,j,p=0,0,0

path = "D:\\Benign\\" ## Absolute path for the database.
dirs = os.listdir(path) ## gets all the directories in the given folder.
path2 = "D:\\Darsh\\Work\\VIT\\Sem 5 - Fall sem 2022-23\\CSE3501 ISAA\\Project\\Dataset256\\" ## Absolute path for storing 256x256 image.
path3 = "D:\\Darsh\\Work\\VIT\\Sem 5 - Fall sem 2022-23\\CSE3501 ISAA\\Project\\Dataset32\\"  ## Absolute path for storing 36x36 image.

for fil in dirs:
	image = np.zeros((256, 256))
	with open(path + fil, "rb") as f: ## read byte mode
		byte = f.read(1)  ## reading 1 byte at a time. returns a byte object.
		i,j,p=0,0,0
		print(fil)
		print(byte)
		while byte != b"": ## itereating till the end of the file
			#print (ord(byte),byte) ## ord gives ASCII value of that char.
			if(i<256):
				#if(byte.decode("utf-8")==' ' or byte.decode("utf-8")=='\r' or byte.decode("utf-8")=='\n' or byte.decode("utf-8")=='?' ):
				#	byte = f.read(1)
				#	continue
				# image[i][p]=0
				#print (byte.decode("utf-8"),int(byte.decode("utf-8"),16),i)
				if(j>8):
					image[i][p]= int.from_bytes(byte, byteorder='little') ## byte.decode gives the value of the byte object which is converted into hexa-decimal.
					p=p+1
				j=j+1
				if(p>255):
					p=0
					j=0
					i=i+1
			byte = f.read(1)
		a=np.matrix(image)
		# a=np.matrix(image)
		print (a)
		result.append(np.array_equal(temp1,a))
		temp1=a
		img = Image.fromarray(image, 'L') ## makes 2D array into an image. L-> The kind of pixel (8-bit pixel black and white)
		#img.show()
		img.save(path2+fil[0:-3] + ".jpg")
		img = img.resize((36,36),Image.ANTIALIAS)
		img.save(path3+fil[0:-3] + ".jpg")
		#img.show()
print (result)
