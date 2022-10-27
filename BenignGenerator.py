import os
import shutil

def recursive_walk(folder):
	for folderName, subfolders, filenames in os.walk(folder):
		if subfolders:
			for subfolder in subfolders:
				recursive_walk(subfolder)
		for filename in filenames:
			if filename.endswith('.exe'):
				shutil.copy(folderName + "/" + filename, dir_dst)

unallowed = ['desktop.ini','WindowsApps']
l = os.listdir("D:\\Apps\\")
dir_src = ("D:\\Apps\\")
dir_dst = ("D:\\Benign")
for i in l:
	if i in unallowed:
		continue
	print('D:\\Apps\\' + i)
	recursive_walk('D:\\Apps\\' + i)
