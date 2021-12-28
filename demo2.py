import numpy as np

aaa = np.array([1,2,4,5,6,4,5,6])
for i in range(3):
	print("i : {}".format(i))
	for j in range(aaa.shape[0]):
		print("j : {}".format(j))
		if aaa[j]>3:
			break



