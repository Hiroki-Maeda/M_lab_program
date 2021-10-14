import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0,100,100)
clist = np.random.rand(100)*10
clist = clist.astype(np.int8)
print(clist)

y = np.sin(x)
plt.figure()
color = ["#000000","#000044","#000088","#001100","#008800","#00dd55","#00dd88","#003399","#ff0033","#ff3300"]
for i in range(10):
	plt.scatter(x[clist==i],y[clist==i],c=color[i])
plt.show()
