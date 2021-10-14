
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np

#Data_Path = os.path.join("/","mnt","c","Users","HirokiMaeda","Desktop","M1","1_word_HMM","data","vec_data")
Data_Path = os.path.join("/","mnt","c","Users","HirokiMaeda","Desktop","M1","1_word_HMM","data","div_data")


Save_Path =  os.path.join("/","mnt","c","Users","HirokiMaeda","Desktop","M1","1_word_HMM","picture")

words = ["a","i","u","e","o"]
#ch_names = [" F7-REF"," T7-REF"," Cz-REF",]
#ch_names = ["_F7-T7","_T7-Cz","_Cz-F7"]
#ch_names = ["F7-REF","T7-REF","P7-REF","O1-REF","O2-REF","P8-REF","T8-REF","F8-REF","Fp2-REF","Fp1-REF","F3-REF","C5-REF","P3-REF","P4-REF","C6-REF","F4-REF","Fz-REF","Cz-REF","Pz-REF"]
ch_names = [" F7-REF"," T7-REF"," P7-REF"," O1-REF"," O2-REF"," P8-REF"," T8-REF"," F8-REF"," Fp2-REF"," Fp1-REF"," F3-REF"," C5-REF"," P3-REF"," P4-REF"," C6-REF"," F4-REF"," Fz-REF"," Cz-REF"," Pz-REF"]


b_num = len(ch_names)

print("read start")
for j in range(len(words)):
	temp_sum_data = np.empty((0,one_data_len))
	for k in range(b_num):
		temp_data = [[] for i in range(one_state_num+1)]

		
		data = pd.read_csv(os.path.join(Data_Path,"data_"  + words[j] +ch_names[k] +".csv")).values
		#Data_Path = os.path.join("/","mnt","c","Users","HirokiMaeda","Desktop","M1","1_word_HMM","data","vec_data")
		#data = pd.read_csv(os.path.join(Data_Path,"data_a_Cz-F7.csv")).values

		data = (data-data.mean(axis=1).reshape(-1,1))/data.std(axis=1).reshape(-1,1)
		(data_num, data_length) = filted_data.shape
				
		
		if j == 0 and k == 0:
			all_data = np.zeros((word_num*b_num,data_num,data_length))
		####
		all_data[j*b_num+k] = data	
		#print(temp_sum_data.shape)

print("read end")	
correlate_mat = np.zeros((b_num*(b_num-1)/2,data_length*2))
lags = signal.correlation_lags(len(all_data[0,0]),len(all_data[0,0]))

print("calculate correlation")
for l in range(len(word)):
	correlate_mat = np.zeros((b_num*(b_num-1)/2,data_length*2))

	for i in range(data_num):
		count = 0
		for j in range(len(ch_names)-1):
			for k in range(len(ch_names[j+1;])-1):
				correlate_mat[count] += signal.correlate(all_data[l*b_num+j,i],all_data[l*b_num+j+1+k,i])
				
				count+=1


	correlate_mat = correlate_mat/data_num

	fig = plt.figure()
	plot_num = correlate_mat.shape[0]
	for i in range(plot_num):
		ax = fig.add_subplot(plot_num/2,2,i)
		ax.plot(lags,correlate_mat[i])

	fig.show()
