import numpy as np
import pandas as pd 
import os
from scipy import signal
import matplotlib.pyplot as plt


def lowpass(x, samplerate, fp, fs, gpass, gstop):
	fn = samplerate/2
	wp = fp/fn
	ws = fs/fn
	N,Wn = signal.buttord(wp,ws,gpass,gstop)
	b,a = signal.butter(N,Wn,"low")
	y = signal.filtfilt(b,a,x)
	return y

Data_Path = os.path.join("/","mnt","c","Users","HirokiMaeda","Desktop","M1","1_word_HMM","data","vec_data")
Save_Path =  os.path.join("/","mnt","c","Users","HirokiMaeda","Desktop","M1","1_word_HMM","picture")

words = ["a","i","u","e","o"]
#ch_names = [" F7-REF"," T7-REF"," Cz-REF"]
ch_names = ["_F7-T7","_T7-Cz","_Cz-F7"]
fig_flag = 0
for k in range(len(ch_names)):
	fig= plt.figure(figsize=(20,10))	
	for j in range(len(words)):

		
		data = pd.read_csv(os.path.join(Data_Path,"data_"  + words[j] +ch_names[k] +".csv")).values
		#Data_Path = os.path.join("/","mnt","c","Users","HirokiMaeda","Desktop","M1","1_word_HMM","data","vec_data")
		#data = pd.read_csv(os.path.join(Data_Path,"data_a_Cz-F7.csv")).values

		print(data.shape)

		samplerate = 1000
		fp = 30
		fs = 50

		gpass = 3
		gstop = 40
		filted_data = np.empty((0,data.shape[1]),float)

		for i in range(data.shape[0]):
			filted_data = np.vstack([filted_data,lowpass(data[i,:],samplerate,fp,fs,gpass,gstop)])	
			#print("filted "+str(i)+" sample")
		x = np.linspace(0,filted_data.shape[1]-100,filted_data.shape[1]-100) 


		#sum data

		sum_data = np.zeros(filted_data[0,50:-50].shape)
		"""
		for i in range(filted_data.shape[0]):
			if np.max(abs(filted_data[i,50:-50]))<100:
					sum_data+= filted_data[i]
		"""
		#filted_data = np.clip(filted_data , -100,75)

	
		ax = fig.add_subplot(3,2,j+1)
		ax.set_ylim(-200,200)		
		for i in range(data.shape[0]):
			#plt.figure()
			#plt.plot(x,np.sum(filted_data,axis=0))
			if np.max(abs(filted_data[i,50:-50]))<100:
				ax.plot(x,filted_data[i,50:-50])
				sum_data += filted_data[i,50:-50]
							
			#plt.plot(x,data[i,:])
			#plt.show()
		ax.set_title(words[j])
		ax = fig.add_subplot(3,2,6)
		ax.plot(x,sum_data,label=words[j])	
		

	fig.suptitle(ch_names[k])
	ax.legend()
	fig.savefig(os.path.join(Save_Path,"vec_data_not_sum_"+ch_names[k]+".png"))
		
	plt.show()	
		
"""		
for k in range(len(ch_names)):
	axs[k].legend()
	axs[k].set_title(ch_names[k])		
fig.savefig(os.path.join(Save_Path,"vec_data_erp.png"))
plt.show()
"""
