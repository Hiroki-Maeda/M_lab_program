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
search_len = 800
x = np.linspace(0,search_len ,search_len ) 

for k in range(len(ch_names)):

	base_data = np.zeros((1,search_len ))
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
		#ajust time
		erp_data = np.zeros((1,search_len))						
		ax = fig.add_subplot(3,2,j+1)
		ax.set_ylim(-200,200)		
		mean_min_error_timepoint = 0
		for i in range(data.shape[0]):
			min_error = 1.0e+15
			min_error_timepoint = 0
			if np.max(abs(filted_data[i,50:-50]))<100:

				for time_i in range(50):
					temp_data = base_data-filted_data[i,50+time_i:50+time_i+search_len ]
					if min_error > np.sum(abs(temp_data)):
						min_error =np.sum(abs(temp_data))
						min_error_timepoint = time_i						

				mean_min_error_timepoint += min_error_timepoint
				base_data += filted_data[i,50+min_error_timepoint:50+min_error_timepoint+search_len ]
				erp_data +=  filted_data[i,50+min_error_timepoint:50+min_error_timepoint+search_len ]
				#base_data = (base_data-base_data.min())/(base_data.max()-base_data.min()+1)*(200+200)-200
				#erp_data = (erp_data-erp_data.min())/(erp_data.max()-erp_data.min()+1)*(200+200)-200
			


				#print("min_error_timepoint : ",min_error_timepoint)
				#a = input()
				ax.plot(x,filted_data[i,50+min_error_timepoint:50+min_error_timepoint+search_len ])

		mean_min_error_timepoint = mean_min_error_timepoint/data.shape[0]
		ax.set_title(words[j]+" : mean tp= "+str(round(mean_min_error_timepoint,1)))
		ax = fig.add_subplot(3,2,6)
		erp_data = (erp_data-erp_data.min())/(erp_data.max()-erp_data.min()+1)*(200+200)-200
		ax.plot(x,erp_data[0,:],label=words[j]+":"+str(round(mean_min_error_timepoint,1)))
		

				
		#sum data

		#sum_data = np.zeros(filted_data[0,:].shape)
		"""
		for i in range(filted_data.shape[0]):
			if np.max(abs(filted_data[i,50:-50]))<100:
					sum_data+= filted_data[i]
		"""
		#filted_data = np.clip(filted_data , -100,75)

		"""	
		ax = fig.add_subplot(3,2,j+1)
		ax.set_ylim(-200,200)		
		for i in range(data.shape[0]):
			#plt.figure()
			#plt.plot(x,np.sum(filted_data,axis=0))
			if np.max(abs(filted_data[i,50:-50]))<100:
				ax.plot(x,filted_data[i,:])
							
			#plt.plot(x,data[i,:])
			#plt.show()
		"""
		fig.suptitle(ch_names[k])
	ax = fig.add_subplot(3,2,6)
	base_data = (base_data-base_data.min())/(base_data.max()-base_data.min()+1)*(200+200)-200
	ax.plot(x,base_data[0,:],label="base_data :"+str(round(mean_min_error_timepoint,1)))
	ax.legend()
	plt.subplots_adjust(wspace=0.4,hspace=0.6)
	#fig.savefig(os.path.join(Save_Path,"div_all_data_all_adjust"+ch_names[k]+".png"))
	
	plt.show()	
		
"""		
for k in range(len(ch_names)):
	axs[k].legend()
	axs[k].set_title(ch_names[k])		
fig.savefig(os.path.join(Save_Path,"vec_data_erp.png"))
plt.show()
"""
