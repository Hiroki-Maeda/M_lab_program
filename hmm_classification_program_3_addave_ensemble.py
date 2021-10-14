#all_filted_data : not hmm train data



import numpy as np
import pandas
import os, sys
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
from scipy import signal
import collections
import random
import statistics


def lowpass(x, samplerate, fp, fs, gpass, gstop):
	fn = samplerate/2
	wp = fp/fn
	ws = fs/fn
	N,Wn = signal.buttord(wp,ws,gpass,gstop)
	b,a = signal.butter(N,Wn,"low")
	y = signal.filtfilt(b,a,x)
	return y

Data_Path = os.path.join("/","mnt","c","Users","HirokiMaeda","Desktop","M1","1_word_HMM","data","vec_data")
#Data_Path = os.path.join("/","mnt","c","Users","HirokiMaeda","Desktop","M1","1_word_HMM","data","div_data")


Save_Path =  os.path.join("/","mnt","c","Users","HirokiMaeda","Desktop","M1","1_word_HMM","picture")

color = ["#000000","#44ffff","#88ffff","#bbffff","#eeffff","#ff44ff","#ff88ff","#ffbbff","#ffeeff","#ffff44","#ffff88","#ffffbb","#ffffee","#444444","#888888","#bbbbbb","#eeeeee","#44ff44","#88ff88","#bbffbb","#eeffee"]
words = ["a","i","u","e","o"]
#ch_names = [" F7-REF"," T7-REF"," Cz-REF"]
ch_names = ["_F7-T7","_T7-Cz","_Cz-F7"]
fig_flag = 0
b_num = len(ch_names)
#b_num = 1
word_num = len(words)
all_data = np.empty((b_num,0))
one_data_len= 924
one_state_num = 4
one_state_len = round(one_data_len/one_state_num)
state_num = 1+len(words)*one_state_num
data_div_state =[[[[] for i in range(one_state_num)] for j in range(b_num)] for k in range(word_num)]
data_div_state_0 = [[]for i in range(b_num)]
sum_data = np.empty((b_num,0))
set_init = True
hmm_trainnum_rate = 0.3
#classification_modelnum_rate = 0.4
test_num_rate = 0.7
Restrict = 1
test_add_num = 10
ensemble_num = 7

all_filted_data = []
print("read start")
for j in range(len(words)):
	temp_sum_data = np.empty((0,one_data_len))
	for k in range(b_num):
		temp_data = [[] for i in range(one_state_num+1)]

		
		data = pd.read_csv(os.path.join(Data_Path,"data_"  + words[j] +ch_names[k] +".csv")).values
		#Data_Path = os.path.join("/","mnt","c","Users","HirokiMaeda","Desktop","M1","1_word_HMM","data","vec_data")
		#data = pd.read_csv(os.path.join(Data_Path,"data_a_Cz-F7.csv")).values
		samplerate = 1000
		fp = 30
		fs = 50

		gpass = 3
		gstop = 40
		filted_data = np.empty((0,data.shape[1]-100),float)
		for i in range(data.shape[0]):
			filted_data = np.vstack([filted_data,lowpass(data[i,:],samplerate,fp,fs,gpass,gstop)[50:-50]])	
			#print("filted "+str(i)+" sample")


		filted_data = (filted_data-filted_data.mean(axis=1).reshape(-1,1))/filted_data.std(axis=1).reshape(-1,1)
		(data_num, data_length) = filted_data.shape
				
		
		if j == 0 and k == 0:
			all_filted_data = np.zeros((word_num*b_num,data_num,data_length))
		####
		all_filted_data[j*b_num+k] = filted_data	
		#print(temp_sum_data.shape)
print("read end")	
train_num = round(data_num*hmm_trainnum_rate)
test_num = round(data_num*test_num_rate)
train_shuffle = random.sample(range(data_num),data_num) 
print(f"data_num:{data_num}")
train_data = np.zeros([word_num*b_num,train_num,data_length])
test_data = np.zeros([word_num*b_num,test_num,data_length])

train_data = all_filted_data[:,train_shuffle[0:train_num],:]
test_data = all_filted_data[:,train_shuffle[train_num:data_num],:]
#hmm_model = hmm.GaussianHMM(n_components=state_num,covariance_type="full",init_params="")
if set_init == True:
	model = [hmm.GaussianHMM(n_components=one_state_num,covariance_type="full",init_params="") for i in range(word_num*ensemble_num)]
else:
	model = [hmm.GaussianHMM(n_components=one_state_num,covariance_type="full") for i in range(word_num*ensemble_num)]

#init init_prob
init_prob = []
temp = []
for i in range(one_state_num):
	temp.extend([np.random.rand()])

temp[0] += 1	
init_prob.extend((temp/np.sum(temp)).tolist())


ans = []
print("train start")
one_train_num = train_num//ensemble_num
for i in range(word_num):
	for e in range(ensemble_num):
		if set_init == True:
			#init a : transfer mat
			a = []

			for s_j in range(one_state_num):
				temp = []
				for s_i in range(one_state_num):	
					temp.append(np.random.rand())	
					
				a.append(temp)

			a = np.array(a)

			a[0,:] = 1.0/state_num	
			for j in range(one_state_num-1):
				a[j+1,j+1] += 1

				
			for s_j in range(one_state_num):
				temp = 0
				for s_i in range(one_state_num):
					temp += a[s_j,s_i]	
				a[s_j,:] = a[s_j,:]/temp 

			a = a.tolist()


			#init mu

			train_mu = np.zeros((one_state_num,b_num))
			for s_i in range(one_state_num):
				for s_j in range(b_num):
					train_mu[s_i,s_j] = train_data[i*b_num+s_j,e*one_train_num:(1+e)*one_train_num,s_i*one_state_len:(1+s_i)*one_state_len].mean()
			#init cov
			train_cov = np.tile(np.identity(b_num),(one_state_num,1,1))
			print("train_cov.shape : ",train_cov.shape)
			for s_i in range(one_state_num):
				if b_num ==1:
					train_cov[s_i] = np.cov(train_data[i,e*one_train_num:(1+e)*one_train_num,s_i*one_state_len:(1+s_i)*one_state_len].reshape(1,-1))

				else:
					train_cov[s_i] = np.cov(train_data[i*b_num:i*b_num+2,e*one_train_num:(1+e)*one_train_num,s_i*one_state_len:(1+s_i)*one_state_len].reshape((2,-1)),train_data[i*b_num+2,e*one_train_num:(1+e)*one_train_num,s_i*one_state_len:(1+s_i)*one_state_len].reshape((1,-1)),rowvar=1)
				for b in range(b_num):
					train_cov[s_i,b,b]+=0.0001
			print(i,"-",e)

			#train_cov[i*one_state_num+j+1] = np.cov(data_div_state[i,0:2,j],data_div_state[i,2,j],rowvar=1)

			
			#init params
		
			model[i*ensemble_num+e].startprob_ = init_prob
			model[i*ensemble_num+e].transmat_ = a
			model[i*ensemble_num+e].means_ = train_mu
			model[i*ensemble_num+e].covars_ = train_cov
			
		temp = np.zeros((0,data_length))
		
		for j in range(b_num):
			temp = np.vstack([temp,train_data[i*b_num+j,e*one_train_num:(1+e)*one_train_num,:].sum(axis = 0)/one_train_num])
		print(np.all(np.isnan(temp))==True)	
		print("fit")
		model[i*ensemble_num+e].fit(temp.T)
		
	
print("train end")
print("classification start")
prob_ans = np.zeros(word_num*ensemble_num)
ans_count = 0
for i in range(word_num):	
	print("classification : ",i)
	for j in range(round(test_num/test_add_num)):
		temp = np.zeros((0,data_length))
		temp_ans = np.zeros(ensemble_num)
		for k in range(b_num):
			temp = np.vstack([temp,test_data[i*b_num+k,j*test_add_num:(1+j)*test_add_num,:].mean(axis=0)]) 
		if j == 0:
			print(temp.shape)
		for k in range(word_num*ensemble_num):
			
			prob_ans[k] = model[k].score(temp.T)
		if j==0:
			print(prob_ans)
		for k in range(ensemble_num):
			temp_ans[k] = np.argmax(prob_ans[k:word_num*ensemble_num:ensemble_num])
		

		ans.append(int(statistics.mode(temp_ans)))		
		if ans[-1]==i:
			ans_count +=1
print("classification end")
print("ans : ",ans)
print("ans rate  : ",ans_count/(word_num*test_num/test_add_num))

