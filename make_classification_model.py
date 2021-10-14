#all_filted_data : not hmm train data



import numpy as np
import pandas
import os, sys
import pandas as pd
from hmmlearn import hmm
import matplotlib.pyplot as plt
from scipy import signal
import collections

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

color = ["#000000","#44ffff","#88ffff","#bbffff","#eeffff","#ff44ff","#ff88ff","#ffbbff","#ffeeff","#ffff44","#ffff88","#ffffbb","#ffffee","#444444","#888888","#bbbbbb","#eeeeee","#44ff44","#88ff88","#bbffbb","#eeffee"]
words = ["a","i","u","e","o"]
#ch_names = [" F7-REF"," T7-REF"," Cz-REF"]
ch_names = ["_F7-T7","_T7-Cz","_Cz-F7"]
fig_flag = 0
b_num = len(ch_names)
word_num = len(words)
all_data = np.empty((b_num,0))
one_state_len= 924
one_state_num = 4
state_num = 1+len(words)*one_state_num
data_div_state =[[[[] for i in range(one_state_num)] for j in range(b_num)] for k in range(word_num)]
data_div_state_0 = [[]for i in range(b_num)]
sum_data = np.empty((b_num,0))

hmm_trainnum_rate = 0.3
classification_modelnum_rate = 0.4
test_num_rate = 0.3
Restrict = 1

all_filted_data = []
for j in range(len(words)):
	temp_sum_data = np.empty((0,one_state_len))
	for k in range(len(ch_names)):
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
			all_filted_data = np.zeros((word_num*b_num,round(data_num*(classification_modelnum_rate+test_num_rate)),data_length))
		####
		all_filted_data[j*b_num+k] = filted_data[round(data_num*hmm_trainnum_rate):,:]	
		data_div_state_0[k].extend(filted_data[0:round(data_num*hmm_trainnum_rate),0:24].reshape(-1,).tolist())		
		for i in range(one_state_num):
			data_div_state[j][k][i].extend(filted_data[0:round(data_num*hmm_trainnum_rate),24+i*225:24+i*225+225].reshape(-1,).tolist())		
		#print(temp_sum_data.shape)
		pre_temp_sum_data=np.sum(filted_data,axis=0)
		temp_sum_data = np.vstack([temp_sum_data,(pre_temp_sum_data-pre_temp_sum_data.mean())/pre_temp_sum_data.std()])
		#temp_sum_data = np.vstack([temp_sum_data,np.sum(filted_data,axis=0)])


		"""
		filted_data = (filted_data-filted_data.mean(axis=1).T)/data.std(axis=1).T	
		if k == 0:
			temp_data[0] = data[0:24]
			for i in range(one_state_num):	
				temp_data[i+1] = data[24+i*500:24+i*500+500]	
		else :
			temp_data[0].append(data[0:24])
			for i in range(one_state_num):
				temp_data[i+1].append(data[24+i*500:24+i*500+500])

		"""	
	sum_data = np.hstack([sum_data,temp_sum_data])	

#init a
a = []

for j in range(state_num):
	temp = []
	for i in range(state_num):	
		temp.append(np.random.rand())	
		
	a.append(temp)

a = np.array(a)

a[0,:] = 1.0/state_num	
for j in range(state_num-1):
	if (j+1)%one_state_num == 0:
		a[j+1,0]+=1
	else:
		a[j+1,j+2]+=1
	a[j+1,j+1] += 1

	
for j in range(state_num):
	temp = 0
	for i in range(state_num):
		temp += a[j,i]	
	a[j,:] = a[j,:]/temp 

a = a.tolist()
#init mu and cov
train_mu = np.zeros((state_num,b_num))
train_cov = np.tile(np.identity(b_num),(state_num,1,1))
#data_div_state = np.array(data_div_state)
#print((data_div_state[2,1,1]))

for i in range(b_num):
	#print(type(data_div_state[:][i][0]))
	data_div_state_0 = np.array(data_div_state_0)


	print(np.array(data_div_state_0).shape)
	train_mu[0,i]= np.array(data_div_state_0[i]).mean()
#train_cov[0,i] =  np.cov(data_div_state_0[0:2].reshape((2,-1)),data_div_state_0[2:b_num].reshape(b_num-2,-1),rowvar=0)
train_cov[0] =  np.cov(data_div_state_0,rowvar=1)

print(train_cov[0])
data_div_state = np.array(data_div_state)
print(data_div_state.shape)
print(type(data_div_state[0,:,0]))
print(len(data_div_state[0,:,0]))

for i in range(word_num):
	for j in range(one_state_num):	
		print(f'i={i} : j={j}')
		print(np.array(data_div_state[i,0,j]).shape)
		for k in range(b_num):
			train_mu[i*one_state_num+j+1,k] = np.array(data_div_state[i,k,j]).mean()

		train_cov[i*one_state_num+j+1] = np.cov(data_div_state[i,0:2,j],data_div_state[i,2,j],rowvar=1)
print(train_cov.min())	
for i in range(word_num*one_state_num+1):
	for j in range(b_num):
		if train_cov[i,j,j] <1:
			train_cov[i,j,j] = 1.0

#print(train_mu)
#sys.exit()
#init init_prob
init_prob = []
temp = []
for i in range(state_num):
	temp.extend([np.random.rand()])

temp[0] += 1	
init_prob.extend((temp/np.sum(temp)).tolist())

print(train_cov.shape)

#init hmm_model
hmm_model = hmm.GaussianHMM(n_components=state_num,covariance_type="full",init_params="")

hmm_model.startprob_ = init_prob
hmm_model.transmat_ = a
hmm_model.means_ = train_mu
hmm_model.covars_ = train_cov

print(all_filted_data.shape)

#init plot
x_plot = np.linspace(1,sum_data.shape[1],sum_data.shape[1])

statelist = hmm_model.predict(sum_data.T)
plt.figure(figsize=(22,14))
"""
for t in range(len(x_plot)):
	for i in range(b_num):
		plt.scatter(x_plot[t],sum_data[i,t],c=color[statelist[t]],marker="+")
	print(t)
"""
for j in range(state_num):
	for i in range(b_num):
		plt.scatter(x_plot[statelist==j],sum_data[i,statelist==j],c=color[j],marker="+")
print("init plot")
#plt.savefig(os.path.join(Save_Path,"hmm_demo","init_plot.png"))
plt.show()	
print("init plot end")


#train
print("train_start")
hmm_model.fit(sum_data.T)
print("train_finish")

statelist = hmm_model.predict(sum_data.T)

plt.figure(figsize=(22,14))
for i in range(b_num):
	plt.scatter(x_plot,sum_data[i,:],c="#dddddd",marker="+")	
#plt.savefig(os.path.join(Save_Path,"hmm_demo","criate_plot.png"))
plt.show()

print(max(statelist))
print(sum_data.shape)
print(statelist.shape)
print(len(x_plot))
plt.figure(figsize=(22,14))
print(len(color))
for j in range(state_num):
	for i in range(b_num):
		plt.scatter(x_plot[statelist==j],sum_data[i,statelist==j],c=color[j],marker="+")


#plt.savefig(os.path.join(Save_Path,"hmm_demo","sate_plot.png"))
plt.show()	


##########caffici
#####
#hmm_model	:hmm model
#all_filted_data[j*b_num+k]	:filted data (not hmm train data) [word of 123(word_num*b_num) ,data num (data_num(classification_modelnum_rate+test_num_rate)),data length ()]
#data_length  			;data_lenth 
#word_num
#data_num			:num of data for one word
#classification_modelnum_rate	:rate of train data for claffication model 
#test_num_rate			:rate of test data


####
#train_data			:classification train_data

model_train_num =round(data_num*classification_modelnum_rate)
train_data = all_filted_data[:,:model_train_num,:]
train_label = np.zeros(word_num*model_train_num)
for i in range(word_num):
	train_label[model_train_num*i:model_train_num*(i+1)] = i

import random
train_shuffle = random.sample(range(word_num*model_train_num),word_num*model_train_num) 
train_data_state = np.empty((0,data_length))
for i in range(word_num*model_train_num):
	word_number= i//model_train_num
	#print(all_filted_data[word_number:word_number+3,i%model_train_num,:].shape)
	train_data_state = np.vstack([train_data_state,hmm_model.predict(all_filted_data[word_number:word_number+3,i%model_train_num,:].T)])	

x_plot = np.linspace(1,data_length,data_length)
for i in range(word_num):
	plt.subplot(3,2,i+1)
	for s in range(state_num):
		temp = np.count_nonzero(train_data_state[i*model_train_num:(i+1)*model_train_num] == s , axis=0)
		"""	
		temp = train_data_state[i*model_train_num:(i+1)*model_train_num] 
		for j in range(model_train_num):
			plt.plot(x_plot,temp[j,:])
		"""	
		#print(temp)
		if max(temp)>200:
			continue
		plt.plot(x_plot,temp,label="state "+str(s))

	

#plt.legend()
plt.show()
