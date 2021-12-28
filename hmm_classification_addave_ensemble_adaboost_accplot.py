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
import collections
import time 
from sklearn.model_selection import train_test_split
#filtaring method
def lowpass(x, samplerate, fp, fs, gpass, gstop):
	fn = samplerate/2
	wp = fp/fn
	ws = fs/fn
	N,Wn = signal.buttord(wp,ws,gpass,gstop)
	b,a = signal.butter(N,Wn,"low")
	y = signal.filtfilt(b,a,x)
	return y


#read data method


def read_data(Data_Path,words,ch_names,one_data_len,one_state_num,b_num,samplerate=1000,fp=30,fs=50,gpass=3,gstop=40):	
	all_filted_data = []
	all_label = np.empty(0,dtype=int)
	word_num = len(words)
	print("read start")
	for j in range(len(words)):
		one_word_data = []
		for k in range(len(ch_names)):
	
			data = pd.read_csv(os.path.join(Data_Path,"data_"  + words[j] +ch_names[k] +".csv")).values

			filted_data = np.empty((0,data.shape[1]-100),float)
			
			for i in range(data.shape[0]):
				filted_data = np.vstack([filted_data,lowpass(data[i,:],samplerate,fp,fs,gpass,gstop)[50:-50]])	
				

			data = filted_data	
				
			if k == 0:
				one_word_data = np.empty((data.shape[0],b_num,data.shape[1]))
		
			one_word_data[:,k,:] =data 

			if j == 0 and k == 0:
				#(data_num, data_length) = filted_data.shape	
				(data_num, data_length) = data.shape	
				all_filted_data = np.empty((0,b_num,data_length))
				
		all_filted_data = np.concatenate([all_filted_data,one_word_data],0)
		all_label = np.concatenate([all_label,np.full(one_word_data.shape[0],j,dtype=int)],0)
		
	print("read end")	

	return all_filted_data,all_label

def normalize_func(all_data):
	return  (all_data-all_data.mean(axis=1).reshape(-1,1))/all_data.std(axis=1).reshape(-1,1)

def reject_artifact(all_data,all_label,threshold):
	(data_num,b_num,data_length) = all_data.shape
	applied_data = np.empty((0,b_num,data_length))
	applied_label = np.empty(0)
	for i in range(data_num):
		for j in range(b_num):
			if max(abs(all_data[i,j,:])) >threshold:
				print("reject")
				break
					
			elif j == b_num-1:

				temp = normalize_func(all_data[i])
				applied_data = np.append(applied_data,np.array([temp]),axis=0)
				applied_label = np.append(applied_label,all_label[i])

	return applied_data,applied_label	

Data_Path = os.path.join("/","mnt","c","Users","Hirok","Desktop","M1","1_word_HMM","data","overt","div_data")
#Data_Path = os.path.join("/","mnt","c","Users","Hirok","Desktop","M1","1_word_HMM","data","div_data")


Save_Path =  os.path.join("/","mnt","c","Users","Hirok","Desktop","M1","1_word_HMM","picture")


color = ["#000000","#44ffff","#88ffff","#bbffff","#eeffff","#ff44ff","#ff88ff","#ffbbff","#ffeeff","#ffff44","#ffff88","#ffffbb","#ffffee","#444444","#888888","#bbbbbb","#eeeeee","#44ff44","#88ff88","#bbffbb","#eeffee"]
words = ["a","i","u","e","o"]
ch_names = [" F7-REF"," T7-REF"," Cz-REF"]
#ch_names = ["_F7-T7","_T7-Cz","_Cz-F7"]
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
Restrict = 1
hmm_train_num_rate = False #set later 
test_num_rate = False #set later
test_add_nums = np.array([10,20,30,40,50,60,70,80])
iteration = 10
ensemble_num = 3
#read data
all_data , all_label= read_data(Data_Path,words,ch_names,one_data_len,one_state_num,b_num,samplerate=1000,fp=30,fs=50,gpass=3,gstop=40)
rj_applied_data , rj_applied_label = reject_artifact(all_data,all_label,120)
(data_num,_,data_length) = rj_applied_data.shape

hmm_train_num_rates = (test_add_nums*ensemble_num*word_num)/(data_num)
test_num_rates = 1-hmm_train_num_rates
acc_list = []


for ver in range(len(hmm_train_num_rates)):
	hmm_train_num_rate = hmm_train_num_rates[ver]
	test_num_rate = test_num_rates[ver]
	test_add_num = test_add_nums[ver]
	temp_acc_list = []

	for it in range(iteration):

		#hmm_train_num_rate = (word_num*test_add_num*ensemble_num)/data_num
		#hmm_train_num_rate = (word_num*test_add_num)/data_num

		#test_num_rate = 1-hmm_train_num_rate

		train_num = round(data_num*hmm_train_num_rate)
		test_num = round(data_num*test_num_rate)
		"""
		train_shuffle = random.sample(range(data_num),data_num) 
		print(f"data_num:{data_num}")
		train_data = np.zeros([word_num*b_num,train_num,data_length])
		test_data = np.zeros([word_num*b_num,test_num,data_length])

		train_data = all_filted_data[:,train_shuffle[0:train_num],:]
		test_data = all_filted_data[:,train_shuffle[train_num:data_num],:]
		"""
		x_train , x_test, y_train, y_test = train_test_split(rj_applied_data,rj_applied_label,test_size=1-hmm_train_num_rate,stratify=rj_applied_label)


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
		alpha = np.zeros(ensemble_num)
		data_w = np.ones(x_train.shape[0])	
		#for i in range(word_num):
		for e in range(ensemble_num):
			weighted_train_data = x_train 	
			for i in range(b_num):

				weighted_train_data[:,i,:] = data_w.reshape(-1,1)*weighted_train_data[:,i,:]
			
				
					
			#one_train_num = train_data.shape[0]//ensemble_num
			#print("one_train_num : {}".format(one_train_num))
			#print("train_data num : {}".format(train_data.shape))
			#for e in range(ensemble_num):
			for i in range(word_num):
				#train_data_origin = x_train[y_train==i,:]

				#train_data = data_w[i].reshape(-1,1)*train_data_origin
				train_data = weighted_train_data[y_train==i,:]
				print("train_data_num : {}".format(train_data.shape[0]))
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
							#train_mu[s_i,s_j] = train_data[e*one_train_num:(e+1)*one_train_num,s_j,s_i*one_state_len:(1+s_i)*one_state_len].mean()
							train_mu[s_i,s_j] = train_data[:,s_j,s_i*one_state_len:(1+s_i)*one_state_len].mean()


					#init cov
					train_cov = np.tile(np.identity(b_num),(one_state_num,1,1))
					for s_i in range(one_state_num):
						if b_num==1 :
							#train_cov[s_i] = np.cov(train_data[e*one_train_num:(1+e)*one_train_num,s_i*one_state_len:(1+s_i)*one_state_len].reshape(1,-1))
							train_cov[s_i] = np.cov(train_data[:,s_i*one_state_len:(1+s_i)*one_state_len].reshape(1,-1))

						else:	
							#print(train_data[e*one_train_num:(e+1)*one_train_num,0:b_num-1,s_i*one_state_len:(1+s_i)*one_state_len].shape)
							#train_cov[s_i] = np.cov(train_data[e*one_train_num:(e+1)*one_train_num,0:b_num-1,s_i*one_state_len:(1+s_i)*one_state_len].mean(axis=0).reshape((2,-1)),train_data[e*one_train_num:(e+1)*one_train_num,b_num-1,s_i*one_state_len:(1+s_i)*one_state_len].mean(axis=0).reshape((1,-1)),rowvar=1)
							train_cov[s_i] = np.cov(train_data[:,0:b_num-1,s_i*one_state_len:(1+s_i)*one_state_len].mean(axis=0).reshape((2,-1)),train_data[:,b_num-1,s_i*one_state_len:(1+s_i)*one_state_len].mean(axis=0).reshape((1,-1)),rowvar=1)

							#print(train_cov[s_i])
							#train_cov[s_i] = np.cov(train_data[e*one_train_num:(e+1)*one_train_num,0:b_num-1,s_i*one_state_len:(1+s_i)*one_state_len].transpose(1,0,2).reshape((2,-1)),train_data[e*one_train_num:(e+1)*one_train_num,b_num-1,s_i*one_state_len:(1+s_i)*one_state_len].reshape((1,-1)),rowvar=1)

						for b in range(b_num):
							train_cov[s_i,b,b]+=0.0001

					#init params
					
					model[i*ensemble_num+e].startprob_ = init_prob
					model[i*ensemble_num+e].transmat_ = a
					model[i*ensemble_num+e].means_ = train_mu
					model[i*ensemble_num+e].covars_ = train_cov
					

				temp = np.zeros((0,data_length))
				temp = train_data[:,:].mean(axis=0)
				print(temp.shape)
				#sys.exit()			
				model[i*ensemble_num+e].fit(temp.T)
			#calculate param
			train_miss_count = np.zeros((word_num))
			train_prob_ans = np.zeros((word_num))
			train_ans = []
			for i in range(x_train.shape[0]):
				for j in range(word_num):

					train_prob_ans[j] = model[j*ensemble_num+e].score(x_train[i,:].T)
			
				train_ans.append(np.argmax(train_prob_ans))
			#TF_matrix = np.full((train_data.shape[0],word_num),False)
			TF_matrix=y_train==np.array(train_ans)

			error_num = x_train.shape[0] - np.sum(TF_matrix)
			alpha[e] = (x_train.shape[0] - error_num)/error_num 
			print(f"error_num : {error_num}")
			data_w[TF_matrix==False] +=1
			data_w = x_train.shape[0]*(data_w/np.sum(data_w))
				
			#alpha = np.zeros((word_num,ensemble_num))
			#data_w = np.ones(train_data.shape[0])	
				

		print("train end")

			
		print("train end")
		print("classification start")
		temp_prob_ans = np.zeros(word_num*ensemble_num)
		prob_ans = np.zeros(word_num)
		ans_count = 0
		confusion_matrix = np.zeros((word_num,word_num))
		for i in range(word_num):	
			print("classification : ",i)
			temp_test_data = x_test[y_test==i]
			for j in range(round(temp_test_data.shape[0]/test_add_num)):
				temp_ans = np.zeros(word_num)
				temp_prob = np.zeros(ensemble_num)
				"""
				for k in range(b_num):
					temp = np.vstack([temp,test_data[i*b_num+k,j*test_add_num:(1+j)*test_add_num,:].mean(axis=0)]) 
				if j == 0:
					print(temp.shape)
				"""
				for k in range(word_num*ensemble_num):
					
					temp_prob_ans[k] = model[k].score(temp_test_data[j*test_add_num:(j+1)*test_add_num,:,:].mean(axis=0).T)
				if j==0:
					print(prob_ans)
						
				for k in range(word_num):
					prob_ans[k] = np.sum(alpha * temp_prob_ans[k*ensemble_num:(k+1)*ensemble_num])
				ans.append(np.argmax(prob_ans))
				"""
					temp = np.sum(prob_ans[k:word_num*ensemble_num:ensemble_num])
					temp_ans[](temp)

					temp_ans = np.arg_max(np.sum(prob_ans[k:word_num*ensemble_num:ensemble_num]))
					temp_ans[k] = np.argmax(prob_ans[k:word_num*ensemble_num:ensemble_num])
					temp_prob[k] = np.max(prob_ans[k:word_num*ensemble_num:ensemble_num])

				print(temp_ans)
				count_ans = collections.Counter(temp_ans)
				max_count = max(count_ans.values())
				max_ans = []
				for l in range(word_num):
					if count_ans[l] == max_count :
						max_ans.append(l)
				if not len(max_ans) ==1:
					temp_prob_cf = -float("inf")
					for l in range(len(max_ans)):
						print("l:{}".format(l))
						print("temp_prob:{}".format(temp_prob))
						print("temp_ans:{}".format(temp_ans))
						print("temp_prob_cf:{}".format(np.array(temp_prob)[temp_ans==max_ans[l]]))
						if temp_prob_cf <np.max(np.array(temp_prob)[temp_ans==max_ans[l]]):	
							temp_prob_cf = np.max(np.array(temp_prob)[temp_ans==max_ans[l]])	
							temp_ans_cf = max_ans[l]
				else:
					temp_ans_cf = max_ans[0]			
				ans.append(temp_ans_cf)	
				#ans.append(int(statistics.mode(temp_ans)))		
				temp	
				"""

				confusion_matrix[i,ans[-1]] += 1

				if ans[-1]==i:
					ans_count +=1
		print("classification end")
		print("ans : ",ans)
		print("ans rate  : ",ans_count/(test_num/test_add_num))
		temp_acc_list.append(ans_count/(test_num/test_add_num))
		#print("confusion matrix")
		#print(confusion_matrix)
		#for k in range(word_num*ensemble_num):
		#	print(model[k].startprob_)	
		
		print(alpha)
	acc_list.append(temp_acc_list)
plt.boxplot(acc_list)
plt.show()
