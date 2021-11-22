import numpy as np
import random

class qlearning_core:
	def __init__(self,action_space,observation_space):
		self.action_space = list(action_space)
		self.observation_space = list(observation_space)
		self.qTable = np.zeros([len(observation_space),len(action_space)])

	def set_exploration_parameters(self,start_exploration_rate,final_exploration_rate,exploration_dec_rate):
		self.exploration_rate = float(start_exploration_rate)
		self.final_exploration_rate = float(final_exploration_rate)
		self.exploration_dec_rate = float(exploration_dec_rate)

	def set_traing_rates(self,learning_rate,discount_rate):
		self.learning_rate = float(learning_rate)
		self.discount_rate = float(discount_rate)

	def computate_bellman_equation(self,reward_SA,new_S):
		return reward_SA + self.discount_rate*(max(self.qTable[new_S,]))

	#UPDATE QTABLE
	def update_qtable(self,current_S,action_A,reward_SA,new_S):
		self.qTable[current_S,action_A] = round((1-self.learning_rate)*self.qTable[current_S,action_A]+\
			self.learning_rate*(self.computate_bellman_equation(reward_SA,new_S)),5)

	#CHOOSE ACTION
	def choose_action(self,current_S):
		#Explore
		if(random.uniform(0,1)<self.exploration_rate):
			return random.choice(self.action_space)
		#Exploit
		else:
			index = self.qTable[current_S,].tolist().index(max(self.qTable[current_S,]))
			return self.action_space[index]

	def update_exploration_rate(self):
		if(self.exploration_rate > self.final_exploration_rate):
			self.exploration_rate -= self.exploration_dec_rate

	def print_Q_table(self):
		input(self.qTable)


class continues_to_discrete:
	def define_bounds(self,center_value,size_window,numb_windows):
		list_bounds = []
		numb_windows_above_center = int(round(numb_windows/2,0))		
		for i in range(numb_windows-1):
			list_bounds.append(round(center_value-((numb_windows_above_center-1-i)*size_window),5))
		return list_bounds

	def prepare(self,center_values,size_windows,numb_windows):
		self.list_bounds_per_dim = []
		numb_categ = 1
		#len(center_values)==len(pases)==len(numb_windows)
		for dim in range(len(center_values)):
			self.list_bounds_per_dim.append(self.define_bounds(center_values[dim],size_windows[dim],numb_windows[dim]))
			numb_categ = numb_categ*(len(self.list_bounds_per_dim[dim])+1)
		#print(self.list_bounds_per_dim)
		#print("Number of Observations: "+str(numb_categ))
		return range(numb_categ)

	def define_category_dimention(self,value,list_bounds):
		for i in range(len(list_bounds)):
			if(value<list_bounds[i]):
				return i
		return len(list_bounds)

	def define_category(self,list_values):
		category = 0
		sum_dim_sizes = 1
		for i in range(len(list_values)):
			category += (sum_dim_sizes)*self.define_category_dimention(list_values[i],self.list_bounds_per_dim[i])
			sum_dim_sizes = sum_dim_sizes*(len(self.list_bounds_per_dim[i])+1)
		return category
		
