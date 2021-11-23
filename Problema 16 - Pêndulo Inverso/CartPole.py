import numpy as np
import matplotlib.pyplot as plt
import gym
import time
import sys

from qlearning import qlearning_core,continues_to_discrete

np.set_printoptions(threshold=sys.maxsize)

class CartPole:
	def __init__(self):
		self.env = gym.make('CartPole-v0')
		self.cTd = continues_to_discrete()

		#Plot Lists
		self.number_interactions = []
		self.exploration_rate = []

		# center_values,size_windows,numb_windows
		center_values = [0,0,0,0]
		size_windows = [20,20,0.2,0.2]
		numb_windows = [4,4,36,36]
		#numb_windows = [2,2,2,2]
		list_observations = self.cTd.prepare(center_values,size_windows,numb_windows)

		# action_space,observation_space
		self.core = qlearning_core([0,1],list_observations)
		# start_exploration_rate,final_exploration_rate,exploration_dec_rate
		self.core.set_exploration_parameters(1,0.01,0.00002)
		# learning_rate,discount_rate
		#self.core.set_traing_rates(0.3,0.99)
		self.core.set_traing_rates(0.1,0.99)

	def run(self,episodes):
		#plt.axis([0,1000, 0, 350])
		dic_results = {0:0,100:0,200:0}
		continuous_200 = 0
		printable_ep = 0
		for ep in range(episodes):
			#print("EPISODE: "+str(ep+1))
			observation = self.env.reset()
			observation = self.cTd.define_category(observation)
			done = False
			total_reward = 0
			max_iterations = 0
			while(not(done)):
				#Choose and take action
				action = self.core.choose_action(observation)
				new_observation, reward, done, info = self.env.step(action)
				#input(new_observation)
				new_observation = self.cTd.define_category(new_observation)
				total_reward += reward
				#total_reward = reward
				if(done and max_iterations<199):
					total_reward = 0
				#Update Q-Table
				# current_S,action_A,reward_SA,new_S
				self.core.update_qtable(observation,action,total_reward,new_observation)
				self.core.update_exploration_rate()
				observation = new_observation
				max_iterations += 1
				if(ep%100 == 0):
					self.env.render()
			#input(self.core.qTable)


			self.number_interactions.append(max_iterations)
			self.exploration_rate.append(self.core.exploration_rate)
			if(ep % 100 == 0):
				print(dic_results)
				dic_results = {0:0,100:0,200:0}
			if(max_iterations < 100):
				dic_results[0] += 1
				continuous_200 = 0
			elif(max_iterations <= 199):
				dic_results[100] += 1
				continuous_200 = 0
			else:
				#Sucess
				dic_results[200] += 1
				continuous_200 += 1
			if(continuous_200 == 200):
				print("***")
				print("Episodes until sucess: "+str(ep+1))
				return
				#sys.exit()
		

	def finish(self):
		self.env.close()
		self.ep_list = list(range(1,len(self.number_interactions)+1))
		fig, ax1 = plt.subplots()

		ax1.plot(self.ep_list,self.number_interactions,'b.')
		ax1.set_xlabel("Episodes")
		# Make the y-axis label, ticks and tick labels match the line color.
		ax1.set_ylabel('Interactions', color='b')
		ax1.tick_params('y', colors='b')

		ax2 = ax1.twinx()
		ax2.plot(self.ep_list,self.exploration_rate,'r-')
		ax2.set_ylabel("Exploration Rate", color='r')
		ax2.tick_params('y', colors='r')

		fig.tight_layout()
		plt.show()

machine = CartPole()
machine.run(50000)
machine.finish()

