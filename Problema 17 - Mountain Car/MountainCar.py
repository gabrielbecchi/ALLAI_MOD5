import numpy as np
import matplotlib.pyplot as plt
import gym
import time

from qlearning import qlearning_core,continues_to_discrete


class MountainCar:
	def __init__(self):
		self.env = gym.make('MountainCar-v0')

		self.cTd = continues_to_discrete()
		# center_values,size_windows,numb_windows
		center_values = [-0.3,0]
		size_windows = [0.15,0.02]
		numb_windows = [18,5]
		list_observations = self.cTd.prepare(center_values,size_windows,numb_windows)

		# action_space,observation_space
		self.core = qlearning_core([0,1,2],list_observations)
		# start_exploration_rate,final_exploration_rate,exploration_dec_rate
		self.core.set_exploration_parameters(1,0.01,0.0001)
		# learning_rate,discount_rate
		self.core.set_traing_rates(0.05,0.9)

	def run(self,episodes):
		#plt.axis([0,1000, 0, 350])
		dic_results = {"Failed":0,"Sucess":0}
		countinous_100 = 0
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
				total_reward = reward
				distance = new_observation[0]
				if(done and max_iterations>=199):
					total_reward = 10*new_observation[0]
				#print(new_observation[0])
				if(done and distance >= 0.5):
					total_reward = 10000

				new_observation = self.cTd.define_category(new_observation)
				#Update Q-Table
				# current_S,action_A,reward_SA,new_S
				self.core.update_qtable(observation,action,total_reward,new_observation)
				self.core.update_exploration_rate()
				observation = new_observation
				max_iterations += 1
				if(ep%100 == 0):
					self.env.render()

			#plt.scatter(ep,max_iterations)
			#plt.pause(0.01)
			#input(max_iterations)
			if(ep % 100 == 0):
				print(dic_results)
				dic_results = {"Failed":0,"Sucess":0}
				#np.set_printoptions(threshold=np.inf)
				#print(self.core.qTable)
			if(distance >= 0.5):
				dic_results["Sucess"] += 1
				countinous_100 += 1
				if(countinous_100 == 100):
					print("Finish EP: "+str(ep))
					break
				continue
			else:
				countinous_100 = 0
				dic_results["Failed"] += 1
				continue
			#time.sleep(1)
		#print(dic_results)

machine = MountainCar()
machine.run(100000)