import random

from deap import base
from deap import creator
from deap import tools

from func_drugs import drug_classifier

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


toolbox.register("k", random.randint, 1, 10)
toolbox.register("weights", random.randint, 0, 1)

toolbox.register("individual", tools.initRepeat, creator.Individual, 
	toolbox.k, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, 
	toolbox.weights, 1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# the goal ('fitness') function to be maximized
def evalOneMax(individual):
	input(individual)
	return drug_classifier(individual[0]),

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

#----------

def main():
	random.seed(64)
	pop = toolbox.population(n=10)
	CXPB, MUTPB = 0.5, 0.2
	
	print("Start of evolution")
	fitnesses = list(map(toolbox.evaluate, pop))
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit
	
	print("  Evaluated %i individuals" % len(pop))
	fits = [ind.fitness.values[0] for ind in pop]

	g = 0
	while max(fits) < 2 and g < 100:
		g = g + 1
		print("-- Generation %i --" % g)
		offspring = toolbox.select(pop, len(pop))
		# Clone the selected individuals
		offspring = list(map(toolbox.clone, offspring))
		for child1, child2 in zip(offspring[::2], offspring[1::2]):
			if random.random() < CXPB:
				toolbox.mate(child1, child2)
				del child1.fitness.values
				del child2.fitness.values

		for mutant in offspring:
			if random.random() < MUTPB:
				toolbox.mutate(mutant)
				del mutant.fitness.values
	
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit
		
		print("  Evaluated %i individuals" % len(invalid_ind))
		pop[:] = offspring
		fits = [ind.fitness.values[0] for ind in pop]
		length = len(pop)
		mean = sum(fits) / length
		sum2 = sum(x*x for x in fits)
		std = abs(sum2 / length - mean**2)**0.5
		
		print("  Min %s" % min(fits))
		print("  Max %s" % max(fits))
		print("  Avg %s" % mean)
		print("  Std %s" % std)
	
	print("-- End of (successful) evolution --")
	best_ind = tools.selBest(pop, 1)[0]
	print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

if __name__ == "__main__":
	main()