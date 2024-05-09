import random
import matplotlib.pyplot as plt

#Const 
ONE_MAX_LENGTH = 100

#Const Gen Algorithm
POPULTION_SIZE = 200 #The number of individuals in the population
P_CROSSOVER = 0.9 #The probability of crossover
P_MUTATION = 0.1 #The probability of mutation
MAX_GENERATIONS = 50 #The max number of generations(iterations in the algorithm)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

class FitnessMax:
    def __init__(self) -> None:
        self.values = [0]


class Individual(list):
    def __init__(self, *args: int) -> None:
        super().__init__(*args)
        self.fitness = FitnessMax()

def oneMaxFitness(individual: Individual) -> FitnessMax:
    return sum(individual) # Sum of elements in the array

def individualCreator() -> Individual:
    return Individual([random.randint(0, 1) for i in range(ONE_MAX_LENGTH)])

def populationCreator(n: int = 0) -> list:
    return list([individualCreator() for i in range(n)])

population = populationCreator(n = POPULTION_SIZE)
generationCounter = 0

fitnessValues = list(map(oneMaxFitness, population))

for individual, fitnessValue in zip(population, fitnessValues):
    individual.fitness.values = fitnessValue

maxFitnessValues = []
meanFitnessValues = []

def clone(value):
    ind = Individual(value[:])
    ind.fitness.values[0] = value.fitness.values[0]
    return ind

def selTournament(population: list, p_len: int = 0) -> list:
    offspring = []
    for n in range(p_len):
        i1 = i2 = i3 = 0
        while i1 == i2 or i1 == i3 or i2 == i3:
            i1, i2, i3 = random.randint(0, p_len - 1), random.randint(0, p_len - 1), random.randint(0, p_len - 1)

        offspring.append(max([population[i1], population[i2], population[i3]], key=lambda ind: ind.fitness.values[0]))

    return offspring

def cxOnePoint(ind1: Individual, ind2: Individual):
    s = random.randint(2, len(ind1) - 3)
    ind1[s:], ind2[s:] = ind2[s:], ind1[s:]

def mutFlipBit(mutant, indpb = 0.1):
    for indx in range(len(mutant)):
        if random.random() < indpb:
            mutant[indx] = 0 if mutant[indx] == 1 else 1

fitnessValues = [individual.fitness.values[0] for individual in population]

while generationCounter < MAX_GENERATIONS and max(fitnessValues) < ONE_MAX_LENGTH:
    generationCounter += 1
    offspring = selTournament(population, p_len = len(population))
    offspring = list(map(clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < P_CROSSOVER:
            cxOnePoint(child1, child2)

        for mutant in offspring:
            if random.random() < P_MUTATION:
                mutFlipBit(mutant, indpb=1.0/ONE_MAX_LENGTH)
        
        freshFitnessValues = list(map(oneMaxFitness, offspring))
        for individual, fitnessValue in zip(offspring, freshFitnessValues):
            individual.fitness.values = fitnessValue

        population[:] = offspring

        fitnessValues = [ind.fitness.values[0] for ind in population]

        maxFitness = max(fitnessValues)
        meanFitness = sum(fitnessValues) / len(population)
        maxFitnessValues.append(maxFitness)
        meanFitnessValues.append(meanFitness)
        print("Generation:", generationCounter, "Max Fitness:", maxFitness, "Mean Fitness:", meanFitness)

        best_index = fitnessValues.index(max(fitnessValues))
        print("Best individual: ", *population[best_index], "\n")

plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Generation')
plt.ylabel('Max / Mean Fitness')
plt.title('Max and Mean Fitness over Generations')
plt.show() 