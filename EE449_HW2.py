# Imports
import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint, random, shuffle, uniform
from copy import deepcopy as dcopy
from multiprocessing import Pool

from numpy.lib.utils import source
source_img = cv2.imread("painting.png")
HEIGHT, WIDTH, CHANNELS = source_img.shape

# Helpers


def ramp(x):
    return max(0, x)


def create_blank_img():
    img = np.ones((HEIGHT, WIDTH, 3), np.uint8)
    img[:] = 255
    return dcopy(img)


def Pythagorean(x, y):
    return (x**2+y**2)**0.5


def create_empty_list(empty_list=None):
    if empty_list == None:
        empty_list = []
    return empty_list

# Classes


class Gene():
    global HEIGHT, WIDTH

    def __init__(self, x=0, y=0, r=0, R=0, G=0, B=0, A=0):
        self.x = x
        self.y = y
        self.r = r
        self.R = R
        self.G = G
        self.B = B
        self.A = A

    def random_initialize(self):
        self.x = randint(0, HEIGHT)
        self.y = randint(0, WIDTH)
        self.r = randint(15, 60)
        self.R = randint(0, 255)
        self.G = randint(0, 255)
        self.B = randint(0, 255)
        self.A = random()

    def print_object(self):
        print("x =", self.x, ", y =", self.y, ", r =", self.r, ", R =",
              self.R, ", G =", self.G, ", B =", self.B, ", A =", self.A)

    def check_if_in_image(self):
        if self.x < self.r or abs(self.x-WIDTH) < self.r or abs(self.y-HEIGHT) < self.r or self.y < self.r or (0 < self.x < WIDTH and 0 < self.y < HEIGHT):
            return True
        else:
            return False

# Since an individual has only one chromosome, I avoided introducing extra classes and used genes directly under individiual.


class Ind():

    def __init__(self, num_genes=None, genes=None, fitness=0):
        i = 0
        if genes is None:
            genes = create_empty_list()
            while i < num_genes:
                gene = Gene()
                gene.random_initialize()
                if gene.check_if_in_image():
                    genes.append(gene)
                    i += 1
        self.genes = genes
        self.calculate_fitness()

    def draw(self):
        sorted_genes = sorted(
            self.genes, key=lambda item: item.r, reverse=True)
        res_img = create_blank_img()
        for gene in sorted_genes:
            overlay = res_img
            cv2.circle(overlay, center=(gene.x, gene.y), radius=gene.r,
                       color=tuple(map(lambda item: item/1, (gene.B, gene.G, gene.R))), thickness=-1)
            res_img = cv2.addWeighted(overlay, gene.A, res_img, 1-gene.A, 0)
        return res_img

    def calculate_fitness(self):
        fitness = -1 * np.linalg.norm(np.int64(source_img)-self.draw())**2
        self.fitness = fitness


def evaluate_population(population):
    for ind in population:
        ind.calculate_fitness()
    return population


def create_population(num_individuals, num_genes):
    Population = create_empty_list()
    for _ in range(num_individuals):
        ind = Ind(num_genes)
        ind.calculate_fitness()
        Population.append(ind)
    return Population


def choose_n_elites(population, frac_of_elites):
    n = int(len(population) * frac_of_elites)
    sorted_pop = sorted(
        dcopy(population), key=lambda item: item.fitness, reverse=True)
    return sorted_pop[0:n], sorted_pop[n:]


def tournament_selection(Remaining, tm_size, parent_size):
    Winners = create_empty_list()
    for _ in range(parent_size):
        shuffle(Remaining)
        Remaining = Remaining[0:tm_size]
        sorted_remains = sorted(
            Remaining, key=lambda item: item.fitness, reverse=True)
        Winners.append(dcopy(sorted_remains[0]))
    return Winners


def unguided_mutation(gene: Gene):
    gene.random_initialize()
    return gene


def guided_mutation(gene: Gene):
    gene.x += randint(int(-0.25*gene.x), int(0.25*gene.x))
    gene.y += randint(int(-0.25*gene.y), int(0.25*gene.y))

    # Negative values are not allowed, hence used ramp
    gene.r = randint(ramp(gene.r-10), gene.r+10)
    gene.R = randint(ramp(gene.R-64), gene.R+64) if gene.R+64 < 255 else 255
    gene.G = randint(ramp(gene.G-64), gene.G+64) if gene.G+64 < 255 else 255
    gene.B = randint(ramp(gene.B-64), gene.B+64) if gene.B+64 < 255 else 255

    # [0,1]->[-1,1]->[-0.25,0.25]
    shifted_rn = uniform(-0.25, 0.25)
    gene.A += shifted_rn
    gene.A = gene.A if gene.A <= 1 and gene.A >= 0 else (
        0 if gene.A < 0 else 1)
    return gene


def mutate_individual(mutation, mutation_prob, ind: Ind):

    if random() < mutation_prob:
        genes = ind.genes
        gene_idx = randint(0, num_genes-1)
        if mutation.lower() == "guided":
            gene = guided_mutation(genes[gene_idx])
        else:
            gene = unguided_mutation(genes[gene_idx])
        ind.genes[gene_idx] = gene
    return ind


def draw_Pop(ind_list):
    genes = create_empty_list()
    for ind in ind_list:
        genes = genes + ind.genes
    IND = Ind(len(genes), genes=genes)
    img = IND.draw()
    return img


def crossover(parents: list):

    p1_genes = parents[0].genes
    p2_genes = parents[1].genes

    # initialize children genes
    ch1 = Ind(genes=p1_genes)
    ch2 = Ind(genes=p2_genes)

    # Exchange of genes with probability 0.5
    for i in range(len(ch1.genes)):
        if random() > 0.5:
            ch1.genes[i], ch2.genes[i] = dcopy(
                ch1.genes[i]), dcopy(ch2.genes[i])

    return [ch1, ch2]


def crossover_population(Parents: list):
    child_list = [crossover([Parents[2*i], Parents[2*i+1]])
                  for i in range((len(Parents)//2))]
    children = [x for l in child_list for x in l]
    return children


def round_to_even(a: int):
    if int(a) % 2 == 0:
        return int(a)
    else:
        return int(a)-1




# Set params


frac_of_elites = 0.2
num_individuals = 20
num_genes = 50
frac_parents = 0.6
mutation_prob = 0.2
mutation_type = "guided"
tm_size = 5
generations = 10000

curve_freq = 100
directory = "OUTPUTs//DEFAULTS//"
EXP_NAME = ""

plt.figure(figsize=(15,10))

#####################         EVOLUTION           #########################


# First create population
Pop = create_population(num_individuals=num_individuals, num_genes=num_genes)

# Fitness Curve
Fitness_Curve = create_empty_list()

for generation in range(generations+1):

    L = len(Pop)

    Pop = evaluate_population(Pop)

    Elites, NonElites = choose_n_elites(Pop, frac_of_elites)
    R = len(NonElites)

    num_parents = int(L*frac_parents)
    to_be_mutated = R - num_parents

    Winners = tournament_selection(
        NonElites, tm_size, R)

    Parents, Remainings = Winners[0:num_parents], Winners[-1*to_be_mutated:]
    Children = crossover_population(Parents)

    Children = list(map(lambda child: mutate_individual(
        mutation_type, mutation_prob, child), Children))

    Remainings = list(map(lambda remaining: mutate_individual(
        mutation_type, mutation_prob, remaining), Remainings))

    Next_Pop = dcopy(evaluate_population(Elites + Remainings + Children))

    Next_Pop = sorted(Next_Pop, key=lambda item: item.fitness, reverse=True)

    if generation % 1000 == 0:
        temp_img = Next_Pop[0].draw()
        #temp_img = cv2.cvtColor(temp_img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        cv2.imwrite(directory + EXP_NAME + "defaults_params" + "_generation" +
                    str(generation)+".png", temp_img)

    if generation % curve_freq == 0:
        best_fitness = Next_Pop[0].fitness
        Fitness_Curve.append(best_fitness)

    Pop = dcopy(Next_Pop)

plt.subplot(1,2,1)
plt.plot(Fitness_Curve[0:10])
plt.ylabel("Fitness score")
plt.xlabel("x" + str(curve_freq) + " Generations")
plt.title("Fitness of the best individual")

plt.subplot(1,2,2)
plt.plot(np.linspace(10,100,len(Fitness_Curve[10:])),Fitness_Curve[10:])
plt.ylabel("Fitness score")
plt.xlabel("x" + str(curve_freq) + " Generations")
plt.title("Fitness of the best individual")


plt.suptitle("Fitness Plot for Experiment with Default Parameters")
plt.savefig(directory + EXP_NAME + "Default_Params" + "_Fitness.png")



