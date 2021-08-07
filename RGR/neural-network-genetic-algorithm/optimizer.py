"""
Class that holds a genetic algorithm for evolving a network.
"""
from functools import reduce
from operator import add
import random
from network import Network

class Optimizer():
    """Class that implements genetic algorithm for MLP optimization."""

    def __init__(self, nn_param_choices, retain=0.4,
                 random_select=0.1, mutate_chance=0.25, radical_mutate_chance=0.1):
        """Create an optimizer.

        Args:
            nn_param_choices (dict): Possible network paremters
            retain (float): Percentage of population to retain after
                each generation
            random_select (float): Probability of a rejected network
                remaining in the population
            mutate_chance (float): Probability a network will be
                randomly mutated

        """

        self.mutate_chance = mutate_chance
        self.radical_mutate_chance = radical_mutate_chance
        self.random_select = random_select
        self.retain = retain
        self.nn_param_choices = nn_param_choices

    def create_population(self, count):
        """Create a population of random networks.

        Args:
            count (int): Number of networks to generate, aka the
                size of the population

        Returns:
            (list): Population of network objects

        """

        pop = []
        
        for _ in range(0, count):
            # Create a random network.
            network = Network(self.nn_param_choices)
            network.create_random()

            # Add the network to our population.
            pop.append(network)

        return pop

    @staticmethod
    def fitness(network):
        """Return the accuracy, which is our fitness function."""
        return network.accuracy

    def grade(self, pop):
        """Find average fitness for a population.

        Args:
            pop (list): The population of networks

        Returns:
            (float): The average accuracy of the population

        """

        summed = reduce(add, (self.fitness(network) for network in pop))
        return summed / float((len(pop)))

    def breed(self, mother, father):
        """Make two children as parts of their parents.

        Args:
            mother (dict): Network parameters
            father (dict): Network parameters

        Returns:
            (list): Two network objects

        """

        children = []
        for _ in range(2):

            child = {}

            # Loop through the parameters and pick params for the kid.
            for param in self.nn_param_choices:
                child[param] = random.choice(
                    [mother.network[param], father.network[param]]
                )

            # Now create a network object.
            network = Network(self.nn_param_choices)
            network.create_set(child)

            # Randomly mutate some of the children.
            if self.mutate_chance > random.random():
                network = self.mutate(network)

            children.append(network)

        return children

    def int_mutate(self, number, mutate_percent=10):
        """Mutates integer number up to mutate_percent from its initial value,
        as example, 10 could be mutated to a value from range [9:11].

        Args:
            number (int): Number to mutate.
            father (dict): Percent of the mutation.

        Returns:
            (int): Mutated number
        """

        percent = number * (mutate_percent/100)
        return random.randint(int(number - percent), int(number + percent))

    def float_mutate(self, number, mutate_percent=15):
        """Mutates float number up to mutate_percent from its initial value.

        Args:
            number (float): Number to mutate.
            father (dict): Percent of the mutation.

        Returns:
            (float): Mutated number
        """

        percent = number * (mutate_percent/100)
        return random.uniform(number - percent, number + percent)

    def random_layers(self, tuple_layers):
        """Mutates tuple that represents number of neurons in every layer of MLP.

        Args:
            tuple_layers (tuple): tuple that represents number of neurons in every layer of MLP

        Returns:
            (tuple): Mutated tuple
        """

        mutated = []

        for layer_num in tuple_layers:
            mutated.append(self.int_mutate(layer_num))

        return tuple(mutated)

    def mutate(self, network):
        """Randomly mutate one part of the network.

        Args:
            network (dict): The network parameters to mutate

        Returns:
            (Network): A randomly mutated network object

        """
             
        # Choose a random key.
        key = random.choice(list(self.nn_param_choices.keys()))
        val = network.network[key]
        # Mutate one of the params.
        if isinstance(val, tuple):
            # with a little chance - change the number of layers
            if random.random() > self.radical_mutate_chance:
                network.network[key] = self.random_layers(val)
            else:
                network.network[key] = random.choice(self.nn_param_choices[key])
        if isinstance(val, int):
            # Chance of radical mutation
            if random.random() > self.radical_mutate_chance:
                network.network[key] = self.int_mutate(val) 
            else:
                network.network[key] = random.choice(self.nn_param_choices[key])
        if isinstance(val, float): 
            if random.random() > self.radical_mutate_chance:
                network.network[key] = self.float_mutate(val) 
            else:
                network.network[key] = random.choice(self.nn_param_choices[key])
        if isinstance(val, str): 
            network.network[key] = random.choice(self.nn_param_choices[key])

        return network

    def evolve(self, pop):
        """Evolve a population of networks.

        Args:
            pop (list): A list of network parameters

        Returns:
            (list): The evolved population of networks
        """
        
        # Get scores for each network.
        graded = [(self.fitness(network), network) for network in pop]
        # Sort on the scores.
        graded = [x[1] for x in sorted(graded, key=lambda x: x[0], reverse=True)]
        # Get the number we want to keep for the next gen.
        retain_length = int(len(graded)*self.retain)
        # The parents are every network we want to keep.
        parents = graded[:retain_length]

        # For those we aren't keeping, randomly keep some anyway.
        for individual in graded[retain_length:]:
            if self.random_select > random.random():
                parents.append(individual)

        # Now find out how many spots we have left to fill.
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []

        # Add children, which are bred from two remaining networks.
        while len(children) < desired_length:
            self.born_childrens(parents, children, desired_length)

        parents.extend(children)

        return parents

    def born_childrens(self, parents, children, desired_length):
        """Create childrens from given networks
        
        Args:
            parents (list): networks from wich childrens will be created
            children (list): list of childrens to extend
            desired_length (int): number of childrens to born
        """

        # Get a random mom and dad.
        male = random.randint(0, len(parents)-1)
        female = random.randint(0, len(parents)-1)

        # Assuming they aren't the same network...
        if male == female:
            return

        male = parents[male]
        female = parents[female]

        # Breed them.
        babies = self.breed(male, female)

        # Add the children one at a time.
        for baby in babies:
            # Don't grow larger than desired length.
            if len(children) >= desired_length:
                return
            
            children.append(baby)