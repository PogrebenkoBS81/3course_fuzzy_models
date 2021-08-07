"""Entry point to evolving the neural network. Start here.

Code was based on: https://github.com/harvitronix/neural-network-genetic-algorithm
"""
from optimizer import Optimizer
from tqdm import tqdm
import pandas as pd 
from pathlib import Path
import pickle 
from datetime import datetime

# Number of networks to save to disk
SAVE_NET_NUM = 3
# % of networks to keep from previous generation
# There is no crossover chance, best parents will be kept with their childrens in purpose to avoid degenerating process
RETAIN = 0.3
# Chanche of mutation 
MUTATE_CHANCE = 0.2
# Chance that during mutation int\tuple variables will be changed to some value from param list
# Used to add variance to algorithm and delay convergence (sometimes it's unnecessary, but anyway).
RADICAL_MUTATE_CHANCE = 0.1

# Number of generations to run 
GENERATIONS = 50
# Number of networks in each population
POPULATION = 60

def train_networks(networks, x, y):
    """Train each network.

    Args:
        networks (list): Current population of networks
        x (DataFrame): df of all collected data params
        y (DataFrame): df of same height with normal/abnormal classification of x
    """

    # Progress bar to display work of the program.
    pbar = tqdm(total=len(networks))
    # Train each network.
    for network in networks:
        network.train(x, y)
        pbar.update(1)
    # Close progress bar.
    pbar.close()

def get_average_accuracy(networks):
    """Get the average accuracy for a group of networks (calculated onle by best retained nets).

    Args:
        networks (list): List of networks

    Returns:
        float: The average accuracy of a population of networks.
    """

    total_accuracy = 0
    # Sort from best to worst, and get num of networks to save.
    sorted_net = sorted(networks, reverse=True, key=lambda x: x.accuracy)
    to_retain = int(len(sorted_net) * RETAIN)
    
    for network in sorted_net[:to_retain]:
        total_accuracy += network.accuracy
    # Get avg accuracy.
    return total_accuracy / to_retain

def generate(generations, population, nn_param_choices, x, y):
    """Generate a network with the genetic algorithm.

    Args:
        generations (int): Number of times to evole the population
        population (int): Number of networks in each generation
        nn_param_choices (dict): Parameter choices for networks
        x (DataFrame): df of all collected data params
        y (DataFrame): df of same height with normal/abnormal classification of x
    Args example:
        x:
                    Col1       Col2       Col3       Col4        Col5       Col6      Col7     Col8     Col9     Col10      Col11    Col12
            0    63.027818  22.552586  39.609117  40.475232   98.672917  -0.254400  0.744503  12.5661  14.5386  15.30468 -28.658501  43.5123
            2    33.841641   5.073991  36.641233  28.767649  123.945244  -0.199249  0.674504  19.3825  17.6963  13.72929   1.783007  40.6049
            ..         ...        ...        ...        ...         ...        ...       ...      ...      ...       ...        ...      ...

        y:
            0      Abnormal
            1      Normal
                    ...

    Returns:
        networks: the list of best networks.
    """

    # I could use rollback to old generation (if new avg acc < new avg acc),
    # but since code keeps the best networks "alive"and unchanged, result cannot go down. 
    # Any light acc drop during the work of the programm - result of non-deterministic learning process of MLP.
    optimizer = Optimizer(nn_param_choices, retain=RETAIN, mutate_chance=MUTATE_CHANCE, radical_mutate_chance=RADICAL_MUTATE_CHANCE)
    networks = optimizer.create_population(population)

    # Evolve the generation.
    for i in range(generations):
        print("***Doing generation %d of %d***" %(i + 1, generations))
        # Train and get accuracy for networks.
        train_networks(networks, x, y)
        # Get the average accuracy for this generation.
        average_accuracy = get_average_accuracy(networks)
        # Print out the average accuracy each generation.
        print("Generation top average: %.2f%%" % (average_accuracy * 100))
        # Evolve, except on the last iteration.
        if i != generations - 1:
            # Do the evolution.
            networks = optimizer.evolve(networks)
        print('#'*80)

    # Sort our final population.
    networks = sorted(networks, key=lambda x: x.accuracy, reverse=True)

    return networks

def print_networks(networks):
    """Print a list of networks.

    Args:
        networks (list): The population of networks
    """

    print('#'*80)

    for network in networks:
        network.print_network()

def get_data():
    """Prepares data from Dataset_spine.csv"""

    df = pd.read_csv('Dataset_spine.csv')
    # Drop dummy column.
    df = df.drop(['Unnamed: 13'], axis=1)
    # Rename columns according to: https://towardsdatascience.com/an-exploratory-data-analysis-on-lower-back-pain-6283d0b0123.
    df.rename(columns={
            "Col1" : "pelvic_incidence",
            "Col2" : "pelvic_tilt",
            "Col3" : "lumbar_lordosis_angle",	
            "Col4" : "sacral_slope",
            "Col5" : "pelvic_radius",	
            "Col6" : "degree_spondylolisthesis",
            "Col7" : "pelvic_slope",
            "Col8" : "Direct_tilt",	
            "Col9" : "thoracic_slope",	
            "Col10" : "cervical_tilt",	
            "Col11" : "sacrum_angle",	
            "Col12" : "scoliosis_slope",	
        }
    )
    # Generate set of data and it's classification set for MLP.
    y = df['Class_att']
    x = df.drop(['Class_att'], axis=1)

    return x, y

def save_networks(networks, dir_path = "./networks"):
    """Saves best networks to the disk."""

    Path(dir_path).mkdir(parents=True, exist_ok=True)

    for network in networks:
        file_name = f"{dir_path}/classifier_{datetime.today().strftime('%Y-%m-%d')}_acc{network.accuracy}.pkl"

        with open(file_name, 'wb') as fid:
            pickle.dump(network, fid)  


def main():
    """Evolve a network."""

    if POPULATION < SAVE_NET_NUM:
        raise Exception("population number must be >= ", SAVE_NET_NUM)

    nn_param_choices = {
        "hidden_layer_sizes": [(12), (12, 12), (12,12,12),
        (50), (50, 50), (50,50,50),
        (100), (100, 100), (100, 100, 100)], 
        "max_iter": [6000, 7000, 8000, 9000], 
        "alpha": [0.001, 0.0001, 0.00001],
        "solver": ['lbfgs', 'sgd', 'adam'], # The solver for weight optimization.
        "activation" : ['identity', 'logistic', 'tanh', 'relu'],
        "tol" : [1e-3, 1e-4, 1e-5, 1e-6],
    }
    
    print("***Evolving %d generations with population %d***" %
                 (GENERATIONS, POPULATION))

    x, y = get_data()
    
    networks = generate(GENERATIONS, POPULATION, nn_param_choices, x, y)
    # Print out the top 5 networks.
    print_networks(networks[:SAVE_NET_NUM])
    save_networks(networks[:SAVE_NET_NUM])
    
if __name__ == '__main__':
    import time

    start = time.time()
    main()
    end = time.time()
    print("!!!!!!!!!!!!", end - start)
    
