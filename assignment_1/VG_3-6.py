from sko.ACA import ACA_TSP
import pandas as pd
import numpy as np
from scipy import spatial
import multiprocessing as mp
from tqdm import tqdm
import time

df = pd.read_csv('assignment_1\data\distanceslonglat.csv')
cities = df['Start'].unique()

cities_index = dict(zip(cities, range(len(cities))))
df['Start'] = [cities_index[city] for city in df['Start']]
df['Target'] = [cities_index[city] for city in df['Target']]
start = df['Start'].to_list()

new_df = pd.DataFrame(columns=set(start))

for i in start:
    rows = df.loc[df['Start'] == i]
    distances = rows['Distance'].to_list()
    new_df.loc[i] = distances

distance_matrix = new_df.to_numpy()
distance_matrix = distance_matrix.astype('float64')


def aca_tsp(max_iter):
    aca = ACA_TSP(func=cal_total_distance, n_dim=len(set(start)),
                  size_pop=50, max_iter=max_iter,
                  distance_matrix=distance_matrix)
    aca.run()

    return str(aca.best_y)


def cal_total_distance(routine):
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])


def main():

    start = time.time()

    n_cores = mp.cpu_count()
    pool = mp.Pool(n_cores)
    max_iterations = [100, 250, 500]
    result = pool.map(aca_tsp, max_iterations)
    pool.close()

    total_time = time.time() - start
    print(total_time)
    print(result)


if __name__ == '__main__':
    mp.freeze_support()
    main()
