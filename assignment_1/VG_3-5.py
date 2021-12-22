import pandas as pd
from sko.GA import GA_TSP
from itertools import product
import time
import multiprocessing as mp
from tqdm import tqdm
import os


def create_distance_matrix(df):
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

    return distance_matrix


def genetic_tsp_variant(distance_matrix, num_points, max_iter, prob_mut, queue=None):
    def cal_total_distance(routine):
        '''The objective function. input routine, return total distance.
        cal_total_distance(np.arange(num_points))
        '''
        num_p, = routine.shape
        return sum([distance_matrix[routine[i % num_p], routine[(i + 1) % num_p]] for i in range(num_p)])
    
    ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=max_iter, prob_mut=prob_mut)
    ga_tsp.run()

    result = (ga_tsp.best_x, ga_tsp.best_y, (num_points, max_iter, prob_mut))

    if queue:
        queue.put(result)
    
    return result


def calc_with_normal(variations):
    results = []
    start = time.time()
    for variation in tqdm(variations):
        result = genetic_tsp_variant(*variation)
        results.append(result)


    total = time.time() - start
    results = sorted(results, key=lambda x: x[1]) 
    print(f'Took total time of {total} seconds.')
    print(f'Best Result was: {results[0][1]} (num_points: {results[0][2][0]} max_iter: {results[0][2][1]} prob_mut:{results[0][2][2]})')
    print(f'Worst Result was: {results[-1][1]} (num_points: {results[-1][2][0]} max_iter: {results[-1][2][1]} prob_mut:{results[-1][2][2]})')

def calc_with_pooling(variations):
    results = []
    start = time.time()
    n_cores=mp.cpu_count()

    pool = mp.Pool(n_cores)
    results = pool.starmap(genetic_tsp_variant, variations)
    pool.close()

    total = time.time() - start
    results = sorted(results, key=lambda x: x[1]) 
    print(f'Took total time of {total} seconds.')
    print(f'Best Result was: {results[0][1]} (num_points: {results[0][2][0]} max_iter: {results[0][2][1]} prob_mut:{results[0][2][2]})')
    print(f'Worst Result was: {results[-1][1]} (num_points: {results[-1][2][0]} max_iter: {results[-1][2][1]} prob_mut:{results[-1][2][2]})')


def calc_with_mprocess(variations):
    start = time.time()
    results = []
    processes = []
    queue = mp.Queue()

    for variant in variations:
        p = mp.Process(target=genetic_tsp_variant, args=(*variant, queue))
        processes.append(p)
        p.start()


    for process in processes:
        process.join()

    results = [queue.get() for _ in processes]

    total = time.time() - start
    results = sorted(results, key=lambda x: x[1]) 
    print(f'Took total time of {total} seconds.')
    print(f'Best Result was: {results[0][1]} (num_points: {results[0][2][0]} max_iter: {results[0][2][1]} prob_mut: {results[0][2][2]})')
    print(f'Worst Result was: {results[-1][1]} (num_points: {results[-1][2][0]} max_iter: {results[-1][2][1]} prob_mut: {results[-1][2][2]})')

def main():
    p = os.path.join('assignment_1', 'data', 'distanceslonglat.csv')
    # Load data
    df = pd.read_csv(p)
    # Create distance matrix
    distance_matrix = create_distance_matrix(df)

    # prepare variations
    num_points = [len(df['Start'].unique())]
    max_iterations = [100, 500, 1000]
    prob_mutations = [0.001, 0.01, 0.05]
    variations = list(product(num_points, max_iterations, prob_mutations))
    variations = [list(v) for v in variations]
    for v in variations:
        v.insert(0, distance_matrix)

    # Run tests.

    # normal
    print('Normal')
    calc_with_normal(variations)

    # pooling
    print('Pool')
    calc_with_pooling(variations)

    # mprocess
    print('Process')
    calc_with_mprocess(variations)

if __name__ == '__main__':
    mp.freeze_support()
    main()
