# Archivo principal de la implementación con Pythran y OpenMP

from util import MonitoringThread, show_chessboard
import genetic_algorithm_pythran
import time

if __name__ == '__main__':
    start = time.time()
    t = MonitoringThread()
    t.start()
    solution, _ = genetic_algorithm_pythran.train_ga(10000, 1000)
    print("Solution found:", solution)
    end = time.time()
    print('Total execution time:', end - start)

    t.stop()

    if len(solution) > 0:
        show_chessboard(solution)
