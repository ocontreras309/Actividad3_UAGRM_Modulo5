# Archivo principal de la implementación con Pythran y OpenMP

from util import MonitoringThread, show_chessboard
import genetic_algorithm_pythran
import time

if __name__ == '__main__':
    start = time.time()
    t = MonitoringThread()
    t.start()
    solution, history = genetic_algorithm_pythran.train_ga(1000000, 1000)
    print("Solution found:", solution)
    print("Fitness history length:", len(history))
    end = time.time()
    print(end - start)

    t.stop()

    if len(solution) > 0:
        show_chessboard(solution)
