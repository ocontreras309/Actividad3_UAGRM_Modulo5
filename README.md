# Actividad 3: Concurrencia y paralelismo

Se requiere un sistema operativo Linux para compilar y ejecutar los ejercicios. Se recomienda realizar el proceso en un entorno virtual de Python.

Para realizar la instalación de todos los requisitos:

```
pip install -r requirements.txt
```

Para compilar el ejercicio de Pythran:

```
pythran -DUSE_XSIMD -fopenmp genetic_algorithm_pythran.py -o genetic_algorithm_pythran.so
```

Para ejecutar cada programa:

Escenario | Comando | Descripción
--- | ---
Algoritmo genético sin paralelismo | python genetic_algorithm_no_concurrency.py | No contiene optimizaciones ni uso de hilos. Se tomó como base
Algoritmo genético con Pythran y OpenMP | python run_ga_pythran.py | Implementa optimizaciones a nivel de código máquina y también paralelismo a través de OpenMP
Algoritmo genético con PyKKA | python genetic_algorithm_pykka.py | Implementa Pykka basado en el modelo de actores donde cada fase del algoritmo genético es tratada como un actor.

Cada archivo implementa las funciones de: Generación de la población, fitness, selección, cruzamiento y mutación con características específicas.

En el archivo util.py se tienen funciones utilitarias para la visualización de soluciones y también para obtener los reportes de consumo de CPU y memoria.
