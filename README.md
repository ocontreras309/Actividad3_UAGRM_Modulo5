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

Escenario | Comando
--- | ---
Algoritmo genético sin concurrencia | python genetic_algorithm_no_concurrency.py
Algoritmo genético con Pythran y OpenMP | python run_ga_pythran.py
Algoritmo genético con PyKKA | python genetic_algorithm_pykka.py
