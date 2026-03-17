question preliminaire: car la galaxie est plate
on peut donc prends nk = 1 car 

affichage : 3ms: algorithme: ~1600 ms, pas besoin de paralleliser l'algorithme

numba:
@njit(parallel=True)
ajout de prange sur touts les range(nbodies) - line 51 etc
1thread = 320
2: 170