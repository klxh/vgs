#include <stdio.h>
#include <stdlib.h>

#define vegas_cycles 1 // numero di iterazioni vegas
#define dim 2 // numero di variabili della funzione integranda
#define Nc 10 // numero di suddivisioni degli intervalli di integrazione


// definizione della funzione integranda
double f(double r[dim])
{
	return r[0] + r[1];	
}


int main ()
{
	// definizione della griglia di integrazione e inizializzazione con equispaziature
	double grid[dim * (Nc + 1)];
	for(int i = 0; i < dim; i++)
	{
		grid[i * (Nc + 1)] = 0;
		printf("grid[%d][%d] = %f\n", i, 0, grid[i * (Nc + 1)]);
		for(int j = 1; j < Nc + 1; j++)
		{
			grid[i * (Nc + 1) + j] = grid[i * (Nc + 1) + j - 1] + 1./Nc;
		  printf("grid[%d][%d] = %f\n", i, j, grid[i * (Nc + 1) + j]);
		}
	}

	// definizione della griglia delle spaziature
	double spacings[dim * Nc];
	for(int i = 0; i < dim * Nc; i++) { spacings[i] = 1./Nc; }


	// cicli vegas
	for (int it = 0; it < vegas_cycles; it++)
	{
		printf("###### ITERAZIONE VEGAS %d ######\n\n", it);
	}
	
	return 0;
}
