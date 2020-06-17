#include <stdio.h>
#include <stdlib.h>

#define vegas_cycles 1
#define dim 2
// definizione della funzione integranda
float f(float r[dim])
{
	return r[0] + r[1];	
}


int main ()
{
	// cicli vegas
	for (int it = 0; it < vegas_cycles; it++)
	{
		printf("###### ITERAZIONE VEGAS %d ######\n\n", it);
		float r[2] = {1, 2};
		printf("f(1, 2) = %f\n", f(r));
	}
	
	return 0;
}
