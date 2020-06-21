#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define K 1000
#define dim 8
#define alpha 1.5

#define vegas_cycles 3
#define Nc 5
#define N 100
const int Nh = pow(Nc, dim);


// definizione della funzione integranda
double f(double r[dim])
{
  return exp(-50 * pow((r[0] - 0.5), 2)) / sqrt(2 * M_PI * 0.01) * exp(-50 * pow((r[1] - 0.5), 2)) / sqrt(2 * M_PI * 0.01)
  * exp(-50 * pow((r[2] - 0.5), 2)) / sqrt(2 * M_PI * 0.01) * exp(-50 * pow((r[3] - 0.5), 2)) / sqrt(2 * M_PI * 0.01)
  * exp(-50 * pow((r[4] - 0.5), 2)) / sqrt(2 * M_PI * 0.01) * exp(-50 * pow((r[5] - 0.5), 2)) / sqrt(2 * M_PI * 0.01)
  * exp(-50 * pow((r[6] - 0.5), 2)) / sqrt(2 * M_PI * 0.01) * exp(-50 * pow((r[7] - 0.5), 2)) / sqrt(2 * M_PI * 0.01);
//  * exp(-50 * pow((r[8] - 0.5), 2)) / sqrt(2 * M_PI * 0.01) * exp(-50 * pow((r[9] - 0.5), 2)) / sqrt(2 * M_PI * 0.01);
}



int main()
{
  int s = time(NULL);
  srandom(s);

  double grid[dim][Nc + 1]; for (int i = 0; i < dim; i++) { for (int j = 0; j < Nc + 1; j++) { grid[i][j] = 0; } }
  for(int i = 0; i < dim; i++)
  {
    grid[i][0] = 0;
    printf("grid[%d][0] = %f\n", i, grid[i][0] );
    for(int j = 1; j < Nc + 1; j++)
    {
      grid[i][j] += grid[i][j - 1] + 1./Nc;
      printf("grid[%d][%d] = %f\n", i, j, grid[i][j] );
    }
  }

  double spacings[dim][Nc];
  for(int i = 0; i < dim; i++) { for(int j = 0; j < Nc; j++) { spacings[i][j] = 1./Nc; } }

  double I[vegas_cycles] = {0}; // salva valore integrali per ogni iterazione
  double E[vegas_cycles] = {0}; // salva varianza per ogni iterazione

  double diff = 0;

  for (int iter = 0; iter < vegas_cycles; iter++)
  {
    clock_t start = clock();
    
    printf("##############################\nIterazione di VEGAS # %d\n\n", iter + 1);

    double abs_S = 0;
    double partial_S[Nh]; for (int i = 0; i < Nh; i++) { partial_S[i] = 0; }
    double partial_abs[Nh]; for (int i = 0; i < Nh; i++) { partial_abs[i] = 0; }

    for(int i = 0; i < Nh; i++)
    {
      double quads = 0;
      double hypervolume = 0;
      for(int count = 0; count < N; count++)
      {
        int index[dim];
        int buffer[dim];
        index[0] = i % Nc;
        buffer[0] = index[0];
        double r[dim];
        r[0] = grid[0][index[0]] + (double) random() / RAND_MAX * (grid[0][index[0] + 1] - grid[0][index[0]]);
        hypervolume = grid[0][index[0] + 1] - grid[0][index[0]];
        for(int dir = 1; dir < dim; dir++)
        {
          buffer[dir] = buffer[dir - 1] + (i - buffer[dir - 1]) % (int) pow(Nc, dir + 1);
          index[dir] = ( (i - buffer[dir - 1]) % (int) pow(Nc, dir + 1) ) / pow(Nc, dir);
          r[dir] = grid[dir][index[dir]] + (double) random() / RAND_MAX * (grid[dir][index[dir] + 1] - grid[dir][index[dir]]);
          hypervolume *= (grid[dir][index[dir] + 1] - grid[dir][index[dir]]);
        }
        partial_S[i] += f(r) * hypervolume / N;
        partial_abs[i] += abs(f(r)) * hypervolume / N;
        quads += f(r) * f(r);
      }
      double partial_err = pow(hypervolume, 2) / (N * (N - 1)) * quads - 1./N * pow(partial_S[i], 2) ;
      I[iter] += partial_S[i];
      abs_S += partial_abs[i];
      E[iter] += partial_err ;
    }
		printf("Integrale = %f\n", I[iter]);
		printf("Errore quadratico = %f\n", E[iter]);
    printf("\n##############################\n");

    
    double pdfs[dim][Nc]; for (int i = 0; i < dim; i++) { for (int j = 0; j < Nc; j++) { pdfs[i][j] = 0; } };
    for(int i = 0; i < Nh; i++)
    {
      int index[dim];
      int buffer[dim];
      index[0] = i % Nc;
      buffer[0] = index[0];
      pdfs[0][index[0]] += partial_abs[i] / abs_S;
      for(int dir = 1; dir < dim; dir++)
      {
        buffer[dir] = buffer[dir - 1] + (i - buffer[dir - 1]) % (int) pow(Nc, dir + 1);
        index[dir] = ( (i - buffer[dir - 1]) % (int) pow(Nc, dir + 1) ) / pow(Nc, dir);
        pdfs[dir][index[dir]] += partial_abs[i] / abs_S;
      }
    }

    // grid update
    for(int direction = 0; direction < dim; direction ++)
    {
      double M = 0;
      double m[Nc];
      for(int i = 0; i < Nc; i++)
      {
        double r = pdfs[direction][i];
        m[i] = K * pow( (r - 1) / log(r), alpha );
        if(m[i] == 0) { m[i] = 1; }
        M += m[i];
      }

      int L = floor (M / Nc) * Nc;
      int l[Nc];
      int sum = 0;
      for(int elem = 0; elem < Nc; elem++) { l[elem] = floor (m[elem]); sum += l[elem]; }

      if(sum < L)
      {
        int D = L - sum;
        while(D != 0)
        {
          int q = random() % Nc;
          l[q]++;
          D--;
        }
      }

      else if(sum > L)
      {
        int D = sum - L;
        while(D != 0)
        {
          int q = random() % Nc;
          l[q]--;
          D--;
        }
      }

      // ora devo aggiornare la griglia di integrazione lungo l'asse d
      double diffs[Nc];
      for(int j = 0; j < Nc; j++)
      {
        diffs[j] = spacings[direction][j] / l[j];
        spacings[direction][j] = 0;
      }

      int p = 0;
      double count = 0;
      for(int k = 0; k < Nc; k++)
      {
        for(int a = 0; a < L/Nc; a++)
        {
          l[p]--;
          spacings[direction][k] += diffs[p];
          if (l[p] == 0) { p++; }
        }
        count += spacings[direction][k]; // risommo i sottointervalli (check)
				printf("spacings[%d][%d] = %f\n", direction, k, spacings[direction][k]);
      }
			printf("\ncount = %f\n##########################\n", count);
    }

    for(int i = 0; i < dim; i++)
    {
      grid[i][0] = 0;
			printf("\ngrid[%d][0] = %f\n", i, grid[i][0]);
      for(int j = 1; j < Nc + 1; j++)
      {
        grid[i][j] = 0;
        grid[i][j] += grid[i][j - 1] + spacings[i][j - 1];
				printf("grid[%d][%d] = %f\n", i, j, grid[i][j]);
      }
      printf("\n##############################\n");
    }

    clock_t end = clock();
    diff += (double) (end - start) / CLOCKS_PER_SEC;
  }

  // stima globale dell'integrale e dell'errore; calcolo del chi2
  double I_avg = 0;
  double inv = 0;
  double chi2 = 0;
  for(int i = 0; i < vegas_cycles; i++) { inv += 1. / E[i]; }
  double E_avg = 1. / inv;
  for (int i = 0; i < vegas_cycles; i++) { I_avg += E_avg * I[i] / E[i]; }
  for(int i = 0; i < vegas_cycles; i++) { chi2 += pow( (I[i] - I_avg) , 2) / E[i]; }

  printf("Stima globale dell'integrale = %f\n", I_avg);
  printf("Stima globale dell'errore = %f\n", sqrt(E_avg));
  printf("chi quadro = %f\n", chi2);
  printf("chi quadro per iterazione (meno uno) = %f\n", chi2 / (vegas_cycles - 1));

  printf("Elapsed time = %f\n", diff);


	return 0;
}
