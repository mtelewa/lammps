/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://lammps.sandia.gov/, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Tobias Kliesch(KIT)
------------------------------------------------------------------------- */

#include "fix_nh_trigo.h"

#include "atom.h"
#include "domain.h"
#include "error.h"
#include "force.h"

#include "update.h"
#include "input.h"
#include "modify.h"
#include "variable.h"
#include "memory.h"
#include "bond.h"
#include "neighbor.h"
#include "comm.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace FixConst;

extern "C"
{
extern void dsteqr_(char*, int*, double*, double*, double*, int* ,double* ,int*);
}

/* ---------------------------------------------------------------------- */

FixNHTrigo::FixNHTrigo(LAMMPS *lmp, int narg, char **arg) :
  FixNH(lmp, narg, arg)
{  
}

int FixNHTrigo::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= POST_NEIGHBOR;
  mask |= FINAL_INTEGRATE;
  //mask |= THERMO_ENERGY;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  if (pre_exchange_flag) mask |= PRE_EXCHANGE;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixNHTrigo::init()
{
  FixNH::init();
}

/* ----------------------------------------------------------------------
   perform half-step update of rotational velocities
-----------------------------------------------------------------------*/

void FixNHTrigo::setup(int /*vflag*/)
{
  double **f = atom->f;
  int *type = atom->type;
  double *mass = atom->mass;
  int nlocal = atom->nlocal, dim = 3;

  double delta_t = sqrt(force->ftm2v) * update->dt;
  int tmp;
  int i, n, d; 

  FixNH::setup(i);

  for(n = 0; n < 3; n++) {
    for (d = 0; d < 2; d++) {
      hetProcStrcs[n][d].maxDiagRecv=0;
      hetProcStrcs[n][d].maxDiagSend=0;
      hetProcStrcs[n][d].maxSend=0;
      hetProcStrcs[n][d].maxRecv=0;
      hetProcStrcs[n][d].maxNtv=0;
      hetProcStrcs[n][d].maxFrn=0;
      hetProcStrcs[n][d].maxPckdRecv=0;
    }
  }

  maxHet = 0;
  maxHess = 0;
  maxLoc = 0;

  setMemory();
  update_hetComm();

  memory->create(forceVec, 3 * nlocal, "trigonometric:forceVec");
  memory->create(forceOut, 3 * nlocal, "trigonometric:forceOut");
  
  tmp = updateHessian();

  MPI_Allreduce(&tmp, &nRho, 1, MPI_INT, MPI_MAX, world);
  maxRho = nRho;
  
  createStrc(&forceKry);
}

void FixNHTrigo::nve_v()
{
  // standard nve_v velocity update
  double **v = atom->v;
  int *type = atom->type;
  double *mass = atom->mass;
  int nlocal = atom->nlocal, dim = 3;
  double **f = atom->f;

  double delta_t = sqrt(force->ftm2v) * update->dt;

  int i, d;
  int tmp;

  for (i = 0; i < nlocal; i++) 
    for (d = 0; d < dim; d++) 
      forceVec[index(i,d)] = f[i][d];// / sqrt(mass[type[i]]);

  tmp = updateHessian();
  MPI_Allreduce(&tmp, &nRho, 1, MPI_INT, MPI_MAX, world);

  if (maxRho < nRho) {
    if (maxRho > 0) {
      memory->destroy(forceKry.matV);
      memory->destroy(forceKry.alpha);
      memory->destroy(forceKry.beta);    
    }
    memory->create(forceKry.matV, nRho, nlocal * 3, "trigonometric:KrylovBase");
    memory->create(forceKry.alpha, nRho, "trigonometric:Alpha");
    memory->create(forceKry.beta, nRho, "trigonometric:Beta");    
  
    maxRho = nRho;
  } 

  tmp = dim * nlocal;
  lanczosImplPara(forceVec, &forceKry, tmp);  

  if(forceKry.kDim > 0) {
    //Evaluate functions
    matFuncPara(&forceKry, delta_t, 3, tmp, forceOut);  //psi

    //Set new force
    
    for (i = 0; i < nlocal; i++)
      for (d = 0; d < dim; d++)
        f[i][d] = forceOut[index(i,d)];
    
  }


  for (i = 0; i < nlocal; i++) {
    for (d = 0; d < dim; d++) {
      v[i][d] *= sqrt(mass[type[i]]); 
      f[i][d] *= sqrt(mass[type[i]]);  
    }
  }

  FixNH::nve_v();

  for (i = 0; i < nlocal; i++) {
    for (d = 0; d < dim; d++) {
      v[i][d] /= sqrt(mass[type[i]]);  
      f[i][d] /= sqrt(mass[type[i]]);  
    }
  }
  
}

/* ----------------------------------------------------------------------
   perform full-step update of position with dipole orientation, if requested
-----------------------------------------------------------------------*/

void FixNHTrigo::nve_x()
{
  double **x = atom->x;
  double **v = atom->v;
  int *type = atom->type;
  double *mass = atom->mass;
  int nlocal = atom->nlocal, dim = 3;
  int i ,d;

  for (i = 0; i < nlocal; i++) {
    for (d = 0; d < dim; d++) {   
      x[i][d] *= sqrt(mass[type[i]]);
      v[i][d] *= sqrt(mass[type[i]]);
    }
  }

  FixNH::nve_x();

  for (i = 0; i < nlocal; i++) {
    for (d = 0; d < dim; d++) {
      x[i][d] /= sqrt(mass[type[i]]);
      v[i][d] /= sqrt(mass[type[i]]);   
    }
  }

}
      
void FixNHTrigo::post_neighbor() {
  int nlocal = atom->nlocal;
  setMemory();
  update_hetComm();

  if (nlocal > maxLoc) {
    if (maxLoc > 0) {
      memory->destroy(forceVec);
      memory->destroy(forceOut);
      if (maxRho > 0) memory->destroy(forceKry.matV);
    }
    memory->create(forceKry.matV, nRho, nlocal * 3, "trigonometric:KrylovBase");
    memory->create(forceVec, 3 * nlocal, "trigonometric:forceVec");
    memory->create(forceOut, 3 * nlocal, "trigonometric:forceOut");

    maxLoc = nlocal;
  }
}


void FixNHTrigo::matVecImpl(double* in, double* out) {
  //Grab Some Information
  int i, n, d, dm, b;
  int i1, i2;
  double tmp11, tmp22, tmp12;

  double *m = atom->mass;
  int *aType = atom->type;
  int **bondlist = neighbor->bondlist;
  int nlocal = atom->nlocal;
  int dim = 3;

  commVec(in);

  //printVec(in, dim * nlocal);

  for (i = 0; i < nlocal; i++)
    for (d = 0; d < dim; d++)
      out[index(i, d)] = .0;

  //only nHess local bonds in system.
  for (n = 0; n < nHess; n++) {

    i1 = homBondList[n][0];
    i2 = homBondList[n][1];

    // Left right mult of sqrt(M^{-1})
    tmp11 = m[aType[i1]];
    tmp22 = m[aType[i2]];
    tmp12 = sqrt(tmp11) * sqrt(tmp22);

    for (d = 0; d < dim; d++) {
      for (b = 0; b < dim; b++) {
        out[index(i1,d)] += (implHessian[n][d * dim + b] / tmp12) * in[index(i2,b)];
        out[index(i1,d)] -= (implHessian[n][d * dim + b] / tmp11) * in[index(i1,b)];
        //Symmetry in all things
        out[index(i2,b)] += (implHessian[n][d * dim + b] / tmp12) * in[index(i1,d)];
        out[index(i2,b)] -= (implHessian[n][d * dim + b] / tmp22) * in[index(i2,d)];
      }
    }
  }

  for (i = 0; i < 3; i++) {
    for (dm = 0; dm < 2; dm++) {
      for (n = 0; n < hetProcStrcs[i][dm].nShrdHet; n++) {
        i1 = hetProcStrcs[i][dm].i1[n];
        i2 = hetProcStrcs[i][dm].i2[n];

        // Left right mult of sqrt(M^{-1})
        tmp11 = m[aType[i1]];
        tmp22 = m[aType[i2]];
        tmp12 = sqrt(tmp11) * sqrt(tmp22);

        for (d = 0; d < dim; d++) {
          for (b = 0; b < dim; b++) {
            //i1 always local - i2 always ghost.
            out[index(i1,d)] += (hetProcStrcs[i][dm].shrdHessian[n][d * dim + b] / tmp12) * hetProcStrcs[i][dm].hetRecv[index(n,b)];
            out[index(i1,d)] -= (hetProcStrcs[i][dm].shrdHessian[n][d * dim + b] / tmp11) * hetProcStrcs[i][dm].hetSend[index(n,b)];
          }
        }
      }
    }
  }
}

void FixNHTrigo::valVecProd(double a, double* v, double* out, int n) {
  for (int i = 0; i < n; i++)
    out[i] = a * v[i];
}

void FixNHTrigo::vecVecSub(double* a, double* b, double* out, int n) {
  for (int i = 0; i < n; i++)
    out[i] = a[i] - b[i];
}

double FixNHTrigo::vecVecProdPara(double* inA, double* inB, int n) {
  double prodGlob = 0.0;
  double prodLoc = 0.0;

  for (int i = 0; i < n; i++) 
    prodLoc += (inA[i] *  inB[i]);

  MPI_Allreduce(&prodLoc, &prodGlob, 1, MPI_DOUBLE, MPI_SUM, world);

  return prodGlob;
}

double FixNHTrigo::calcNormPara(double* vec, int n) {
  double sumLoc = 0.0;
  double sumGlob = 0.0;

  for (int i = 0; i < n; i++) 
    sumLoc += sqr(vec[i]);
  
  MPI_Allreduce(&sumLoc, &sumGlob, 1, MPI_DOUBLE, MPI_SUM, world);
  sumGlob = sqrt(sumGlob);

  return sumGlob;
}

void FixNHTrigo::lanczosImplPara(double *b, KrylovStrc *strc, int n) {
  int counter = 0, i, j, m;
  double *mat_v, *a_v, *b_v;

  strc->norm = calcNormPara(b, n);
  if (strc->norm > 0) {
    mat_v = new double[n];
    a_v = new double[n];
    b_v = new double[n];

    for (i = 0; i < n; i++) 
      strc->matV[0][i] = b[i] / strc->norm;

    matVecImpl(strc->matV[0], mat_v);
    strc->alpha[0] = vecVecProdPara(mat_v, strc->matV[0], n);

    for (i = 0; i < n; i++) 
      strc->matV[1][i] = mat_v[i] - strc->alpha[0] * strc->matV[0][i];

     for (m = 1; m < nRho - 1; m++) {
       //getchar();
      counter++;
      strc->beta[m - 1] = calcNormPara(strc->matV[m], n);
      if (strc->beta[m - 1] < 0.001) break;

      for (i = 0; i < n; i++) 
        strc->matV[m][i] /= strc->beta[m-1];

      matVecImpl(strc->matV[m], mat_v);

      strc->alpha[m] = vecVecProdPara(mat_v, strc->matV[m], n);
      
      //subtraction in 1 step
      for (i = 0; i < n; i++)
        strc->matV[m + 1][i] = mat_v[i] - strc->alpha[m] * strc->matV[m][i] - strc->beta[m - 1] * strc->matV[m - 1][i];
    } 

    delete [] mat_v;
    delete [] a_v;
    delete [] b_v;
  }
  strc->kDim = counter;
}

void FixNHTrigo::createStrc(KrylovStrc *strc) {
  maxLoc = atom->nlocal;// + atom->nghost;
  strc->kDim = 0;
  memory->create(strc->alpha, nRho, "trigonometric:Alpha");
  memory->create(strc->beta, nRho, "trigonometric:Beta");
  memory->create(strc->matV, nRho, maxLoc * 3, "trigonometric:KrylovBase");
}

void FixNHTrigo::clearStrc(KrylovStrc *strc) {
  //free krylov basis
  if(maxRho > 0) {
    memory->destroy(strc->alpha);
    memory->destroy(strc->beta);
    if(maxLoc >0) memory->destroy(strc->matV);
  }
}

double facNH(double x) {
  if (x > 1) {
    return x * facNH(x - 1);
  } else {
    return 1;
  }
}

double evaluatePsiNH(double value, double delta_t, int taylorLV) {
  double tmpVal, sum;
  if (fabs((sqr(delta_t) / 4.0) * value) > 0.001) {
    if(value > 0) {				
      value = sqrt(value);
      value = sin((delta_t / 2.0) * value) / ((delta_t / 2.0) * value);
    } else {
      value = sqrt(fabs(value));
      value = sinh((delta_t / 2.0) * value) / ((delta_t / 2.0) * value);
    }
    value *= value;
  } else {
    sum = 1;
    tmpVal = fabs((sqr(delta_t) / 4.0) * value);
    for (int j = 1; j < taylorLV; j++) {
      if (j % 2 == 0) {
        sum += tmpVal / facNH(2.0 * j + 1.0);
      }
      else {
        sum -= tmpVal / facNH(2.0 * j + 1.0);
      }
      tmpVal *= fabs((sqr(delta_t) / 4.0) * value);
    }
    value = sqr(sum);
  }

  return value;
}

double evaluatePsi_0NH(double value, double delta_t, int taylorLV) {
  if (value> 0) {
    value = cos(delta_t * sqrt(value));
  } else {
    value = cosh(delta_t * sqrt(-value));
  }
  return value;
}

double evaluatePsi_1NH(double value, double delta_t, int taylorLV) {
  return 1.0;
}

double evaluatePhiNH(double value, double delta_t, int taylorLV) {
  if (fabs(value) > 0.001) {
    if(value > 0) {				
      value = sqrt(value);
      value = (sin(delta_t * value) / (delta_t * value)) * (1.0 + (1.0 / 3.0) * sqr(sin((delta_t / 2.0) * value)));
    } else {
      value = sqrt(-value);
      value = (sinh(delta_t * value) / (delta_t * value)) * (1.0 + (1.0 / 3.0) * sqr(sinh((delta_t / 2.0) * value)));
    }	
  } else {
    value = 1 - ((sqr(delta_t)) * value)/12.0 - sqr((sqr(delta_t)) * value)/80.0 + (19.0 * cube((sqr(delta_t)) * value))/10080.0;
  }

  return value;
}

void evaluateFuncNH(double delta_t, double* eval, int func, int dim) {
  int taylorLV = 5;
  double tmpVal, sum;
  
  //0: cos 1:sinc 2:(x/h)*sin 3:psi 4:psi_0 5:psi_1 6: phi
  switch(func) {
    case 0:
      for (int i = 0; i < dim; i++) { 
        if (eval[i] > 0) {
          eval[i] = cos(delta_t * sqrt(eval[i]));
        } else {
          eval[i] = cosh(delta_t * sqrt(-eval[i]));
        }
      };
    break;
    case 1:
      for (int i = 0; i < dim; i++) {
        if (fabs(eval[i]) > 0.001) {
          if(eval[i] > 0) {
            eval[i] = sqrt(eval[i]);
            eval[i] = sin(delta_t * eval[i]) / (delta_t * eval[i]);
          } else {
            eval[i] = sqrt(-eval[i]);
            eval[i] = sinh(delta_t * eval[i]) / (delta_t * eval[i]);
          }
        } else {
          sum = 1;
          tmpVal = fabs((sqr(delta_t)) * eval[i]);
          for (int j = 1; j < taylorLV; j++) {
            if (j % 2 == 0) {
              sum += tmpVal / facNH(2.0 * j + 1.0);
            } else {
              sum -= tmpVal / facNH(2.0 * j + 1.0);
            }
          }
          eval[i] = sum;
        }
      };
    break;
    case 2:
      for (int i = 0; i < dim; i++) {
        if(eval[i] > 0) {
          eval[i] = sqrt(eval[i]);
          eval[i] = eval[i] * sin(delta_t * eval[i]);
        } else {
          //Complex value - lets not talk about this .. for now
          eval[i] = sqrt(-eval[i]);
          eval[i] = eval[i] * sinh(delta_t * eval[i]);
        }

      };
    break;
    case 3:
      for (int i = 0; i < dim; i++) {
        eval[i] = evaluatePsiNH(eval[i], delta_t, taylorLV);
      };
    break;
    case 4:
      for (int i = 0; i < dim; i++) {
        eval[i] = evaluatePsi_0NH(eval[i], delta_t, taylorLV);
      };
    break;
    case 5:
      for (int i = 0; i < dim; i++) {
        eval[i] = evaluatePsi_1NH(eval[i], delta_t, taylorLV);
      };
    break;
    case 6:
      for (int i = 0; i < dim; i++) {
        eval[i] = evaluatePhiNH(eval[i], delta_t, taylorLV);
      };
    break;
    default:
      printf("FAULTY -func- FAULT\n");
    break;
  }
}

void FixNHTrigo::matFuncSmaller(double *alpha, double *beta, int aDim, int func, double delta_t, double *out) {

  // Locals 
	int n = aDim, lda = aDim, info, lwork;
	double wkopt, sumc;
	double *work, *d, *e, *z;
  int i, k;

  d = new double[n];
  e = new double[n];
  work = new double[2*n - 2];
  z = new double[aDim * aDim];

	for (i = 0; i < n - 1; i++) {
    d[i] = alpha[i];
    e[i] = beta[i];
	}
  d[n - 1] = alpha[n - 1]; 
	// Query and allocate the optimal workspace 
  char compz[] = "I"; // eigenvalues and eigenvectors
  dsteqr_(compz, &n, d, e, z, &aDim, work, &info);
	// Check for convergence 
	if( info > 0 ) {
		printf( "The algorithm failed to compute eigenvalues.\n" );
		exit( 1 );
	}
  
  // Apply function to Eigenvalue
  evaluateFuncNH(delta_t, d, func, n);  

  for (i = 0; i < n; i++) {
    sumc = 0;
    for (k = 0; k < n; k++) {
      sumc += z[i + k * aDim] * d[k] * z[k * aDim];
    }
    out[i] = sumc;
	}

  delete [] d;
  delete [] e;
  delete [] work;
  delete [] z;
}

void FixNHTrigo::matFuncPara(KrylovStrc *strc, double delta_t, int func, int n, double* out) {
  int dim = 3, nlocal = atom->nlocal;
  double *matOut = new double[strc->kDim];
  double sumLoc;
  int i,k;
  
  if(comm->me == 0) {
    matFuncSmaller(strc->alpha, strc->beta, strc->kDim, func, delta_t, matOut);
  } 

  MPI_Bcast(matOut, strc->kDim, MPI_DOUBLE, 0, world);
  for (i = 0; i < n; i++) {
    sumLoc = 0;
   for (k = 0; k < strc->kDim; k++) {
      sumLoc += strc->matV[k][i] * matOut[k];
    }
    out[i] = strc->norm * sumLoc; 
  }

  delete [] matOut;
}

void FixNHTrigo::printMat(double** matIn, int n, int m) {
	printf(" ----------MATRIX---------- \n");
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			printf("%f ", matIn[i][j]);
		}
		printf("\n");
	}
	printf(" -------------------------- \n");
}

int FixNHTrigo::findProc(double *x) {
  int ret = 0;
  int igx, igy, igz; //Dont think I need you guys
  double *tmpX;
  tmpX = new double[3];

    // HATE IT
  
  for (int d = 0; d < 3; d++) {
    tmpX[d] = x[d];

    if (tmpX[d] > domain->boxhi[d]) {
      tmpX[d] -= (domain->boxhi[d] - domain->boxlo[d]);
    }
    if (tmpX[d] < domain->boxlo[d]) {
      tmpX[d] += (domain->boxhi[d] - domain->boxlo[d]);
    }
  }

  ret = comm->coord2proc(tmpX, igx, igy, igz);
  delete [] tmpX;
  return ret;
}

void FixNHTrigo::assignHetBonds(double *x, int i, int *storeProc) {
  int proc = 0;
  int igx, igy, igz;
  double *tmpX = new double[3];
  int *tmpDiff = new int[3];
  
  //TAKE PERIODIC BOUNDARY INTO ACCOUNT FOR POSITION
  for (int d = 0; d < 3; d++) {
    tmpX[d] = x[d];

    if (tmpX[d] > domain->boxhi[d]) {
      tmpX[d] -= (domain->boxhi[d] - domain->boxlo[d]);
    } else if (tmpX[d] < domain->boxlo[d]) {
      tmpX[d] += (domain->boxhi[d] - domain->boxlo[d]);
    }
  }

  proc = comm->coord2proc(tmpX, igx, igy, igz);

  tmpDiff[0] = igx - comm->myloc[0];
  tmpDiff[1] = igy - comm->myloc[1];
  tmpDiff[2] = igz - comm->myloc[2];

  //Periodic Boundary for proc grid
  if (tmpDiff[0] < -1) tmpDiff[0] =  1;
  if (tmpDiff[0] >  1) tmpDiff[0] = -1;
  if (tmpDiff[1] < -1) tmpDiff[1] =  1;
  if (tmpDiff[1] >  1) tmpDiff[1] = -1;
  if (tmpDiff[2] < -1) tmpDiff[2] =  1;
  if (tmpDiff[2] >  1) tmpDiff[2] = -1;

  int tmpVal;
  for (int n = 0; n < 3; n++) {
    tmpVal = abs(tmpDiff[(n+1)%3]) + abs(tmpDiff[(n+2)%3]);
    //printf("WHAT\n");
    if(tmpDiff[n] < 0) {
      if (tmpVal == 0) {

        hetProcStrcs[n][0].nMyBonds[0]++;
      } else {

        hetProcStrcs[n][0].nMyBonds[1]++;
      }

      storeProc[4*i+0] = n+3*0;
      storeProc[4*i+1] = tmpDiff[0];
      storeProc[4*i+2] = tmpDiff[1];
      storeProc[4*i+3] = tmpDiff[2];
      break;
    } 
    if (tmpDiff[n] > 0) {
      if (tmpVal == 0) {
        hetProcStrcs[n][1].nMyBonds[0]++;
      } else {
        hetProcStrcs[n][1].nMyBonds[1]++;
      }
      storeProc[4*i+0] = n+3*1;
      storeProc[4*i+1] = tmpDiff[0];
      storeProc[4*i+2] = tmpDiff[1];
      storeProc[4*i+3] = tmpDiff[2];
      break;
    }
  }

  delete []tmpX;
  delete []tmpDiff;
}

void FixNHTrigo::assignBonds(int *storeProc) {
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  double** x = atom->x;
  int nlocal = atom->nlocal;
  
  int c1 = 0, c2 = 0;
  int i1, i2, type;
  tagint i1G, i2G;
  int mppd;

  //WAY TO COMPLICATED AND INEFFICIENT... [Not used too often]
  for (int n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];
    //Only conisder 'local' bonds [not pure ghosts]
    if (i1 < nlocal || i2 < nlocal) {
      //PURE LOCAL
      if (i1 < nlocal && i2 < nlocal) {
        homBondList[c1][0] = i1;
        homBondList[c1][1] = i2;
        homBondList[c1++][2] = type;
      } else {
        //TODO: FOR now easier to read- later only put when need
        i1G = atom->tag[i1];
        i2G = atom->tag[i2];
        if (i2 >= nlocal) {
          mppd = atom->map(i2G);
          //LOCAL BUT OVER BOUNDARY
          if (mppd < nlocal) {
            homBondList[c1][0] = i1;
            homBondList[c1][1] = mppd;
            homBondList[c1++][2] = type;
          } else { // TRUE HETBOND
            hetBondList[c2][0] = i1G;
            hetBondList[c2][1] = i2G;
            hetBondList[c2][2] = type;
            assignHetBonds(x[i2], c2, storeProc);
            c2++;
          }
        } else {
          mppd = atom->map(i1G);
          if (mppd < nlocal) {
            homBondList[c1][0] = mppd;
            homBondList[c1][1] = i2;
            homBondList[c1++][2] = type;
          } else { // TRUE HETBOND
            hetBondList[c2][0] = i1G;
            hetBondList[c2][1] = i2G;
            hetBondList[c2][2] = type;
            assignHetBonds(x[i1], c2, storeProc);
            c2++;
          }
        }
      }
    }
  }
}


int FixNHTrigo::findIndex(int n, int d, int i1, int i2) {
  int ret = 0;

  while(!(i1 == hetProcStrcs[n][d].diagInfoNtv[6 * ret + 3] && i2 == hetProcStrcs[n][d].diagInfoNtv[6 * ret + 4]))
    ret++;

  return ret;
}

void FixNHTrigo::sortDiagInfo(int myN, int myD, bool dir) {
  //JUST RESORT DONT ALLOCATE [Just Grow] Send/Recv YET
  HetProcStrc strc = hetProcStrcs[myN][myD];
  int frnC = 0, ntvC = 0; 
  int i, j, n;

  hetProcStrcs[myN][myD].nDiagRecv = 0;
  for (i = 0; i < strc.nYourBonds[1]; i++) {
    for (n = myN+1; n < 3; n++) {
      if(strc.diagInfoRecv[6 * i + n] == -1) {
        if(++hetProcStrcs[n][0].nFrnDiag > hetProcStrcs[n][0].maxFrn) {
          if(hetProcStrcs[n][0].maxFrn > 0){
            memory->grow(hetProcStrcs[n][0].diagInfoFrn, 6 * hetProcStrcs[n][0].nFrnDiag, "trigonometric:diagInfoFrn");
          } else {
            memory->create(hetProcStrcs[n][0].diagInfoFrn, 6 * hetProcStrcs[n][0].nFrnDiag, "trigonometric:diagInfoFrn");
          }
          hetProcStrcs[n][0].maxFrn = hetProcStrcs[n][0].nFrnDiag;
        }
        for (j = 0; j < 6; j++) {
          hetProcStrcs[n][0].diagInfoFrn[6 * (hetProcStrcs[n][0].nFrnDiag -1) + j] = strc.diagInfoRecv[6 * i + j];
        }

        if(dir) {
          hetProcStrcs[myN][myD].diagDir[hetProcStrcs[myN][myD].nDiagRecv][0] = n;
          hetProcStrcs[myN][myD].diagDir[hetProcStrcs[myN][myD].nDiagRecv][1] = 0; //left
          hetProcStrcs[myN][myD].diagDir[hetProcStrcs[myN][myD].nDiagRecv++][2] = -1; //FRN DIAG
        }

        goto nextDiag;
      } else if (strc.diagInfoRecv[6 * i + n] == 1) {
        if(++hetProcStrcs[n][1].nFrnDiag > hetProcStrcs[n][1].maxFrn) {
          if(hetProcStrcs[n][1].maxFrn > 0){
            memory->grow(hetProcStrcs[n][1].diagInfoFrn, 6 * hetProcStrcs[n][1].nFrnDiag, "trigonometric:diagInfoFrn");
          } else {
            memory->create(hetProcStrcs[n][1].diagInfoFrn, 6 * hetProcStrcs[n][1].nFrnDiag, "trigonometric:diagInfoFrn");
          }

          hetProcStrcs[n][1].maxFrn = hetProcStrcs[n][1].nFrnDiag;
        }
        
        for (j = 0; j < 6; j++) {
          hetProcStrcs[n][1].diagInfoFrn[6 * (hetProcStrcs[n][1].nFrnDiag -1) + j] = strc.diagInfoRecv[6 * i + j];
        }

        if(dir) {
          hetProcStrcs[myN][myD].diagDir[hetProcStrcs[myN][myD].nDiagRecv][0] = n;
          hetProcStrcs[myN][myD].diagDir[hetProcStrcs[myN][myD].nDiagRecv][1] = 1; //left
          hetProcStrcs[myN][myD].diagDir[hetProcStrcs[myN][myD].nDiagRecv++][2] = -1; //FRN DIAG                  
        }
        goto nextDiag;
      }
    }
    //DIAG FOUND ITS PROC
    //Reverse Comm
    for (n = 0; n < 3; n++) {
      if(strc.diagInfoRecv[6 * i + n] == -1) {
        if (dir) {
          hetProcStrcs[myN][myD].diagDir[hetProcStrcs[myN][myD].nDiagRecv][0] = n;
          hetProcStrcs[myN][myD].diagDir[hetProcStrcs[myN][myD].nDiagRecv][1] = 1;
          hetProcStrcs[myN][myD].diagDir[hetProcStrcs[myN][myD].nDiagRecv++][2] = findIndex(n, 1, strc.diagInfoRecv[6 * i + 4], strc.diagInfoRecv[6 * i + 3]);
        } else { 
          if(++hetProcStrcs[n][1].nNtvDiag > hetProcStrcs[n][1].maxNtv) {
            if(hetProcStrcs[n][1].maxNtv > 0){
              memory->grow(hetProcStrcs[n][1].diagInfoNtv, 6 * hetProcStrcs[n][1].nNtvDiag, "trigonometric:diagInfoNtv");
            } else {
              memory->create(hetProcStrcs[n][1].diagInfoNtv, 6 * hetProcStrcs[n][1].nNtvDiag, "trigonometric:diagInfoNtv");
            }
            hetProcStrcs[n][1].maxNtv = hetProcStrcs[n][1].nNtvDiag;
          }
          
          for (j = 0; j < 3; j++) {
            hetProcStrcs[n][1].diagInfoNtv[6 * (hetProcStrcs[n][1].nNtvDiag-1) + j] = -strc.diagInfoRecv[6 * i + j];
          }
          hetProcStrcs[n][1].diagInfoNtv[6 *  (hetProcStrcs[n][1].nNtvDiag-1) + 3] = strc.diagInfoRecv[6 * i + 4]; //switch send and recv
          hetProcStrcs[n][1].diagInfoNtv[6 *  (hetProcStrcs[n][1].nNtvDiag-1) + 4] = strc.diagInfoRecv[6 * i + 3];
          hetProcStrcs[n][1].diagInfoNtv[6 *  (hetProcStrcs[n][1].nNtvDiag-1) + 5] = strc.diagInfoRecv[6 * i + 5];
        }
        goto nextDiag;
      } else if (strc.diagInfoRecv[6 * i + n] == 1) {
        if (dir) {
          hetProcStrcs[myN][myD].diagDir[hetProcStrcs[myN][myD].nDiagRecv][0] = n;
          hetProcStrcs[myN][myD].diagDir[hetProcStrcs[myN][myD].nDiagRecv][1] = 0;
          hetProcStrcs[myN][myD].diagDir[hetProcStrcs[myN][myD].nDiagRecv++][2] = findIndex(n, 0, strc.diagInfoRecv[6 * i + 4], strc.diagInfoRecv[6 * i + 3]);
        } else { 
          if(++hetProcStrcs[n][0].nNtvDiag > hetProcStrcs[n][0].maxNtv) {
            if(hetProcStrcs[n][0].maxNtv > 0){
            memory->grow(hetProcStrcs[n][0].diagInfoNtv, 6 * hetProcStrcs[n][0].nNtvDiag, "trigonometric:diagInfoNtv");
            } else {
            memory->create(hetProcStrcs[n][0].diagInfoNtv, 6 * hetProcStrcs[n][0].nNtvDiag, "trigonometric:diagInfoNtv");
            }
            hetProcStrcs[n][0].maxNtv = hetProcStrcs[n][0].nNtvDiag;
          }

          for (j = 0; j < 3; j++) {
            hetProcStrcs[n][0].diagInfoNtv[6 * (hetProcStrcs[n][0].nNtvDiag-1) + j] = -strc.diagInfoRecv[6 * i + j];
          }
          hetProcStrcs[n][0].diagInfoNtv[6 *  (hetProcStrcs[n][0].nNtvDiag-1) + 3] = strc.diagInfoRecv[6 * i + 4]; //switch send and recv
          hetProcStrcs[n][0].diagInfoNtv[6 *  (hetProcStrcs[n][0].nNtvDiag-1) + 4] = strc.diagInfoRecv[6 * i + 3];
          hetProcStrcs[n][0].diagInfoNtv[6 *  (hetProcStrcs[n][0].nNtvDiag-1) + 5] = strc.diagInfoRecv[6 * i + 5];     
        }
        goto nextDiag;
      }
    }

    nextDiag:;
  }
}

void FixNHTrigo::findGhostIndex(HetProcStrc *strc) {
  double **x = atom->x;
  int *sametag = atom->sametag;
  Bond *bond = force->bond;
  int i1, i2;
  int d;
  int type;
  int bdim, dim = 3;
  double *del = new double[3];
  double r0B, rsq;
  double *r0 = (double*)bond->extract("r0", bdim);

  for (int n = 0; n < strc->nShrdHet; n++) {
    i1 = atom->map(strc->hetSendInd[n]);
    i2 = atom->map(strc->hetRecvInd[n]);

    type = strc->bondType[n];
    r0B = r0[type];
    rsq = 0;

    for (d = 0; d < dim; d++) {
      del[d] = x[i2][d] - x[i1][d];
      rsq += del[d] * del[d];
    }

    while (sqrt(rsq) > 3 * r0B) {
      i2 = sametag[i2];
      rsq = 0;
      for (d = 0; d < dim; d++) {
        del[d] = x[i2][d] - x[i1][d];
        rsq += del[d] * del[d];
      }
    }

    strc->i1[n] = i1;
    strc->i2[n] = i2;
  }
  delete []del;
}

void FixNHTrigo::postProcess() {
  int n,d,i;
  int tmpSend, tmpRecv;

  for (n = 0; n < 3; n++) {
    for (d = 0; d < 2; d++) {   
      hetProcStrcs[n][d].nMyBonds[1] = hetProcStrcs[n][d].nNtvDiag + hetProcStrcs[n][d].nFrnDiag;
      hetProcStrcs[n][d].nShrdHet += hetProcStrcs[n][d].nNtvDiag;
      tmpRecv = hetProcStrcs[n][d].nShrdHet;
      tmpSend = tmpRecv + hetProcStrcs[n][d].nFrnDiag;
      if (tmpSend > hetProcStrcs[n][d].maxSend) {
        if(hetProcStrcs[n][d].maxSend > 0) {
          memory->grow(hetProcStrcs[n][d].hetSend, 3 * tmpSend, "trigonometric:hetSend");
        } else {
          memory->create(hetProcStrcs[n][d].hetSend, 3 * tmpSend, "trigonometric:hetSend");
        }
        hetProcStrcs[n][d].maxSend = tmpSend;
      }

      if (tmpRecv > hetProcStrcs[n][d].maxRecv) {
        if(hetProcStrcs[n][d].maxRecv > 0) {
          memory->grow(hetProcStrcs[n][d].shrdHessian, tmpRecv, 9, "trigonometric:shrdHessian"); 

          memory->grow(hetProcStrcs[n][d].hetRecv, 3 * tmpRecv, "trigonometric:hetRecv");

          memory->grow(hetProcStrcs[n][d].hetSendInd, tmpRecv, "trigonometric:hetSendInd");
          memory->grow(hetProcStrcs[n][d].hetRecvInd, tmpRecv, "trigonometric:hetRecvInd");
          memory->grow(hetProcStrcs[n][d].i1, tmpRecv, "trigonometric:i1");
          memory->grow(hetProcStrcs[n][d].i2, tmpRecv, "trigonometric:i2");
          memory->grow(hetProcStrcs[n][d].bondType, tmpRecv, "trigonometric:bondType");
        } else {
          memory->create(hetProcStrcs[n][d].shrdHessian, tmpRecv, 9, "trigonometric:shrdHessian"); 

          memory->create(hetProcStrcs[n][d].hetRecv, 3 * tmpRecv, "trigonometric:hetRecv");

          memory->create(hetProcStrcs[n][d].hetSendInd, tmpRecv, "trigonometric:hetSendInd");
          memory->create(hetProcStrcs[n][d].hetRecvInd, tmpRecv, "trigonometric:hetRecvInd");
          memory->create(hetProcStrcs[n][d].i1, tmpRecv, "trigonometric:i1");
          memory->create(hetProcStrcs[n][d].i2, tmpRecv, "trigonometric:i2");
          
          memory->create(hetProcStrcs[n][d].bondType, tmpRecv, "trigonometric:bondType");
        }
        hetProcStrcs[n][d].maxRecv = tmpRecv;
      }
      hetProcStrcs[n][d].nPckdRecv = hetProcStrcs[n][d].nMyBonds[0] + hetProcStrcs[n][d].nYourBonds[0] + hetProcStrcs[n][d].nDiagRecv;
      if (hetProcStrcs[n][d].nPckdRecv > hetProcStrcs[n][d].maxPckdRecv) {
        if (hetProcStrcs[n][d].maxPckdRecv > 0) {
          memory->destroy(hetProcStrcs[n][d].packedRecv);
        }
        memory->create(hetProcStrcs[n][d].packedRecv, 3 * hetProcStrcs[n][d].nPckdRecv, "trigonometric:packedRecv");

        hetProcStrcs[n][d].maxPckdRecv = hetProcStrcs[n][d].nPckdRecv;
      }
    }
  }

  for (n = 0; n < 3; n++) {
    for (d = 0; d < 2; d++) {   
      for (i = 0; i < hetProcStrcs[n][d].nNtvDiag; i++) {
        hetProcStrcs[n][d].hetSendInd[hetProcStrcs[n][d].nShrdHet -  hetProcStrcs[n][d].nNtvDiag + i] = hetProcStrcs[n][d].diagInfoNtv[6 * i + 3];
        hetProcStrcs[n][d].hetRecvInd[hetProcStrcs[n][d].nShrdHet -  hetProcStrcs[n][d].nNtvDiag + i] = hetProcStrcs[n][d].diagInfoNtv[6 * i + 4];
        hetProcStrcs[n][d].bondType[hetProcStrcs[n][d].nShrdHet -  hetProcStrcs[n][d].nNtvDiag + i] = hetProcStrcs[n][d].diagInfoNtv[6 * i + 5];
      }
      findGhostIndex(&hetProcStrcs[n][d]);
    }
  }
}

void FixNHTrigo::update_hetComm() {
  int n, d, i;
  int tmpPos;
  int tmpMB;
  int tmpSum;
  int tmpSend, tmpRecv;
  
  int *packedSend, *packedRecv;

  MPI_Request *request = new MPI_Request[6];
  MPI_Status *status = new MPI_Status[6];
  int *storeProc = new int[4 * nHet];
  int **c;

  memory->create(c, 3, 2, "trigonometric:c");

  for (n = 0; n < 3; n++) {
    for (d = 0; d < 2; d++) {     
      c[n][d] = 0; 
      hetProcStrcs[n][d].nNtvDiag = 0;
      hetProcStrcs[n][d].nFrnDiag = 0;
      for (i = 0; i < 2; i++)
        hetProcStrcs[n][d].nMyBonds[i] = 0;
    }
  }
  
  assignBonds(storeProc);

  for (n = 0; n < 3; n++) {
    for (d = 0; d < 2; d++) {
      hetProcStrcs[n][d].nNtvDiag = hetProcStrcs[n][d].nMyBonds[1];

      if (hetProcStrcs[n][d].nNtvDiag > hetProcStrcs[n][d].maxNtv) {
        if(hetProcStrcs[n][d].maxNtv > 0) {
          memory->destroy(hetProcStrcs[n][d].diagInfoNtv);
        }
        memory->create(hetProcStrcs[n][d].diagInfoNtv, 6 * hetProcStrcs[n][d].nNtvDiag, "trigonometric:diagInfoNtv");

        hetProcStrcs[n][d].maxNtv = hetProcStrcs[n][d].nNtvDiag;
      }
    }
  }
  //fill Diag arrays with diags STARTING in me
  for (i = 0; i < nHet; i++) {
    for (n = 0; n < 3; n++) {
      for (d = 0; d < 2; d++) {
        if (storeProc[4*i] == (n + 3 * d)) {
          tmpSum = abs(storeProc[4*i + 1 + (n+1)%3]) + abs(storeProc[4*i + 1 + (n+2)%3]);
          if (tmpSum != 0) {
            hetProcStrcs[n][d].diagInfoNtv[6 * c[n][d] + 0] = storeProc[4*i+1];
            hetProcStrcs[n][d].diagInfoNtv[6 * c[n][d] + 1] = storeProc[4*i+2];
            hetProcStrcs[n][d].diagInfoNtv[6 * c[n][d] + 2] = storeProc[4*i+3];
            hetProcStrcs[n][d].diagInfoNtv[6 * c[n][d] + 3] = hetBondList[i][0];
            hetProcStrcs[n][d].diagInfoNtv[6 * c[n][d] + 4] = hetBondList[i][1];
            hetProcStrcs[n][d].diagInfoNtv[6 * c[n][d] + 5] = hetBondList[i][2];
            c[n][d]++;
          }
          goto nextHet;
        }
      }
    }
    nextHet:;
  }
  //THIS COULD BE MADE WITH ONE COM STEP - MORE COMPLICATED - COULD IT REALLY?
  //Comm existence of diag bonds - only necessary if newton on
  for (n = 0; n < 3; n++) {
    for (d = 0; d < 2; d++) {
      hetProcStrcs[n][1-d].nMyBonds[1] = hetProcStrcs[n][1-d].nNtvDiag + hetProcStrcs[n][1-d].nFrnDiag;

      //Dont need nYourBonds[0] - But does it really hurt? 
      MPI_Irecv(hetProcStrcs[n][d].nYourBonds, 2,MPI_INT,comm->procneigh[n][d],0,world,&request[n+3*d]);
      MPI_Send(hetProcStrcs[n][1-d].nMyBonds,2,MPI_INT,comm->procneigh[n][1-d],0,world);
      MPI_Wait(&request[n+3*d],&status[n+3*d]);
      //allocate recv memory
      if (hetProcStrcs[n][d].nYourBonds[1] > hetProcStrcs[n][d].maxDiagRecv) {
        if(hetProcStrcs[n][d].maxDiagRecv > 0) {
          memory->destroy(hetProcStrcs[n][d].diagInfoRecv);
          memory->destroy(hetProcStrcs[n][d].diagDir);
        }
        memory->create(hetProcStrcs[n][d].diagInfoRecv, 6 * hetProcStrcs[n][d].nYourBonds[1], "trigonometric:diagInfoRecv");
        memory->create(hetProcStrcs[n][d].diagDir, hetProcStrcs[n][d].nYourBonds[1], 3, "trigonometric:diagDir");
        hetProcStrcs[n][d].maxDiagRecv = hetProcStrcs[n][d].nYourBonds[1];
      }

      //pack data
      if (hetProcStrcs[n][1-d].nMyBonds[1] > hetProcStrcs[n][1-d].maxDiagSend) {
        if(hetProcStrcs[n][1-d].maxDiagSend > 0) {
          memory->destroy(hetProcStrcs[n][1-d].diagInfoSend);
        }
        memory->create(hetProcStrcs[n][1-d].diagInfoSend, 6 * hetProcStrcs[n][1-d].nMyBonds[1], "trigonometric:diagInfoSend");

        hetProcStrcs[n][1-d].maxDiagSend = hetProcStrcs[n][1-d].nMyBonds[1];
      }

      for (i = 0; i < 6 * hetProcStrcs[n][1-d].nNtvDiag; i++) 
        hetProcStrcs[n][1-d].diagInfoSend[i] = hetProcStrcs[n][1-d].diagInfoNtv[i];

      for (i = 0; i < 6 * hetProcStrcs[n][1-d].nFrnDiag; i++) 
        hetProcStrcs[n][1-d].diagInfoSend[6 * hetProcStrcs[n][1-d].nNtvDiag + i] = hetProcStrcs[n][1-d].diagInfoFrn[i];

      MPI_Irecv(hetProcStrcs[n][d].diagInfoRecv, 6 * hetProcStrcs[n][d].nYourBonds[1],MPI_INT,comm->procneigh[n][d],0,world,&request[n+3*d]);
      MPI_Send(hetProcStrcs[n][1-d].diagInfoSend, 6 * hetProcStrcs[n][1-d].nMyBonds[1],MPI_INT,comm->procneigh[n][1-d],0,world);
      MPI_Wait(&request[n+3*d],&status[n+3*d]);

      sortDiagInfo(n, d, false);
    }
  }

  for (n = 0; n < 3; n++) {
    for (d = 0; d < 2; d++) {
      //reset counter
      c[n][d] = 0;
      //Allocate Send shared Bonds plus MY diag bonds
      hetProcStrcs[n][d].nShrdHet = hetProcStrcs[n][d].nMyBonds[0] + hetProcStrcs[n][d].nYourBonds[0];
      tmpSend = hetProcStrcs[n][d].nShrdHet;
      tmpRecv = tmpSend;
      if (tmpSend > hetProcStrcs[n][d].maxSend) {
        if(hetProcStrcs[n][d].maxSend > 0) {
          memory->destroy(hetProcStrcs[n][d].hetSend);
        }
        memory->create(hetProcStrcs[n][d].hetSend, 3 * tmpSend, "trigonometric:hetSend");

        hetProcStrcs[n][d].maxSend = tmpSend;
      }

      if (tmpRecv > hetProcStrcs[n][d].maxRecv) {
        if(hetProcStrcs[n][d].maxRecv > 0) {
          memory->destroy(hetProcStrcs[n][d].hetSendInd);
          memory->destroy(hetProcStrcs[n][d].hetRecvInd);
          memory->destroy(hetProcStrcs[n][d].i1);
          memory->destroy(hetProcStrcs[n][d].i2);
          memory->destroy(hetProcStrcs[n][d].hetRecv);
          memory->destroy(hetProcStrcs[n][d].bondType);
          memory->destroy(hetProcStrcs[n][d].shrdHessian); 
        }
        memory->create(hetProcStrcs[n][d].hetSendInd, tmpRecv, "trigonometric:hetSendInd");
        memory->create(hetProcStrcs[n][d].hetRecvInd, tmpRecv, "trigonometric:hetRecvInd");
        memory->create(hetProcStrcs[n][d].i1, tmpRecv, "trigonometric:i1");
        memory->create(hetProcStrcs[n][d].i2, tmpRecv, "trigonometric:i2");
        memory->create(hetProcStrcs[n][d].hetRecv, 3 * tmpRecv, "trigonometric:hetRecv");
        memory->create(hetProcStrcs[n][d].bondType, tmpRecv, "trigonometric:bondType");
        memory->create(hetProcStrcs[n][d].shrdHessian, tmpRecv, 9, "trigonometric:shrdHessian"); 

        hetProcStrcs[n][d].maxRecv = tmpRecv;
      }
    }
  }

  for (i = 0; i < nHet; i++) {
    for (n = 0; n < 3; n++) {
      for (d = 0; d < 2; d++) {
        if (storeProc[4*i] == (n + 3 * d)) {
         tmpSum = abs(storeProc[4*i + 1 + (n+1)%3]) + abs(storeProc[4*i + 1 + (n+2)%3]);
          if (tmpSum == 0) { //Direct Bond
            hetProcStrcs[n][d].hetSendInd[c[n][d]] = hetBondList[i][0];
            hetProcStrcs[n][d].hetRecvInd[c[n][d]] = hetBondList[i][1];
            hetProcStrcs[n][d].bondType[c[n][d]++] = hetBondList[i][2];
          }
        }
      }
    }
  }

//Reset Frn
  for (n = 0; n < 3; n++)
    for (d = 0; d < 2; d++)
      hetProcStrcs[n][d].nFrnDiag = 0;


  //FINAL SORT
  for (n = 0; n < 3; n++) {
    for (d = 0; d < 2; d++) {

      //Send again - Frgn diag mightve increased nMy|Yourbonds[0] should never change
      hetProcStrcs[n][1-d].nMyBonds[1] = hetProcStrcs[n][1-d].nNtvDiag + hetProcStrcs[n][1-d].nFrnDiag;
      MPI_Irecv(hetProcStrcs[n][d].nYourBonds, 2,MPI_INT,comm->procneigh[n][d],0,world,&request[n+3*d]);
      MPI_Send(hetProcStrcs[n][1-d].nMyBonds,2,MPI_INT,comm->procneigh[n][1-d],0,world);
      MPI_Wait(&request[n+3*(d)],&status[n+3*d]);

      //allocate memory
      if (hetProcStrcs[n][d].nYourBonds[1] > hetProcStrcs[n][d].maxDiagRecv) {
        if(hetProcStrcs[n][d].maxDiagRecv > 0) {
          memory->destroy(hetProcStrcs[n][d].diagInfoRecv);
          memory->grow(hetProcStrcs[n][d].diagDir, hetProcStrcs[n][d].nYourBonds[1], 3, "trigonometric:diagDir");
        } else {
          memory->create(hetProcStrcs[n][d].diagDir, hetProcStrcs[n][d].nYourBonds[1], 3, "trigonometric:diagDir");
        }
        memory->create(hetProcStrcs[n][d].diagInfoRecv, 6 * hetProcStrcs[n][d].nYourBonds[1], "trigonometric:diagInfoRecv");

        hetProcStrcs[n][d].maxDiagRecv = hetProcStrcs[n][d].nYourBonds[1];
      }

      tmpSend = 3 * hetProcStrcs[n][1-d].nMyBonds[0] + 6 * hetProcStrcs[n][1-d].nMyBonds[1];
      tmpRecv = 3 * hetProcStrcs[n][d].nYourBonds[0] + 6 * hetProcStrcs[n][d].nYourBonds[1];
      packedSend = new int[tmpSend];
      packedRecv = new int[tmpRecv];

      // pack data  
      tmpPos = 0;
      //Het send ind
      for(i = 0; i < hetProcStrcs[n][1-d].nMyBonds[0]; i++) 
        packedSend[tmpPos + i] = hetProcStrcs[n][1-d].hetSendInd[i];

      tmpPos += hetProcStrcs[n][1-d].nMyBonds[0];

      //Het recv ind
      for(i = 0; i < hetProcStrcs[n][1-d].nMyBonds[0]; i++) 
        packedSend[tmpPos + i] = hetProcStrcs[n][1-d].hetRecvInd[i];
      tmpPos += hetProcStrcs[n][1-d].nMyBonds[0];

      //Het bond type
      for(i = 0; i < hetProcStrcs[n][1-d].nMyBonds[0]; i++) 
        packedSend[tmpPos + i] = hetProcStrcs[n][1-d].bondType[i];    
      tmpPos += hetProcStrcs[n][1-d].nMyBonds[0];

      //Diag info
      for(i = 0; i < 6 * hetProcStrcs[n][1-d].nNtvDiag; i++) 
        packedSend[tmpPos + i] = hetProcStrcs[n][1-d].diagInfoNtv[i];

      tmpPos += 6 * hetProcStrcs[n][1-d].nNtvDiag;

      for (i = 0; i < 6 * hetProcStrcs[n][1-d].nFrnDiag; i++)
        packedSend[tmpPos + i] = hetProcStrcs[n][1-d].diagInfoFrn[i];

      MPI_Isend(packedSend, tmpSend, MPI_INT, comm->procneigh[n][1-d], 0, world, &request[3*d]);
      MPI_Recv(packedRecv, tmpRecv, MPI_INT, comm->procneigh[n][d], 0, world, &status[3*d]);
      //unpack data
      tmpMB = hetProcStrcs[n][d].nMyBonds[0];
      tmpPos = 0;
      //Het recv ind
      for(i = 0; i < hetProcStrcs[n][d].nYourBonds[0]; i++) 
        hetProcStrcs[n][d].hetRecvInd[tmpMB + i] = packedRecv[tmpPos + i];

      tmpPos += hetProcStrcs[n][d].nYourBonds[0];
      //Het send ind
      for(i = 0; i < hetProcStrcs[n][d].nYourBonds[0]; i++) 
        hetProcStrcs[n][d].hetSendInd[tmpMB + i] = packedRecv[tmpPos + i];
      tmpPos += hetProcStrcs[n][d].nYourBonds[0];
      //Het bond type
      for(i = 0; i < hetProcStrcs[n][d].nYourBonds[0]; i++) 
        hetProcStrcs[n][d].bondType[tmpMB + i] = packedRecv[tmpPos + i];    
      tmpPos += hetProcStrcs[n][d].nYourBonds[0];
      //Diag info
      for(i = 0; i < 6 * hetProcStrcs[n][d].nYourBonds[1]; i++) 
        hetProcStrcs[n][d].diagInfoRecv[i] = packedRecv[tmpPos + i];
      
      sortDiagInfo(n, d, true);

      delete []packedSend;
      delete []packedRecv;
    }
  }
  postProcess();
  memory->destroy(c);
  delete []storeProc;
  delete []request;
  delete []status;
}

void FixNHTrigo::commVec(double *vec) {
  int locInd;
  int n,d,i,j,dm;
  int het;
  int tmpPos;
  int tmpN, tmpD;
  int tmpSend;
  int tmpRecv;
  int tmpPrev;

  MPI_Status *status = new MPI_Status[6];
  MPI_Request *request = new MPI_Request[6];
  //double *packedRecv;
  int **tmpC;
  memory->create(tmpC, 3, 2, "trigonometric:tmpC");
  for (n = 0; n < 3; n++) 
    for (d = 0; d < 2; d++) 
      tmpC[n][d] = 0;

  //Pack Comm
  for (n = 0; n < 3; n++) {
    for (d = 0; d < 2; d++) {
      het = hetProcStrcs[n][d].nMyBonds[0] + hetProcStrcs[n][d].nYourBonds[0];
      for (i = 0; i < hetProcStrcs[n][1-d].nShrdHet; i++) {
        locInd = hetProcStrcs[n][1-d].i1[i];
        for (dm = 0; dm < 3; dm++) {
          hetProcStrcs[n][1-d].hetSend[index(i,dm)] = vec[index(locInd,dm)];
        }
      }
      tmpSend = hetProcStrcs[n][1-d].nMyBonds[0] + hetProcStrcs[n][1-d].nYourBonds[0] + hetProcStrcs[n][1-d].nNtvDiag + hetProcStrcs[n][1-d].nFrnDiag;
      tmpRecv = het + hetProcStrcs[n][d].nDiagRecv;// hetProcStrcs[n][d].nFrnRecv + hetProcStrcs[n][d].nNtvRecv;

      MPI_Irecv(hetProcStrcs[n][d].packedRecv, 3 * tmpRecv,MPI_DOUBLE,comm->procneigh[n][d],0,world,&request[n+3*d]);
      MPI_Send(hetProcStrcs[n][1-d].hetSend,3 * tmpSend,MPI_DOUBLE,comm->procneigh[n][1-d],0,world);
      MPI_Wait(&request[n+3*d],&status[n+3*d]);
      //unpack
      
      tmpPos = 0;
      for (i = 0; i < 3 * hetProcStrcs[n][d].nYourBonds[0]; i++) 
        hetProcStrcs[n][d].hetRecv[3*hetProcStrcs[n][d].nMyBonds[0] + i] = hetProcStrcs[n][d].packedRecv[i];
      tmpPos += 3*hetProcStrcs[n][d].nYourBonds[0];

      for (i = 0; i < 3 * hetProcStrcs[n][d].nMyBonds[0]; i++) 
        hetProcStrcs[n][d].hetRecv[i] = hetProcStrcs[n][d].packedRecv[tmpPos + i];
      tmpPos += 3*hetProcStrcs[n][d].nMyBonds[0];

      for(i = 0; i < hetProcStrcs[n][d].nDiagRecv; i++) {
        tmpN = hetProcStrcs[n][d].diagDir[i][0];
        tmpD = hetProcStrcs[n][d].diagDir[i][1];
        tmpPrev = hetProcStrcs[tmpN][tmpD].nMyBonds[0] + hetProcStrcs[tmpN][tmpD].nYourBonds[0];
        if (hetProcStrcs[n][d].diagDir[i][2] == -1) {
          for (j = 0; j < 3; j++) 
            hetProcStrcs[tmpN][tmpD].hetSend[3 * (tmpPrev + hetProcStrcs[tmpN][tmpD].nNtvDiag + tmpC[tmpN][tmpD])+ j] = hetProcStrcs[n][d].packedRecv[tmpPos + 3*i + j];
          tmpC[tmpN][tmpD]++;
        } else {
          for (j = 0; j < 3; j++)
            hetProcStrcs[tmpN][tmpD].hetRecv[3 * (tmpPrev + hetProcStrcs[n][d].diagDir[i][2]) + j] = hetProcStrcs[n][d].packedRecv[tmpPos + 3*i + j];
        }
      }
    }
  }

  memory->destroy(tmpC);

  delete []status;
  delete []request;
}

void FixNHTrigo::clearProcStrc() {
  for (int n = 0; n < 3; n++) {
    for (int d = 0; d < 2; d++) {
      if(hetProcStrcs[n][d].maxNtv > 0)
        memory->destroy(hetProcStrcs[n][d].diagInfoNtv);      
      if(hetProcStrcs[n][d].maxFrn > 0)
        memory->destroy(hetProcStrcs[n][d].diagInfoFrn);
      if(hetProcStrcs[n][d].maxDiagRecv > 0) {
        memory->destroy(hetProcStrcs[n][d].diagDir);
        memory->destroy(hetProcStrcs[n][d].diagInfoRecv);
      }
      if(hetProcStrcs[n][d].maxDiagSend > 0) 
        memory->destroy(hetProcStrcs[n][d].diagInfoSend);
      
      if(hetProcStrcs[n][d].maxSend > 0) 
        memory->destroy(hetProcStrcs[n][d].hetSend);
      if(hetProcStrcs[n][d].maxRecv > 0) {
        memory->destroy(hetProcStrcs[n][d].hetRecvInd);
        memory->destroy(hetProcStrcs[n][d].hetSendInd);
        memory->destroy(hetProcStrcs[n][d].i1);
        memory->destroy(hetProcStrcs[n][d].i2);
        memory->destroy(hetProcStrcs[n][d].hetRecv);
        memory->destroy(hetProcStrcs[n][d].bondType);
        memory->destroy(hetProcStrcs[n][d].shrdHessian);
      }
    }
  }
}

void FixNHTrigo::setMemory() {
  int nbondlist = neighbor->nbondlist;
  int **bondlist = neighbor->bondlist; 
  int nlocal = atom->nlocal; 
  int i1, i2;

  nHet = 0;
  nHess = 0;

   //WAY TO COMPLICATED AND INEFFICIENT...
  for (int n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    //Only conisder 'local' bonds [not pure ghosts]
    if (i1 < nlocal || i2 < nlocal) {
      //PURE LOCAL
      if (i1 < nlocal && i2 < nlocal) {
        nHess++;
      } else {
        if (i2 >= nlocal) {
          //LOCAL BUT OVER BOUNDARY
          if (atom->map(atom->tag[i2]) < nlocal) {
            nHess++;
          } else { // TRUE HETBOND
            nHet++;
          }
        } else {
          if (atom->map(atom->tag[i1]) < nlocal) {
            nHess++;
          } else { // TRUE HETBOND
            nHet++;
          }
        }
      }
    }
  }

  if(nHess > maxHess) {
    if (maxHess > 0) {
      memory->destroy(implHessian); 
      memory->destroy(homBondList);
    } 
    memory->create(implHessian, nHess, 9, "trigonometric:implHessian"); 
    memory->create(homBondList, nHess, 3,"trigonometric:homBondList");

    maxHess = nHess;
  }

  if(nHet > maxHet) {
    if (maxHet > 0) {
      memory->destroy(hetBondList);
    }
    memory->create(hetBondList, nHet, 4, "trigonometric:hetBondList");

    maxHet = nHet;
  }  
}

void FixNHTrigo::printImplHessian() {
  for (int i = 0; i < nHess; i++) {
    for (int j = 0; j < 9; j++) {
      printf("%d:(%d) %.15f  ", comm->me, i, implHessian[i][j]);
    }
    printf("\n");
  }
}

int FixNHTrigo::updateHessian() {
  //Grab Some Information
  double **x = atom->x;
  double *m = atom->mass;
  int **bondlist = neighbor->bondlist;
  Bond *bond = force->bond;
  int nbondlist = neighbor->nbondlist;
  int maxspecial = atom->maxspecial;
  int nlocal = atom->nlocal;

  int i, dm, n, i1, i2, type;
  double rsq,r,r0B, k0B; 

  int dim = 3, bdim;
  int counter;
  double *del, *kappa, *r0;
  int ret;
  double kappaMax = 0.0;
  double tmp1, tmp2;

  del = new double[3];

  //INDICIES 'N COUNTERS
  int j,k,b,d;

  kappa = (double*)bond->extract("kappa", bdim); //What does the dim mean for different bonds?
  r0 = (double*)bond->extract("r0", bdim);

  for (n = 0; n < nHess; n++) {
    counter = 0;
    i1 = homBondList[n][0];
    i2 = homBondList[n][1];
    type = homBondList[n][2];
    r0B = r0[type];
    k0B = 2 * kappa[type];
    rsq = 0;

    if(k0B > kappaMax) kappaMax = k0B;

    for (d = 0; d < dim; d++) {
    //PROBLEM WITH GHOST PARTICLE ON 'WRONG SIDE'
      del[d] = x[i2][d] - x[i1][d];

      if (del[d] > 3 * r0B) {
        del[d] -= (domain->boxhi[d] - domain->boxlo[d]);
      } else  if (del[d] < - 3 * r0B){
        del[d] += (domain->boxhi[d] - domain->boxlo[d]);
      }
      rsq += del[d] * del[d];
    }

    r = sqrt(rsq);

    if (r > 0.0) { 
      for (d = 0; d < dim; d++) {
        for (b = 0; b < dim; b++) {
          implHessian[n][counter++] = - (k0B) * ((del[d] * del[b]) * r0B) / (r * r * r);
        }
      }
    } else {
      for (d = 0; d < dim; d++) {
        for (b = 0; b < dim; b++) {
          implHessian[n][counter++] = 0.0;
        }
      }
    }
  }

  for (i = 0; i < 3; i++) {
    for (dm = 0; dm < 2; dm++) {
      for (n = 0; n < hetProcStrcs[i][dm].nShrdHet; n++) {
        counter = 0;

        i1 = hetProcStrcs[i][dm].i1[n];
        i2 = hetProcStrcs[i][dm].i2[n];
        type = hetProcStrcs[i][dm].bondType[n];

        r0B = r0[type];
        k0B = 2.0 * kappa[type];
        rsq = .0;
        if(k0B > kappaMax) kappaMax = k0B;
        for (d = 0; d < dim; d++) {          
          del[d] = x[i2][d] - x[i1][d];           
          rsq += del[d] * del[d];
        }

        r = sqrt(rsq);

        if (r > 0.0) { 
          for (d = 0; d < dim; d++) {
            for (b = 0; b < dim; b++) {
              hetProcStrcs[i][dm].shrdHessian[n][counter++] = - (k0B) * ((del[d] * del[b]) * r0B) / (r * r * r);
            }
          }
        } else {
          for (d = 0; d < dim; d++) {
            for (b = 0; b < dim; b++) {
              hetProcStrcs[i][dm].shrdHessian[n][counter++] = 0.0;
            }
          }
        }
      }
    }
  }

  ret = 5;//(int)sqrt(kappaMax * maxspecial);

  delete []del;

  return ret;
}
