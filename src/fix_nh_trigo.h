/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_FIX_NH_TRIGO_H
#define LMP_FIX_NH_TRIGO_H

#include "fix_nh.h"

namespace LAMMPS_NS {

class FixNHTrigo : public FixNH {
  #define index(i, d) i * 3 + d
  #define sqr(x) ((x)*(x))
  #define cube(x) ((x)*(x)*(x))//Square & cube of a double value WHERE DOES THIS BELONG???

 public:
  FixNHTrigo(class LAMMPS *, int, char **);
  virtual ~FixNHTrigo() {};
  void init();
  void setup(int /*vflag*/);
  int setmask();
  virtual void post_neighbor();



 protected:
  void nve_v();
  void nve_x();


    /*KRYLOV STRUCTURE*/
    typedef struct {
      int kDim;           //Dimension of Krylov space (hopefully small)
      double norm;        //norm of b
      double** matV;      //ONB of Krylov space - dim dim*nlocal x kDim
      double* alpha;
      double* beta;
    }KrylovStrc;

    /*HET PROC STRUCTURE*/
    typedef struct {
      int nMyBonds[2];  //number of HetPart send by this proc 0:direct 1:diagonnal
      int nYourBonds[2];
      int nShrdHet;     //Ideally hetrecv hetsend diagrecv diagsend combined

      int nNtvDiag;
      int nFrnDiag;
      int nFrnRecv;
      int nNtvRecv;
      int nDiagRecv;
      int nPckdRecv;

      int maxDiagSend;
      int maxDiagRecv;
      int maxSend;
      int maxRecv;
      int maxNtv;
      int maxFrn;
      int maxPckdRecv;

      int* diagInfoNtv;   //diffX, diffY, diffZ, i1, i2, bondType
      int* diagInfoFrn;   //diffX, diffY, diffZ, i1, i2, bondType
      int** diagDirNtv;   //n(0|1|2) d(0|1) idx
      int** diagDirFrn;   //n(0|1|2) d(0|1)
      int** diagDir;      //n(0|1|2) d(0|1) idx

      int* diagInfoSend;      //diffX, diffY, diffZ, SendInd, RecvInd, bondType
      int* diagInfoRecv;      //diffX, diffY, diffZ, SendInd, RecvInd, bondType

      double* hetRecv;
      double* hetSend;
      double* packedRecv;
      tagint* hetRecvInd;
      tagint* hetSendInd;
      int* i1;
      int* i2;
      int* bondType;
      double** shrdHessian;
    }HetProcStrc;

    KrylovStrc forceKry; 
    HetProcStrc hetProcStrcs[3][2];

    //int *procLocals;
    //int *displacements;

    double** implHessian;
    int** homBondList;
    int** hetBondList;
    double *forceVec;
    double *forceOut;

    int nlocBond;
    int maxHess;
    int nHess;
    int maxHet;
    int nHet;
    int maxRho;
    int nRho;
    int maxLoc;

    /* Calculates the implicit Hessian vector product */
    void matVecImpl(double* in, double* out);

    /* Utilities */
    double vecVecProdPara(double* inA, double* inB, int n);
    void valVecProd(double a, double *v, double *out, int n);
    void vecVecSub(double* a, double* b, double* out, int n);
    double calcNormPara(double* vec, int n);
    void printMat(double** matIn, int n, int m);
    void printVec(double* vecIn, int n);
    
    /* Matrixfunction */
    void matFuncSmaller(double* alpha, double* beta, int aDim, int func, double delta_t, double* out);
    void lanczosImplPara(double *b, KrylovStrc *strc, int n);
    void matFuncPara(KrylovStrc *strc, double delta_t, int func, int n, double* out);
    
    void createStrc(KrylovStrc *str);
    void clearStrc(KrylovStrc *strc);

    /* Hetero Bonds */
    int findProc(double *x);
    void fillCommBonds(int n, int d, int val);
    void assignHetBonds(double *x, int i, int *storeProc);
    void assignBonds(int *storeProc);
    void informDiagProcs();
    void sortHetBonds(int *info, int n, int d);
    void allocateBondMem(HetProcStrc *strc);
    void allocateDiagMem(HetProcStrc *strc);
    void allocateDiagInfoRecv(HetProcStrc * strc, int newSize);
    void allocateDiagInfoSend(HetProcStrc * strc, int newSize);
    void sortDiagBonds(HetProcStrc *strc, int n);
    int findIndex(int n, int d, int i1, int i2);
    void sortDiagInfo(int n, int d, bool dir);
    void sortDiagFinal(int n, int d);
    void findGhostIndex(HetProcStrc *strc);
    void postProcess();
    void update_hetComm();
    void commVec(double *vec);
    void updateHetParts(double *vec);
    void clearProcStrc();

    /* Implicit Hessian */
    void setMemory();
    void printImplHessian();
    /* Update Hessian */
    int updateHessian();
};

}

#endif
