/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Mohamed T. Elewa (KIT)
------------------------------------------------------------------------- */

#include <mpi.h>
#include <cstring>
#include <cstdlib>
#include "fix_avevelocity.h"
#include "update.h"
#include "domain.h"
#include "atom.h"
#include "region.h"
#include "variable.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "force.h"
#include "input.h"

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NONE,CONSTANT,EQUAL};

/* ---------------------------------------------------------------------- */

FixAveVelocity::FixAveVelocity(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  xstr(NULL), ystr(NULL), zstr(NULL)
{
  if (narg < 6) error->all(FLERR,"Illegal fix velocity command");

  dynamic_group_allow = 1;
  vector_flag = 1;
  size_vector = 3;
  global_freq = 1;
  extvector = 1;

  xstr = ystr = zstr = NULL;

  // If the 4th argument (fx) is a variable 'v_' then get the string/variable name 
  // length and store it in 'xstr', if it is stated as NULL then NULL otherwise
  // use the floating point value (ensured by the method numeric in the class force)
  // same applies for y and z.

  if (strstr(arg[3],"v_") == arg[3]) {
    int n = strlen(&arg[3][2]) + 1;		// why + 1 //
    xstr = new char[n];
    strcpy(xstr,&arg[3][2]);
  } else if (strcmp(arg[3],"NULL") == 0) {
    xstyle = NONE;
  } else {
    xvalue = force->numeric(FLERR,arg[3]);
    xstyle = CONSTANT;
  }
  if (strstr(arg[4],"v_") == arg[4]) {
    int n = strlen(&arg[4][2]) + 1;
    ystr = new char[n];
    strcpy(ystr,&arg[4][2]);
  } else if (strcmp(arg[4],"NULL") == 0) {
    ystyle = NONE;
  } else {
    yvalue = force->numeric(FLERR,arg[4]);
    ystyle = CONSTANT;
  }
  if (strstr(arg[5],"v_") == arg[5]) {
    int n = strlen(&arg[5][2]) + 1;
    zstr = new char[n];
    strcpy(zstr,&arg[5][2]);
  } else if (strcmp(arg[5],"NULL") == 0) {
    zstyle = NONE;
  } else {
    zvalue = force->numeric(FLERR,arg[5]);
    zstyle = CONSTANT;
  }  
    voriginal_all[0] = voriginal_all[1] =
    voriginal_all[2] = voriginal_all[3] = 0.0;
}


/* ---------------------------------------------------------------------- */

int FixAveVelocity::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  return mask;
}

/* ---------------------------------------------------------------------- */


void FixAveVelocity::init()
{
  // Check variables, variables are equal style. If the variable  

  if (xstr) {
    xvar = input->variable->find(xstr);
    if (xvar < 0)
      error->all(FLERR,"Variable name for fix avevelocity does not exist");
    if (input->variable->equalstyle(xvar)) xstyle = EQUAL;
    else error->all(FLERR,"Variable for fix aveforce is invalid style");
  }
  if (ystr) {
    yvar = input->variable->find(ystr);
    if (yvar < 0)
      error->all(FLERR,"Variable name for fix avevelocity does not exist");
    if (input->variable->equalstyle(yvar)) ystyle = EQUAL;
    else error->all(FLERR,"Variable for fix aveforce is invalid style");
  }
  if (zstr) {
    zvar = input->variable->find(zstr);
    if (zvar < 0)
      error->all(FLERR,"Variable name for fix avevelocity does not exist");
    if (input->variable->equalstyle(zvar)) zstyle = EQUAL;
    else error->all(FLERR,"Variable for fix aveforce is invalid style");
  }
}


/* ---------------------------------------------------------------------- */

// clean up

FixAveVelocity::~FixAveVelocity()
{
  delete [] xstr;
  delete [] ystr;
  delete [] zstr;
}

/* ---------------------------------------------------------------------- */

void FixAveVelocity::setup(int vflag)
{
  initial_integrate(vflag);
}

/* ---------------------------------------------------------------------- */

void FixAveVelocity::initial_integrate(int /*vflag*/)
{
  // sum velocities on participating atoms

  double **x = atom->x;
  double **v = atom->v;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  double voriginal[4];
  voriginal[0] = voriginal[1] = voriginal[2] = voriginal[3] = 0.0;

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      voriginal[0] += v[i][0];
      voriginal[1] += v[i][1];
      voriginal[2] += v[i][2];
      voriginal[3] += 1.0;
    }

  // average the velocity on participating atoms
  // add in requested amount, computed via variable evaluation if necessary
  // wrap variable evaluation with clear/add

  MPI_Allreduce(voriginal,voriginal_all,4,MPI_DOUBLE,MPI_SUM,world);

  int ncount = static_cast<int> (voriginal_all[3]);
  if (ncount == 0) return;

  if (varflag == EQUAL) {
    modify->clearstep_compute();
    if (xstyle == EQUAL) xvalue = input->variable->compute_equal(xvar);
    if (ystyle == EQUAL) yvalue = input->variable->compute_equal(yvar);
    if (zstyle == EQUAL) zvalue = input->variable->compute_equal(zvar);
    modify->addstep_compute(update->ntimestep + 1);
  }

  double vave[3];
  vave[0] = voriginal_all[0]/ncount + xvalue;
  vave[1] = voriginal_all[1]/ncount + yvalue;
  vave[2] = voriginal_all[2]/ncount + zvalue;

  // set velocity of all participating atoms to same value
  // only for active dimensions

  for (int i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) {
      if (xstyle) v[i][0] = vave[0];
      if (ystyle) v[i][1] = vave[1];
      if (zstyle) v[i][2] = vave[2];
    }
}

/* ----------------------------------------------------------------------
   return components of total velocity on fix group before force was changed
------------------------------------------------------------------------- */

double FixAveVelocity::compute_vector(int n)
{
  return voriginal_all[n];
}

