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
   Contributing authors: Steven E. Strong and Joel D. Eaves
   Joel.Eaves@Colorado.edu
------------------------------------------------------------------------- */

#include <cstdlib>
#include "fix_flow_pump.h"
#include "comm.h"
#include <mpi.h>
#include <cstring>
#include "atom.h"
#include "force.h"
#include "group.h"
#include "update.h"
#include "domain.h"
#include "error.h"
#include "modify.h"
#include "input.h"
#include "variable.h"
#include "citeme.h"
#include "respa.h"

#include <iostream>
using namespace std;

using namespace LAMMPS_NS;
using namespace FixConst;

enum{NONE,CONSTANT,EQUAL};

static const char cite_flow_gauss[] =
  "Gaussian dynamics package applied to a dynamic group:\n\n"
  "@Article{strong_water_2017,\n"
  "title = {The Dynamics of Water in Porous Two-Dimensional Crystals},\n"
  "volume = {121},\n"
  "number = {1},\n"
  "url = {https://doi.org/10.1021/acs.jpcb.6b09387},\n"
  "doi = {10.1021/acs.jpcb.6b09387},\n"
  "urldate = {2016-12-07},\n"
  "journal = {J. Phys. Chem. B},\n"
  "author = {Strong, Steven E. and Eaves, Joel D.},\n"
  "year = {2017},\n"
  "pages = {189--207}\n"
  "}\n\n";

FixFlowPump::FixFlowPump(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  xstr(NULL), ystr(NULL), zstr(NULL)
{
  if (lmp->citeme) lmp->citeme->add(cite_flow_gauss);

  if (narg < 9) error->all(FLERR,"Not enough input arguments");

  // a group which conserves momentum must also conserve particle number
  dynamic_group_allow = 1;

  scalar_flag = 1;
  vector_flag = 1;
  extscalar = 1;
  extvector = 1;
  size_vector = 3;
  energy_global_flag = 1;
  global_freq = 1;    //data available every timestep
  respa_level_support = 1;
  //default respa level=outermost level is set in init()

  xstr = ystr = zstr = NULL;
  dimension = domain->dimension;

  //get inputs
  int tmpFlag;
  for (int ii=0; ii<3; ii++)
  {
    tmpFlag=utils::inumeric(FLERR,arg[3+ii],false,lmp);
    if (tmpFlag==1 || tmpFlag==0)
      flow[ii]=tmpFlag;
    else {
      error->all(FLERR,"Constraint flags must be 1 or 0");
    }
  }

  //current in pump region
  // for (int ii=0; ii<3; ii++) {
  //   current[ii]=force->numeric(FLERR, arg[6+ii]);
  // }

  if (strstr(arg[6],"v_") == arg[6]) {
    int n = strlen(&arg[6][2]) + 1;
    xstr = new char[n];
    strcpy(xstr,&arg[6][2]);
  } else if (strcmp(arg[6],"NULL") == 0) {
    xstyle = NONE;
  } else {
    xvalue = utils::numeric(FLERR,arg[6],false,lmp);
    xstyle = CONSTANT;
  }
  if (strstr(arg[7],"v_") == arg[7]) {
    int n = strlen(&arg[7][2]) + 1;
    ystr = new char[n];
    strcpy(ystr,&arg[7][2]);
  } else if (strcmp(arg[7],"NULL") == 0) {
    ystyle = NONE;
  } else {
    yvalue = utils::numeric(FLERR,arg[7],false,lmp);
    ystyle = CONSTANT;
  }
  if (strstr(arg[8],"v_") == arg[8]) {
    int n = strlen(&arg[8][2]) + 1;
    zstr = new char[n];
    strcpy(zstr,&arg[8][2]);
  } else if (strcmp(arg[8],"NULL") == 0) {
    zstyle = NONE;
  } else {
    zvalue = utils::numeric(FLERR,arg[8],false,lmp);
    zstyle = CONSTANT;
  }

  // by default, do not compute work done
  workflag=false;

  // process optional keyword
  int iarg = 9;
  while (iarg < narg) {
    if ( strcmp(arg[iarg],"energy") == 0 ) {
      if ( iarg+2 > narg ) error->all(FLERR,"Illegal energy keyword");
      if ( strcmp(arg[iarg+1],"yes") == 0 ) workflag = true;
      else if ( strcmp(arg[iarg+1],"no") != 0 )
        error->all(FLERR,"Illegal energy keyword");
      iarg += 2;
    } else error->all(FLERR,"Illegal fix flow/pump command");
  }

  //error checking
  if (dimension == 2) {
    if (flow[2])
      error->all(FLERR,"Can't constrain z flow in 2d simulation");
  }

  dt=update->dt;
  pe_tot=0.0;
}


/* ---------------------------------------------------------------------- */

// clean up

FixFlowPump::~FixFlowPump()
{
  delete [] xstr;
  delete [] ystr;
  delete [] zstr;
}

/* ---------------------------------------------------------------------- */

int FixFlowPump::setmask()
{
  int mask = 0;
  mask |= POST_FORCE;
  mask |= POST_FORCE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixFlowPump::init()
{

  // check variables
  if (xstr) {
    xvar = input->variable->find(xstr);
    if (xvar < 0)
      error->all(FLERR,"Variable name for fix flow/pump does not exist");
    if (input->variable->equalstyle(xvar)) xstyle = EQUAL;
    else error->all(FLERR,"Variable for fix flow/pump is invalid style");
  }
  if (ystr) {
    yvar = input->variable->find(ystr);
    if (yvar < 0)
      error->all(FLERR,"Variable name for fix flow/pump does not exist");
    if (input->variable->equalstyle(yvar)) ystyle = EQUAL;
    else error->all(FLERR,"Variable for fix flow/pump is invalid style");
  }
  if (zstr) {
    zvar = input->variable->find(zstr);
    if (zvar < 0)
      error->all(FLERR,"Variable name for fix flow/pump does not exist");
    if (input->variable->equalstyle(zvar)) zstyle = EQUAL;
    else error->all(FLERR,"Variable for fix flow/pump is invalid style");
  }

  if (xstyle == EQUAL || ystyle == EQUAL || zstyle == EQUAL) varflag = EQUAL;
  else varflag = CONSTANT;

  //if respa level specified by fix_modify, then override default (outermost)
  //if specified level too high, set to max level
  if (utils::strmatch(update->integrate_style,"^respa")) {
    ilevel_respa = ((Respa *) update->integrate)->nlevels-1;
    if (respa_level >= 0)
      ilevel_respa = MIN(respa_level,ilevel_respa);
  }
}

/* ----------------------------------------------------------------------
   setup is called after the initial evaluation of forces before a run, so we
   must remove the total force here too
   ------------------------------------------------------------------------- */
void FixFlowPump::setup(int vflag)
{
  //need to compute work done if set fix_modify energy yes
  if (thermo_energy) workflag = true;

  if (utils::strmatch(update->integrate_style,"^respa")) {
    ((Respa *) update->integrate)->copy_flevel_f(ilevel_respa);
    post_force_respa(vflag,ilevel_respa,0);
    ((Respa *) update->integrate)->copy_f_flevel(ilevel_respa);
  }
  else
    post_force(vflag);
}

/* ----------------------------------------------------------------------
   this is where Gaussian dynamics constraint is applied
   ------------------------------------------------------------------------- */
void FixFlowPump::post_force(int /*vflag*/)
{
  double **f   = atom->f;
  double **v   = atom->v;

  int *mask    = atom->mask;
  int *type    = atom->type;
  double *mass = atom->mass;
  double *rmass = atom->rmass;

  int nlocal   = atom->nlocal;

  int ii,jj;

  //get total mass of group
  double mTot=group->mass(igroup);
  if (mTot <= 0.0)
      error->all(FLERR,"Invalid group mass in fix flow/pump");

  //get total number of atoms in group
  bigint nTot=group->count(igroup);

  //find the total force on all atoms
  //initialize to zero
  double f_thisProc[3];
  double current_thisProc[3], current_tot[3];
  for (ii=0; ii<3; ii++) {
    f_thisProc[ii] = 0.0;
    current_thisProc[ii] = 0.0;
  }

  //add all forces on each processor and compute contribution to current
  for(ii=0; ii<nlocal; ii++)
    if (mask[ii] & groupbit)	// if atoms are in group
      for (jj=0; jj<3; jj++)
        if (flow[jj]) {
          f_thisProc[jj] += f[ii][jj];
          current_thisProc[jj] += v[ii][jj];
          /* Uncomment this and comment above to get mass flux
          if (rmass) {
            current_thisProc[jj] += rmass[ii]*v[ii][jj];
          } else {
            current_thisProc[jj] += mass[type[ii]]*v[ii][jj];
          }
          */
        }

  //add the processor sums together
  MPI_Allreduce(f_thisProc, f_tot, 3, MPI_DOUBLE, MPI_SUM, world);
  MPI_Allreduce(current_thisProc, current_tot, 3, MPI_DOUBLE, MPI_SUM, world);


  if (varflag == EQUAL) {
    modify->clearstep_compute();
    if (xstyle == EQUAL) xvalue = input->variable->compute_equal(xvar);
    if (ystyle == EQUAL) yvalue = input->variable->compute_equal(yvar);
    if (zstyle == EQUAL) zvalue = input->variable->compute_equal(zvar);
    modify->addstep_compute(update->ntimestep + 1);
  }

  double current[3];
  current[0] = xvalue;
  current[1] = yvalue;
  current[2] = zvalue;

  //compute applied acceleration
  for (ii=0; ii<3; ii++)
    a_app[ii] = -f_tot[ii] / mTot;

  //compute velocity correction
  for (ii=0; ii<3; ii++) {
    if (flow[ii]) {
      current_tot[ii] = current[ii] - current_tot[ii] / nTot;
      /* Uncomment this and comment above to get mass flux
      current_tot[ii] = current[ii] - current_tot[ii] / mTot;
      */
    } else {
      current_tot[ii] = 0.0;
    }
  }

  //apply added acceleration to each atom
  double f_app[3];
  double peAdded=0.0;
  for (ii = 0; ii<nlocal; ii++)
    if (mask[ii] & groupbit) {
      v[ii][0] += current_tot[0]; //current_tot[jj] is 0 if flow[jj] is false (if the flag is zero)
      v[ii][1] += current_tot[1];
      v[ii][2] += current_tot[2];

      if (rmass) {
        f_app[0] = a_app[0]*rmass[ii];
        f_app[1] = a_app[1]*rmass[ii];
        f_app[2] = a_app[2]*rmass[ii];
      } else {
        f_app[0] = a_app[0]*mass[type[ii]];
        f_app[1] = a_app[1]*mass[type[ii]];
        f_app[2] = a_app[2]*mass[type[ii]];
      }

      f[ii][0] += f_app[0]; //f_app[jj] is 0 if flow[jj] is false
      f[ii][1] += f_app[1];
      f[ii][2] += f_app[2];

      //calculate added energy, since more costly, only do this if requested
      if (workflag)
        peAdded += f_app[0]*v[ii][0] + f_app[1]*v[ii][1] + f_app[2]*v[ii][2];
    }

  //finish calculation of work done, sum over all procs
  if (workflag) {
    double pe_tmp=0.0;
    MPI_Allreduce(&peAdded,&pe_tmp,1,MPI_DOUBLE,MPI_SUM,world);
    pe_tot += pe_tmp;
  }

}

void FixFlowPump::post_force_respa(int vflag, int ilevel, int /*iloop*/)
{
  if (ilevel == ilevel_respa) post_force(vflag);
}

/* ----------------------------------------------------------------------
   negative of work done by this fix
   This is only computed if requested, either with fix_modify energy yes, or with the energy keyword. Otherwise returns 0.
   ------------------------------------------------------------------------- */
double FixFlowPump::compute_scalar()
{
  return -pe_tot*dt;
}

/* ----------------------------------------------------------------------
   return components of applied force
   ------------------------------------------------------------------------- */
double FixFlowPump::compute_vector(int n)
{
  return -f_tot[n];
}
