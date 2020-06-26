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

/* ----------------------------------------------------------------------
   Contributing author: Mohamed T. Elewa (KIT)
------------------------------------------------------------------------- */

#ifdef FIX_CLASS

FixStyle(avevelocity,FixAveVelocity)

#else

#ifndef LMP_FIX_AVEVELOCITY_H
#define LMP_FIX_AVEVELOCITY_H

#include "fix.h"

namespace LAMMPS_NS {

class FixAveVelocity : public Fix {
 public:
  FixAveVelocity(class LAMMPS *, int, char **);
  ~FixAveVelocity();
  int setmask();
  void init();
  void setup(int);
  void initial_integrate(int);
  //void min_setup(int);
  //void post_force(int);
  //void min_post_force(int);
  double compute_vector(int);

 private:
  double xvalue,yvalue,zvalue;
  int varflag;
  char *xstr,*ystr,*zstr;
  //char *idregion;
  int xvar,yvar,zvar,xstyle,ystyle,zstyle;
  //int iregion;
  double voriginal_all[4];
};

}

#endif
#endif


/* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

E: Region ID for fix velocity does not exist

Self-explanatory.

E: Variable name for fix velocity does not exist

Self-explanatory.

E: Variable for fix velocity is invalid style

Only equal-style variables can be used.

*/

