/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.

   Contributing authors: Steven E. Strong and Joel D. Eaves
   Joel.Eaves@Colorado.edu
   ------------------------------------------------------------------------- */
#ifdef FIX_CLASS
// clang-format off
FixStyle(flow/pump,FixFlowPump);
// clang-format on
#else

#ifndef LMP_FIX_FLOWPUMP_H
#define LMP_FIX_FLOWPUMP_H

#include "fix.h"

  namespace LAMMPS_NS {

    class FixFlowPump : public Fix {
    public:
      FixFlowPump(class LAMMPS *, int, char **);
      ~FixFlowPump();
      int setmask();
      void init();
      void setup(int);
      void post_force(int);
      void post_force_respa(int, int, int);
      double compute_scalar();
      double compute_vector(int n);

    protected:
      int dimension;
      bool flow[3];       //flag if each direction is conserved
      double current[3];  //mass flux
      double a_app[3];    //applied acceleration
      double f_tot[3];    //total applied force
      double pe_tot;      //total added energy
      double xvalue,yvalue,zvalue;
      int varflag;
      char *xstr,*ystr,*zstr;
      int xvar,yvar,zvar,xstyle,ystyle,zstyle;
      double dt;          //timestep
      bool workflag;      //if calculate work done by fix
      int ilevel_respa;

    };

  }

#endif
#endif
