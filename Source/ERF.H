#ifndef _ERF_H_
#define _ERF_H_

#include <iostream>

#include <AMReX_BC_TYPES.H>
#include <AMReX_AmrLevel.H>
#include <AMReX_iMultiFab.H>

#include <AMReX_YAFluxRegister.H>

#ifdef AMREX_LAZY
#include <AMReX_Lazy.H>
#endif

#include "IndexDefines.H"

using std::istream;
using std::ostream;

#define UserBC 6

enum StateType {
  State_Type = 0,
  Vel_x,
  Vel_y,
  Vel_z,
  State_x,
  State_y,
  State_z,
  Work_Estimate_Type
};

// Create storage for all source terms.

enum sources {
  ext_src = 0,
  forcing_src,
  diff_src,
  les_src,
#ifdef ERF_USE_MASA
  mms_src,
#endif
  num_src
};

static amrex::Box
the_same_box(const amrex::Box& b)
{
  return b;
}
static amrex::Box
grow_box_by_one(const amrex::Box& b)
{
  return amrex::grow(b, 1);
}

//
// AmrLevel-derived class for hyperbolic conservation equations for stellar
// media
//

class ERF : public amrex::AmrLevel
{
public:
  //
  // Default constructor.  Builds invalid object.
  //
  ERF();
  //
  // The basic constructor.
  //
  ERF(
    amrex::Amr& papa,
    int lev,
    const amrex::Geometry& level_geom,
    const amrex::BoxArray& bl,
    const amrex::DistributionMapping& dm,
    amrex::Real time);
  //
  // The destructor.
  //
  virtual ~ERF();
  //
  // Restart from a checkpoint file.
  //
  virtual void
  restart(amrex::Amr& papa, istream& is, bool bReadSpecial = false) override;
  //
  // This is called only when we restart from an old checkpoint.
  //
  virtual void
  set_state_in_checkpoint(amrex::Vector<int>& state_in_checkpoint) override;
  //
  // Call AmrLevel::checkPoint and then add radiation info
  //
  virtual void checkPoint(
    const std::string& dir,
    std::ostream& os,
    amrex::VisMF::How how,
    bool dump_old) override;

  virtual void setPlotVariables() override;
  //
  // Write a plotfile to specified directory.
  //
  virtual void writePlotFile(
    const std::string& dir, ostream& os, amrex::VisMF::How how) override;
  virtual void writeSmallPlotFile(
    const std::string& dir, ostream& os, amrex::VisMF::How how) override;
  void writeJobInfo(const std::string& dir);
  static void writeBuildInfo(std::ostream& os);

  //
  // Define data descriptors.
  //
  static void variableSetUp();
  //
  // Cleanup data descriptors at end of run.
  //
  static void variableCleanUp();
  //
  // Initialize grid data at problem start-up.
  //
  virtual void initData() override;

  //
  // Get problem metrics.
  //
  amrex::MultiFab* Area();
  amrex::MultiFab& Area(int dir);
  //
  // The volume of the multifab.
  //
  amrex::MultiFab& Volume();

  // Set time levels of state data.
  //
  virtual void setTimeLevel(
    amrex::Real time, amrex::Real dt_old, amrex::Real dt_new) override;
  //
  // Initialize data on this level from another ERF (during regrid).
  //
  virtual void init(amrex::AmrLevel& old) override;
  //
  // Initialize data on this level after regridding if old level did not
  // previously exist
  //
  virtual void init() override;

  /**
   * Initialize EB geometry for finest_level and level grids for
   * other levels
   */
  // for the Amr class to do timed load balances.
  virtual int WorkEstType() override { return Work_Estimate_Type; }

  //
  // Proceed with next timestep?
  //
  virtual int okToContinue() override;
  //
  // Advance grids at this level in time.
  //
  virtual amrex::Real
  advance(amrex::Real time, amrex::Real dt, int iteration, int ncycle) override;

  amrex::Real do_mol_advance(
    amrex::Real time, amrex::Real dt, int amr_iteration, int amr_ncycle);

  void construct_Snew(
    amrex::MultiFab& S_new, const amrex::MultiFab& S_old, amrex::Real dt);

  void construct_hydro_source(
    const amrex::MultiFab& S,
    amrex::Real time,
    amrex::Real dt,
    int amr_iteration,
    int amr_ncycle,
    int sub_iteration,
    int sub_ncycle);

  void construct_old_source(
    int src,
    amrex::Real time,
    amrex::Real dt,
    int amr_iteration,
    int amr_ncycle,
    int sub_iteration,
    int sub_ncycle);

  void construct_new_source(
    int src,
    amrex::Real time,
    amrex::Real dt,
    int amr_iteration,
    int amr_ncycle,
    int sub_iteration,
    int sub_ncycle);

  void sum_of_sources(amrex::MultiFab& source);

  void construct_old_ext_source(amrex::Real time, amrex::Real dt);

  void construct_new_ext_source(amrex::Real time, amrex::Real dt);

  void fill_ext_source(
    amrex::Real time,
    amrex::Real dt,
    const amrex::MultiFab& S_old,
    const amrex::MultiFab& S_new,
    amrex::MultiFab& ext_src,
    int ng);

  void construct_old_forcing_source(amrex::Real time, amrex::Real dt);

  void construct_new_forcing_source(amrex::Real time, amrex::Real dt);

  void fill_forcing_source(
    amrex::Real time,
    amrex::Real dt,
    const amrex::MultiFab& S_old,
    const amrex::MultiFab& S_new,
    amrex::MultiFab& forcing_src,
    int ng);

#ifdef ERF_USE_MASA
  void construct_old_mms_source(amrex::Real time);

  void construct_new_mms_source(amrex::Real time);

  void fill_mms_source(
    amrex::Real time,
    const amrex::MultiFab& S,
    amrex::MultiFab& mms_src,
    int ng);
#endif

  void getLESTerm(
    amrex::Real time,
    amrex::Real dt,
    amrex::MultiFab& LESTerm,
    amrex::Real flux_factor);

  void getSmagorinskyLESTerm(
    amrex::Real time,
    amrex::Real dt,
    amrex::MultiFab& LESTerm,
    amrex::Real flux_factor);

  void getDynamicSmagorinskyLESTerm(
    amrex::Real time,
    amrex::Real dt,
    amrex::MultiFab& LESTerm,
    amrex::Real flux_factor);

  void construct_old_les_source(
    amrex::Real time, amrex::Real dt, int sub_iteration, int sub_ncycle);

  void construct_new_les_source(
    amrex::Real time, amrex::Real dt, int sub_iteration, int sub_ncycle);

  static void set_active_sources();

  //
  // Estimate time step.
  //
  amrex::Real estTimeStep(amrex::Real dt_old);
  //
  // Compute initial time step.
  //
  amrex::Real initialTimeStep();
  //
  // Compute initial `dt'.
  //
  virtual void computeInitialDt(
    int finest_level,
    int sub_cycle,
    amrex::Vector<int>& n_cycle,
    const amrex::Vector<amrex::IntVect>& ref_ratio,
    amrex::Vector<amrex::Real>& dt_level,
    amrex::Real stop_time) override;
  //
  // Compute new `dt'.
  //
  virtual void computeNewDt(
    int finest_level,
    int sub_cycle,
    amrex::Vector<int>& n_cycle,
    const amrex::Vector<amrex::IntVect>& ref_ratio,
    amrex::Vector<amrex::Real>& dt_min,
    amrex::Vector<amrex::Real>& dt_level,
    amrex::Real stop_time,
    int post_regrid_flag) override;
  //
  // Allocate data at old time.
  //
  virtual void allocOldData() override;
  //
  // Remove data at old time.
  //
  virtual void removeOldData() override;
  //
  // Passes some data about the grid to a Fortran module.
  //
  void setGridInfo();
  //
  // Print information about energy budget.
  //
  void do_energy_diagnostics();
  //
  // Do work after timestep().
  //
  virtual void post_timestep(int iteration) override;
  //
  // Contains operations to be done only after a full coarse timestep.
  //
  virtual void postCoarseTimeStep(amrex::Real cumtime) override;
  //
  // Do work after regrid().
  //
  virtual void post_regrid(int lbase, int new_finest) override;
  //
  // Do work after a restart().
  //
  virtual void post_restart() override;
  //
  // Do work after init().
  //
  virtual void post_init(amrex::Real stop_time) override;
  //
  // Error estimation for regridding.
  //
  virtual void errorEst(
    amrex::TagBoxArray& tb,
    int clearval,
    int tagval,
    amrex::Real time,
    int n_error_buf = 0,
    int ngrow = 0) override;

  // Returns a MultiFab containing the derived data for this level.
  // The user is responsible for deleting this pointer when done
  // with it.  If ngrow>0 the MultiFab is built on the appropriately
  // grown BoxArray.
  virtual std::unique_ptr<amrex::MultiFab>
  derive(const std::string& name, amrex::Real time, int ngrow) override;

  // This version of derive() fills the dcomp'th component of mf with the
  // derived quantity.
  virtual void derive(
    const std::string& name,
    amrex::Real time,
    amrex::MultiFab& mf,
    int dcomp) override;

  static int numGrow();

  void reset_internal_energy(amrex::MultiFab& State, int ng);

  void computeTemp(amrex::MultiFab& State, int ng);

  void getMOLSrcTerm(
    const amrex::MultiFab& S,
    amrex::MultiFab& MOLSrcTerm,
    amrex::Real time,
    amrex::Real dt,
    amrex::Real flux_factor);

  void enforce_consistent_e(amrex::MultiFab& S);

  amrex::Real volWgtSum(
    const std::string& name,
    amrex::Real time,
    bool local = false,
    bool finemask = true);
  amrex::Real volWgtSquaredSum(
    const std::string& name, amrex::Real time, bool local = false);
  amrex::Real volWgtSumMF(
    const amrex::MultiFab& mf,
    int comp,
    bool local = false,
    bool finemask = false);
  amrex::Real
  volWgtSquaredSumDiff(int comp, amrex::Real time, bool local = false);
  amrex::Real
  maxDerive(const std::string& name, amrex::Real time, bool local = false);

  // static int NVAR;
  static int Density, Xmom, Ymom, Zmom, Eden, Eint, Temp;

  static int FirstAdv, NumAdv;
  static int FirstAux;

  /// Index locations into particle state
  static int pstateVel, pstateT, pstateDia, pstateRho, pstateY, pstateNum;

  //
  // This MultiFab is on the coarser level.  This is useful for the coarser
  // level to mask out the finer level.  We only build this when it is needed.
  // This coarse MultiFab has to live on the fine level because it must be
  // updated even when only the fine level changes.
  //
  amrex::MultiFab fine_mask;
  amrex::MultiFab& build_fine_mask();

  amrex::YAFluxRegister flux_reg;

  amrex::YAFluxRegister& getFluxReg();
  amrex::YAFluxRegister& getFluxReg(int lev);

  amrex::YAFluxRegister pres_reg;

  amrex::YAFluxRegister& getPresReg();
  amrex::YAFluxRegister& getPresReg(int lev);

  void avgDown();
  void avgDown(int state_indx);

protected:
  amrex::iMultiFab level_mask;
  //
  // Build a mask that ghost cells overlapping with interior cells in the same
  // multifab are set to 0, whereas others are set to 1.
  //
  amrex::Vector<std::unique_ptr<amrex::iMultiFab>> ib_mask;
  const amrex::iMultiFab* build_interior_boundary_mask(int ng);

  ///
  /// A state array with ghost zones.
  ///
  amrex::MultiFab Sborder;
  ///
  /// Source terms to the hydrodynamics solve.
  ///
  amrex::MultiFab sources_for_hydro;

  ///
  /// Source term representing hydrodynamics update.
  ///
  amrex::MultiFab hydro_source;

  ///
  /// Non-hydro source terms.
  ///
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> old_sources;
  amrex::Vector<std::unique_ptr<amrex::MultiFab>> new_sources;

  static void init_transport();

  void init_les();
  void init_filters();

#ifdef ERF_USE_MASA
  static void init_mms();
#endif

  static void clear_prob();

  static void read_params();

  static void read_tagging_params();

  ERF& getLevel(int lev);

  void reflux();

  amrex::Real
  enforce_min_density(amrex::MultiFab& S_old, amrex::MultiFab& S_new);

  amrex::Real clean_state(amrex::MultiFab& state);

  amrex::Real clean_state(amrex::MultiFab& state, amrex::MultiFab& state_old);

  void buildMetrics();

  /// integrate derived quantities over domain

  amrex::Real
  sumDerive(const std::string& name, amrex::Real time, bool local = false);

  void sum_integrated_quantities();

  void write_info();

  void stopJob();

  //
  // The data.
  //
  amrex::MultiFab volume;
  amrex::MultiFab area[AMREX_SPACEDIM];
  int m_ngrow;
  //
  // Static data members.
  //
#include "erf_params.H"

  static bool signalStopJob;
  static bool dump_old;
  static int verbose;
  static amrex::BCRec phys_bc;

  /// for keeping track of mass changes from negative density resets
  static amrex::Real frac_change;

  /// For keeping track of fluid quantities lost at physical grid boundaries.
  /// This should persist through restarts, but right now only on level 0.
  static const int n_lost = 8;
  amrex::Real material_lost_through_boundary_cumulative[n_lost];
  amrex::Real material_lost_through_boundary_temp[n_lost];

  /// for keeping track of the amount of CPU time used -- this will persist
  /// after restarts
  static amrex::Real previousCPUTimeUsed;
  static amrex::Real startCPUTime;

  static amrex::Real getCPUTime();

  static int diffuse_temp;
  static int diffuse_enth;
  static int diffuse_vel;
  static amrex::Real diffuse_cutoff_density;
  static bool do_diffuse;

#ifdef ERF_USE_MASA
  static bool mms_initialized;
  bool mms_src_evaluated;
  amrex::MultiFab mms_source;
#endif

  bool FillPatchedOldState_ok;

  // There can be only one Diffusion object, it covers all levels:
  static class Diffusion* diffusion;

  static int num_state_type;

  static amrex::Vector<int> src_list;

/* problem-specific includes */
#include <Problem.H>

  static int nGrowTr;

  static bool do_mol_load_balance;
};

void pc_bcfill_hyp(
  amrex::Box const& bx,
  amrex::FArrayBox& data,
  const int dcomp,
  const int numcomp,
  amrex::Geometry const& geom,
  const amrex::Real time,
  const amrex::Vector<amrex::BCRec>& bcr,
  const int bcomp,
  const int scomp);

void pc_nullfill(
  amrex::Box const& bx,
  amrex::FArrayBox& data,
  const int dcomp,
  const int numcomp,
  amrex::Geometry const& geom,
  const amrex::Real time,
  const amrex::Vector<amrex::BCRec>& bcr,
  const int bcomp,
  const int scomp);

//
// Inlines.
//

AMREX_FORCE_INLINE
int
ERF::numGrow()
{
  return NUM_GROW;
}

AMREX_FORCE_INLINE
amrex::MultiFab*
ERF::Area()
{
  return area;
}

AMREX_FORCE_INLINE
amrex::MultiFab&
ERF::Area(int dir)
{
  return area[dir];
}

AMREX_FORCE_INLINE
amrex::MultiFab&
ERF::Volume()
{
  return volume;
}

AMREX_FORCE_INLINE
ERF&
ERF::getLevel(int lev)
{
  return *(ERF*)&parent->getLevel(lev);
}

AMREX_FORCE_INLINE
void
ERF::stopJob()
{
  signalStopJob = true;
}

AMREX_FORCE_INLINE
amrex::YAFluxRegister&
ERF::getFluxReg()
{
  return flux_reg;
}

AMREX_FORCE_INLINE
amrex::YAFluxRegister&
ERF::getFluxReg(int lev)
{
  return getLevel(lev).getFluxReg();
}

AMREX_FORCE_INLINE
amrex::YAFluxRegister&
ERF::getPresReg()
{
  return pres_reg;
}

AMREX_FORCE_INLINE
amrex::YAFluxRegister&
ERF::getPresReg(int lev)
{
  return getLevel(lev).getPresReg();
}

#endif