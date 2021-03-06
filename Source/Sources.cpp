#include "ERF.H"
#include "IndexDefines.H"

void
ERF::construct_old_source(
  int src,
  amrex::Real time,
  amrex::Real dt,
  int amr_iteration,
  int amr_ncycle,
  int sub_iteration,
  int sub_ncycle)
{
  AMREX_ASSERT(src >= 0 && src < num_src);

  switch (src) {

  case ext_src:
    construct_old_ext_source(time, dt);
    break;

  case forcing_src:
    construct_old_forcing_source(time, dt);
    break;

#ifdef ERF_USE_MASA
  case mms_src:
    construct_old_mms_source(time);
    break;
#endif
  } // end switch
}

void
ERF::construct_new_source(
  int src,
  amrex::Real time,
  amrex::Real dt,
  int amr_iteration,
  int amr_ncycle,
  int sub_iteration,
  int sub_ncycle)
{
  AMREX_ASSERT(src >= 0 && src < num_src);

  switch (src) {

  case ext_src:
    construct_new_ext_source(time, dt);
    break;

  case forcing_src:
    construct_new_forcing_source(time, dt);
    break;

#ifdef ERF_USE_MASA
  case mms_src:
    construct_new_mms_source(time);
    break;
#endif
  } // end switch
}

// Obtain the sum of all source terms.
void
ERF::sum_of_sources(amrex::MultiFab& source)
{
  int ng = source.nGrow();

  source.setVal(0.0);

  for (int n = 0; n < src_list.size(); ++n) {
    amrex::MultiFab::Add(source, *old_sources[src_list[n]], 0, 0, NVAR, ng);
  }

  for (int n = 0; n < src_list.size(); ++n) {
    amrex::MultiFab::Add(source, *new_sources[src_list[n]], 0, 0, NVAR, ng);
  }
}
