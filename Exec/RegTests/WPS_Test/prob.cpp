#include "prob.H"
#include "prob_common.H"

#include "ERF_Constants.H"

using namespace amrex;

ProbParm parms;

#include "Prob/init_constant_density_hse.H"

void
amrex_probinit(
  const amrex_real* /*problo*/,
  const amrex_real* /*probhi*/)
{
  // Parse params
  amrex::ParmParse pp("prob");
  pp.query("rho_0", parms.rho_0);
  pp.query("T_0", parms.Theta_0);
}

void
erf_init_rayleigh(Vector<Real>& /*tau*/,
                  Vector<Real>& /*ubar*/,
                  Vector<Real>& /*vbar*/,
                  Vector<Real>& /*wbar*/,
                  Vector<Real>& /*thetabar*/,
                  Geometry      const& /*geom*/)
{
   amrex::Error("Should never get here for WPS tests problem");
}

void
init_custom_prob(
    const Box& /*bx*/,
    const Box& /*xbx*/,
    const Box& /*ybx*/,
    const Box& /*zbx*/,
    Array4<Real      > const&,
    Array4<Real      > const&,
    Array4<Real      > const&,
    Array4<Real      > const&,
    Array4<Real      > const&,
    Array4<Real      > const&,
    Array4<Real const> const&,
    Array4<Real const> const&,
#if defined(ERF_USE_MOISTURE)
    Array4<Real      > const&,
    Array4<Real      > const&,
    Array4<Real      > const&,
#elif defined(ERF_USE_WARM_NO_PRECIP)
    Array4<Real      > const&,
    Array4<Real      > const&,
#endif
    GeometryData const&,
    Array4<Real const> const&,
    Array4<Real const> const&,
    Array4<Real const> const&,
    const SolverChoice&)
{
   // NOTE: this adds perturbations to what has already been defined; here we add nothing
}

void
init_custom_terrain (const Geometry& /*geom*/,
                           MultiFab& /*z_phys_nd*/,
                     const Real& /*time*/)
{
    amrex::Error("We don't belong in init_custom_terrain!");
}
