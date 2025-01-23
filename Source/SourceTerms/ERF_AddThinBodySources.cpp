#include <AMReX_MultiFab.H>
#include <AMReX_ArrayLim.H>
#include <AMReX_BCRec.H>
#include <AMReX_TableData.H>
#include <AMReX_GpuContainers.H>

#include <ERF_NumericalDiffusion.H>
#include <ERF_PlaneAverage.H>
#include <ERF_TI_slow_headers.H>
#include <ERF_SrcHeaders.H>
#include <ERF_Utils.H>

using namespace amrex;

/**
 * Add forcing terms on the thin immersed body, equal to the negative of the
 * momentum source; when added, this negates the RHS to achieve zero flux.
 *
 * @param[in] level
 * @param[in] RHS
 * @param[in] thinbody thin immersed body data structure
 */

void add_thin_body_sources (int level,
                            Vector<MultiFab>& S_rhs,
                            ThinImmersedBody& thinbody)
{
    BL_PROFILE_REGION("erf_add_thin_body_sources()");

    const bool l_have_thin_xforce = (thinbody.fx[level] != nullptr);
    const bool l_have_thin_yforce = (thinbody.fy[level] != nullptr);
    const bool l_have_thin_zforce = (thinbody.fz[level] != nullptr);

    // *****************************************************************************
    // If a thin immersed body is present, add forcing terms
    // *****************************************************************************
    if (l_have_thin_xforce) {
        MultiFab::Copy(*thinbody.fx[level], S_rhs[IntVars::xmom], 0, 0, 1, 0);
        thinbody.fx[level]->mult(-1., 0, 1, 0);
        ApplyInvertedMask(*thinbody.fx[level], *thinbody.xflux_imask[level], 0);
        ApplyMask(S_rhs[IntVars::xmom], *thinbody.xflux_imask[level], 0);
    }

    if (l_have_thin_yforce) {
        MultiFab::Copy(*thinbody.fy[level], S_rhs[IntVars::ymom], 0, 0, 1, 0);
        thinbody.fy[level]->mult(-1., 0, 1, 0);
        ApplyInvertedMask(*thinbody.fy[level], *thinbody.yflux_imask[level], 0);
        ApplyMask(S_rhs[IntVars::ymom], *thinbody.yflux_imask[level], 0);
    }

    if (l_have_thin_zforce) {
        MultiFab::Copy(*thinbody.fz[level], S_rhs[IntVars::zmom], 0, 0, 1, 0);
        thinbody.fz[level]->mult(-1., 0, 1, 0);
        ApplyInvertedMask(*thinbody.fz[level], *thinbody.zflux_imask[level], 0);
        ApplyMask(S_rhs[IntVars::zmom], *thinbody.zflux_imask[level], 0);
    }
}
