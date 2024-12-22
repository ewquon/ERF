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
 * @param[in] xmom_src source terms for x-momentum
 * @param[in] ymom_src source terms for y-momentum
 * @param[in] zmom_src source terms for z-momentum
 * @param[in] thinbody thin immersed body data structure
 */

void add_thin_body_sources (int level,
                            MultiFab& xmom_src,
                            MultiFab& ymom_src,
                            MultiFab& zmom_src,
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
        MultiFab::Copy(*thinbody.fx[level], xmom_src, 0, 0, 1, 0);
        thinbody.fx[level]->mult(-1., 0, 1, 0);
        ApplyInvertedMask(*thinbody.fx[level], *thinbody.xflux_imask[level], 0);
        MultiFab::Add(xmom_src, *thinbody.fx[level], 0, 0, 1, 0);
    }

    if (l_have_thin_yforce) {
        MultiFab::Copy(*thinbody.fy[level], ymom_src, 0, 0, 1, 0);
        thinbody.fy[level]->mult(-1., 0, 1, 0);
        ApplyInvertedMask(*thinbody.fy[level], *thinbody.yflux_imask[level], 0);
        MultiFab::Add(ymom_src, *thinbody.fy[level], 0, 0, 1, 0);
    }

    if (l_have_thin_zforce) {
        MultiFab::Copy(*thinbody.fz[level], zmom_src, 0, 0, 1, 0);
        thinbody.fz[level]->mult(-1., 0, 1, 0);
        ApplyInvertedMask(*thinbody.fz[level], *thinbody.zflux_imask[level], 0);
        MultiFab::Add(zmom_src, *thinbody.fz[level], 0, 0, 1, 0);
    }

#if 0
#ifndef AMREX_USE_GPU
    if (l_have_thin_xforce) {
        // TODO: Implement particles to better track and output these data
        if (nrk==2) {
            for ( MFIter mfi(S_data[IntVars::cons],TileNoZ()); mfi.isValid(); ++mfi)
            {
                const Box& tbx = mfi.nodaltilebox(0);
                const Array4<const Real> & fx = thin_xforce_lev->const_array(mfi);
                const Array4<const int> & mask = xflux_imask_lev->const_array(mfi);
                ParallelFor(tbx, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (mask(i,j,k)==0) {
                        amrex::AllPrint() << "thin body fx"<<IntVect(i,j,k)<<" = " << fx(i,j,k) << std::endl;
                    }
                });
            }
        }
    }
#endif
#endif

#if 0
#ifndef AMREX_USE_GPU
    if (l_have_thin_yforce) {
        // TODO: Implement particles to better track and output these data
        if (nrk==2) {
            for ( MFIter mfi(S_data[IntVars::cons],TileNoZ()); mfi.isValid(); ++mfi)
            {
                const Box& tby = mfi.nodaltilebox(1);
                const Array4<const Real> & fy = thin_yforce_lev->const_array(mfi);
                const Array4<const int> & mask = yflux_imask_lev->const_array(mfi);
                ParallelFor(tby, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (mask(i,j,k)==0) {
                        amrex::AllPrint() << "thin body fy"<<IntVect(i,j,k)<<" = " << fy(i,j,k) << std::endl;
                    }
                });
            }
        }
    }
#endif
#endif

#if 0
#ifndef AMREX_USE_GPU
    if (l_have_thin_zforce) {
        // TODO: Implement particles to better track and output these data
        if (nrk==2) {
            for ( MFIter mfi(S_data[IntVars::cons],TileNoZ()); mfi.isValid(); ++mfi)
            {
                const Box& tbz = mfi.nodaltilebox(2);
                const Array4<const Real> & fz = thin_zforce_lev->const_array(mfi);
                const Array4<const int> & mask = zflux_imask_lev->const_array(mfi);
                ParallelFor(tbz, [=] AMREX_GPU_DEVICE (int i, int j, int k) {
                    if (mask(i,j,k)==0) {
                        amrex::AllPrint() << "thin body fz"<<IntVect(i,j,k)<<" = " << fz(i,j,k) << std::endl;
                    }
                });
            }
        }
    }
#endif
#endif
}
