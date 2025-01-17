#include <AMReX.H>
#include <ERF_Diffusion.H>
#include <ERF_IndexDefines.H>

using namespace amrex;

/**
 * Function for computing the momentum RHS for diffusion operator without terrain.
 *
 * @param[in]  bxx nodal x box for x-mom
 * @param[in]  bxy nodal y box for y-mom
 * @param[in]  bxz nodal z box for z-mom
 * @param[out] rho_u_rhs RHS for x-mom
 * @param[out] rho_v_rhs RHS for y-mom
 * @param[out] rho_w_rhs RHS for z-mom
 * @param[in]  tau11 11 stress
 * @param[in]  tau22 22 stress
 * @param[in]  tau33 33 stress
 * @param[in]  tau12 12 stress
 * @param[in]  tau13 13 stress
 * @param[in]  tau23 23 stress
 * @param[in]  tau12_op 12 stress (for thin-body faces)
 * @param[in]  tau13_op 13 stress (for thin-body faces)
 * @param[in]  tau23_op 23 stress (for thin-body faces)
 * @param[in]  dxInv inverse cell size array
 * @param[in]  mf_m map factor at cell center
 */
void
DiffusionSrcForMom_N (const Box& bxx, const Box& bxy , const Box& bxz,
                      const Array4<Real>& rho_u_rhs  ,
                      const Array4<Real>& rho_v_rhs  ,
                      const Array4<Real>& rho_w_rhs  ,
                      const Array4<const Real>& tau11, const Array4<const Real>& tau22,
                      const Array4<const Real>& tau33, const Array4<const Real>& tau12,
                      const Array4<const Real>& tau13, const Array4<const Real>& tau23,
                      const Array4<const Real>& tau12_op ,
                      const Array4<const Real>& tau13_op ,
                      const Array4<const Real>& tau23_op ,
                      const GpuArray<Real, AMREX_SPACEDIM>& dxInv,
                      const Array4<const Real>& mf_m,
                      const Array4<const Real>& /*mf_u*/,
                      const Array4<const Real>& /*mf_v*/,
                      const ThinImmersedBody& thinbody)
{
    BL_PROFILE_VAR("DiffusionSrcForMom_N()",DiffusionSrcForMom_N);

    auto dxinv = dxInv[0], dyinv = dxInv[1], dzinv = dxInv[2];

//    if (!thinbody) {
        // default calculation w/o thin bodies
        ParallelFor(bxx, bxy, bxz,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real mf   = mf_m(i,j,0);

            rho_u_rhs(i,j,k) -= ( (tau11(i  , j  , k  ) - tau11(i-1, j  ,k  )) * dxinv * mf   // Contribution to x-mom eqn from diffusive flux in x-dir
                                + (tau12(i  , j+1, k  ) - tau12(i  , j  ,k  )) * dyinv * mf   // Contribution to x-mom eqn from diffusive flux in y-dir
                                + (tau13(i  , j  , k+1) - tau13(i  , j  ,k  )) * dzinv );     // Contribution to x-mom eqn from diffusive flux in z-dir;
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real mf   = mf_m(i,j,0);

            rho_v_rhs(i,j,k) -= ( (tau12(i+1, j  , k  ) - tau12(i  , j  , k  )) * dxinv * mf   // Contribution to y-mom eqn from diffusive flux in x-dir
                                + (tau22(i  , j  , k  ) - tau22(i  , j-1, k  )) * dyinv * mf   // Contribution to y-mom eqn from diffusive flux in y-dir
                                + (tau23(i  , j  , k+1) - tau23(i  , j  , k  )) * dzinv );     // Contribution to y-mom eqn from diffusive flux in z-dir;
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real mf   = mf_m(i,j,0);

            rho_w_rhs(i,j,k) -= ( (tau13(i+1, j  , k  ) - tau13(i  , j  , k  )) * dxinv * mf   // Contribution to z-mom eqn from diffusive flux in x-dir
                                + (tau23(i  , j+1, k  ) - tau23(i  , j  , k  )) * dyinv * mf   // Contribution to z-mom eqn from diffusive flux in y-dir
                                + (tau33(i  , j  , k  ) - tau33(i  , j  , k-1)) * dzinv );     // Contribution to z-mom eqn from diffusive flux in z-dir;
        });
#if 0
    } else {
        const auto& tb_xfaces = thinbody.xfacelist_d;
        const auto& tb_yfaces = thinbody.yfacelist_d;
        const auto& tb_zfaces = thinbody.zfacelist_d;

        // For div(tau) at cell centers ~ tau_hi - tau_lo:
        // - If we are on the high side of a thin-body face (indicated by
        //   `next_to_*face` < 0), then the one-sided shear stress on the
        //   thin-body surface is stored in `tau**` and corresponds to tau_lo.
        // - If we are on the low side of a thin-body face (indicated by
        //   `next_to_*face` > 0), then the one-sided shear stress on the
        //   thin-body surface is stored in `tau**_op` and corresponds to
        //   tau_hi.
        // - Otherwise, we are at a standard interior cell (indicated by
        //   `next_to_face` == 0) and both tau_lo and tau_hi correspond to
        //   unmodified values in `tau**`.
        // Note that `rho_*_rhs` coincides with the staggered momenta and
        // `tauij` are each staggered in the i and j directions.

        ParallelFor(bxx, bxy, bxz,
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real mf   = mf_m(i,j,0);

            int next_to_yface = is_touching_thin_body_yface(i, j, k, tb_yfaces, 0); // also check i-1
          //int next_to_zface = is_touching_thin_body_zface(i, j, k, tb_zfaces, 0); // also check i-1
#if 0
            // HACK -- exclude edge pts
            if ((i<=123) ||
                (i>=128) ||
                (j<124)  ||
                (j>125)) {
                next_to_yface = 0;
            }
#endif

            Real tau12_hi = (next_to_yface > 0) ? tau12_op(i, j+1, k  ) : tau12(i, j+1, k  );
          //Real tau13_hi = (next_to_zface > 0) ? tau13_op(i, j  , k+1) : tau13(i, j  , k+1);

            rho_u_rhs(i,j,k) -= ( (tau11(i  , j  , k  ) - tau11(i-1, j  ,k  )) * dxinv * mf   // Contribution to x-mom eqn from diffusive flux in x-dir
                                + (tau12_hi             - tau12(i  , j  ,k  )) * dyinv * mf   // Contribution to x-mom eqn from diffusive flux in y-dir
          //                    + (tau13_hi             - tau13(i  , j  ,k  )) * dzinv );     // Contribution to x-mom eqn from diffusive flux in z-dir;
                                + (tau13(i  , j  , k+1) - tau13(i  , j  ,k  )) * dzinv );     // Contribution to x-mom eqn from diffusive flux in z-dir;
            if ( (i>=123) && (i<=128) &&
                 ((j==124)||(j==125)) ) {
                AllPrint() << "rho_u_rhs" << IntVect(i,j,k) << " = " << rho_u_rhs(i,j,k)
                    << " tau11_hi=" << tau11(i,j,k)
                    << " tau11_lo=" << tau11(i-1,j,k)
                    << " tau12_hi=" << tau12_hi
                    << " tau12_lo=" << tau12(i,j,k)
                    //<< " tau13_hi=" << tau13(i,j,k+1)
                    //<< " tau13_lo=" << tau13(i,j,k)
                    << std::endl;
            }
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real mf   = mf_m(i,j,0);

            rho_v_rhs(i,j,k) -= ( (tau12(i+1, j  , k  ) - tau12(i  , j  , k  )) * dxinv * mf   // Contribution to y-mom eqn from diffusive flux in x-dir
                                + (tau22(i  , j  , k  ) - tau22(i  , j-1, k  )) * dyinv * mf   // Contribution to y-mom eqn from diffusive flux in y-dir
                                + (tau23(i  , j  , k+1) - tau23(i  , j  , k  )) * dzinv );     // Contribution to y-mom eqn from diffusive flux in z-dir;
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k)
        {
            Real mf   = mf_m(i,j,0);

            rho_w_rhs(i,j,k) -= ( (tau13(i+1, j  , k  ) - tau13(i  , j  , k  )) * dxinv * mf   // Contribution to z-mom eqn from diffusive flux in x-dir
                                + (tau23(i  , j+1, k  ) - tau23(i  , j  , k  )) * dyinv * mf   // Contribution to z-mom eqn from diffusive flux in y-dir
                                + (tau33(i  , j  , k  ) - tau33(i  , j  , k-1)) * dzinv );     // Contribution to z-mom eqn from diffusive flux in z-dir;
        });
    }
#endif
}
