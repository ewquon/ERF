#include <ERF_Diffusion.H>

using namespace amrex;

/**
 * Function for computing the stress with constant viscosity and without terrain.
 *
 * @param[in] bxcc cell center box for tau_ii
 * @param[in] tbxxy nodal xy box for tau_12
 * @param[in] tbxxz nodal xz box for tau_13
 * @param[in] tbxyz nodal yz box for tau_23
 * @param[in] mu_eff constant molecular viscosity
 * @param[in] cell_data to access rho if ConstantAlpha
 * @param[in,out] tau11 11 strain -> stress
 * @param[in,out] tau22 22 strain -> stress
 * @param[in,out] tau33 33 strain -> stress
 * @param[in,out] tau12 12 strain -> stress
 * @param[in,out] tau13 13 strain -> stress
 * @param[in,out] tau23 23 strain -> stress
 * @param[in] er_arr expansion rate
 */
void
ComputeStressConsVisc_N (Box bxcc, Box tbxxy, Box tbxxz, Box tbxyz, Real mu_eff,
                         const Array4<const Real>& cell_data,
                         Array4<Real>& tau11, Array4<Real>& tau22, Array4<Real>& tau33,
                         Array4<Real>& tau12, Array4<Real>& tau13, Array4<Real>& tau23,
                         const Array4<const Real>& er_arr)
{
    Real OneThird   = (1./3.);

    if (cell_data)
    // constant alpha (stored in mu_eff)
    {
        // Cell centered strains
        ParallelFor(bxcc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real rhoAlpha  = cell_data(i, j, k, Rho_comp) * mu_eff;
            tau11(i,j,k) = -rhoAlpha * ( tau11(i,j,k) - OneThird*er_arr(i,j,k) );
            tau22(i,j,k) = -rhoAlpha * ( tau22(i,j,k) - OneThird*er_arr(i,j,k) );
            tau33(i,j,k) = -rhoAlpha * ( tau33(i,j,k) - OneThird*er_arr(i,j,k) );
        });

        // Off-diagonal strains
        ParallelFor(tbxxy,tbxxz,tbxyz,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real rho_bar = 0.25*( cell_data(i-1, j  , k, Rho_comp) + cell_data(i, j  , k, Rho_comp)
                                + cell_data(i-1, j-1, k, Rho_comp) + cell_data(i, j-1, k, Rho_comp) );
            tau12(i,j,k) *= -rho_bar * mu_eff;
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real rho_bar = 0.25*( cell_data(i-1, j, k  , Rho_comp) + cell_data(i, j, k  , Rho_comp)
                                + cell_data(i-1, j, k-1, Rho_comp) + cell_data(i, j, k-1, Rho_comp) );
            tau13(i,j,k) *= -rho_bar * mu_eff;
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real rho_bar = 0.25*( cell_data(i, j-1, k  , Rho_comp) + cell_data(i, j, k  , Rho_comp)
                                + cell_data(i, j-1, k-1, Rho_comp) + cell_data(i, j, k-1, Rho_comp) );
            tau23(i,j,k) *= -rho_bar * mu_eff;
        });
    }
    else
    // constant mu_eff
    {
        // Cell centered strains
        ParallelFor(bxcc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            tau11(i,j,k) = -mu_eff * ( tau11(i,j,k) - OneThird*er_arr(i,j,k) );
            tau22(i,j,k) = -mu_eff * ( tau22(i,j,k) - OneThird*er_arr(i,j,k) );
            tau33(i,j,k) = -mu_eff * ( tau33(i,j,k) - OneThird*er_arr(i,j,k) );
        });

        // Off-diagonal strains
        ParallelFor(tbxxy,tbxxz,tbxyz,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            tau12(i,j,k) *= -mu_eff;
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            tau13(i,j,k) *= -mu_eff;
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            tau23(i,j,k) *= -mu_eff;
        });
    }
}

/**
 * Function for computing the stress with variable viscosity and without terrain.
 *
 * @param[in] bxcc cell center box for tau_ii
 * @param[in] tbxxy nodal xy box for tau_12
 * @param[in] tbxxz nodal xz box for tau_13
 * @param[in] tbxyz nodal yz box for tau_23
 * @param[in] mu_eff constant molecular viscosity
 * @param[in] mu_turb variable turbulent viscosity
 * @param[in] cell_data to access rho if ConstantAlpha
 * @param[in,out] tau11 11 strain -> stress
 * @param[in,out] tau22 22 strain -> stress
 * @param[in,out] tau33 33 strain -> stress
 * @param[in,out] tau12 12 strain -> stress
 * @param[in,out] tau13 13 strain -> stress
 * @param[in,out] tau23 23 strain -> stress
 * @param[in,out] tau12_op 12 strain -> stress (for thinbody faces)
 * @param[in,out] tau13_op 13 strain -> stress (for thinbody faces)
 * @param[in,out] tau23_op 23 strain -> stress (for thinbody faces)
 * @param[in] er_arr expansion rate
 */
void
ComputeStressVarVisc_N (Box bxcc, Box tbxxy, Box tbxxz, Box tbxyz, Real mu_eff,
                        const Array4<const Real>& mu_turb,
                        const Array4<const Real>& cell_data,
                        Array4<Real>& tau11, Array4<Real>& tau22, Array4<Real>& tau33,
                        Array4<Real>& tau12, Array4<Real>& tau13, Array4<Real>& tau23,
                        Array4<Real>& tau12_op, Array4<Real>& tau13_op, Array4<Real>& tau23_op,
                        const Array4<const Real>& er_arr,
                        const ThinImmersedBody& thinbody)
{
    Real OneThird   = (1./3.);

    if (cell_data)
    // constant alpha (stored in mu_eff)
    {
        // Cell centered strains
        ParallelFor(bxcc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real rhoAlpha = cell_data(i, j, k, Rho_comp) * mu_eff;
            Real mu_11 = rhoAlpha + 2.0 * mu_turb(i, j, k, EddyDiff::Mom_h);
            Real mu_22 = mu_11;
            Real mu_33 = rhoAlpha + 2.0 * mu_turb(i, j, k, EddyDiff::Mom_v);
            tau11(i,j,k) = -mu_11 * ( tau11(i,j,k) - OneThird*er_arr(i,j,k) );
            tau22(i,j,k) = -mu_22 * ( tau22(i,j,k) - OneThird*er_arr(i,j,k) );
            tau33(i,j,k) = -mu_33 * ( tau33(i,j,k) - OneThird*er_arr(i,j,k) );
        });

        // Off-diagonal strains
        ParallelFor(tbxxy,tbxxz,tbxyz,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real rho_bar = 0.25*( cell_data(i-1, j  , k, Rho_comp) + cell_data(i, j  , k, Rho_comp)
                                + cell_data(i-1, j-1, k, Rho_comp) + cell_data(i, j-1, k, Rho_comp) );
            Real mu_bar = 0.25*( mu_turb(i-1, j  , k, EddyDiff::Mom_h) + mu_turb(i, j  , k, EddyDiff::Mom_h)
                               + mu_turb(i-1, j-1, k, EddyDiff::Mom_h) + mu_turb(i, j-1, k, EddyDiff::Mom_h) );
            Real mu_12  = rho_bar*mu_eff + 2.0*mu_bar;
            tau12(i,j,k) *= -mu_12;
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real rho_bar = 0.25*( cell_data(i-1, j, k  , Rho_comp) + cell_data(i, j, k  , Rho_comp)
                                + cell_data(i-1, j, k-1, Rho_comp) + cell_data(i, j, k-1, Rho_comp) );
            Real mu_bar = 0.25*( mu_turb(i-1, j, k  , EddyDiff::Mom_v) + mu_turb(i, j, k  , EddyDiff::Mom_v)
                               + mu_turb(i-1, j, k-1, EddyDiff::Mom_v) + mu_turb(i, j, k-1, EddyDiff::Mom_v) );
            Real mu_13  = rho_bar*mu_eff + 2.0*mu_bar;
            tau13(i,j,k) *= -mu_13;
        },
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real rho_bar = 0.25*( cell_data(i, j-1, k  , Rho_comp) + cell_data(i, j, k  , Rho_comp)
                                + cell_data(i, j-1, k-1, Rho_comp) + cell_data(i, j, k-1, Rho_comp) );
            Real mu_bar = 0.25*( mu_turb(i, j-1, k  , EddyDiff::Mom_v) + mu_turb(i, j, k  , EddyDiff::Mom_v)
                               + mu_turb(i, j-1, k-1, EddyDiff::Mom_v) + mu_turb(i, j, k-1, EddyDiff::Mom_v) );
            Real mu_23  = rho_bar*mu_eff + 2.0*mu_bar;
            tau23(i,j,k) *= -mu_23;
        });
    }
    else
    // constant mu_eff
    {
        // Cell centered strains
        ParallelFor(bxcc, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
            Real mu_11 = mu_eff + 2.0 * mu_turb(i, j, k, EddyDiff::Mom_h);
            Real mu_22 = mu_11;
            Real mu_33 = mu_eff + 2.0 * mu_turb(i, j, k, EddyDiff::Mom_v);
            tau11(i,j,k) = -mu_11 * ( tau11(i,j,k) - OneThird*er_arr(i,j,k) );
            tau22(i,j,k) = -mu_22 * ( tau22(i,j,k) - OneThird*er_arr(i,j,k) );
            tau33(i,j,k) = -mu_33 * ( tau33(i,j,k) - OneThird*er_arr(i,j,k) );
        });

        if (!thinbody) {
            // Off-diagonal strains -- default calculation
            ParallelFor(tbxxy,tbxxz,tbxyz,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                Real mu_bar = 0.25*( mu_turb(i-1, j  , k, EddyDiff::Mom_h) + mu_turb(i, j  , k, EddyDiff::Mom_h)
                                   + mu_turb(i-1, j-1, k, EddyDiff::Mom_h) + mu_turb(i, j-1, k, EddyDiff::Mom_h) );
                Real mu_12  = mu_eff + 2.0*mu_bar;
                tau12(i,j,k) *= -mu_12;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                Real mu_bar = 0.25*( mu_turb(i-1, j, k  , EddyDiff::Mom_v) + mu_turb(i, j, k  , EddyDiff::Mom_v)
                                   + mu_turb(i-1, j, k-1, EddyDiff::Mom_v) + mu_turb(i, j, k-1, EddyDiff::Mom_v) );
                Real mu_13  = mu_eff + 2.0*mu_bar;
                tau13(i,j,k) *= -mu_13;
            },
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                Real mu_bar = 0.25*( mu_turb(i, j-1, k  , EddyDiff::Mom_v) + mu_turb(i, j, k  , EddyDiff::Mom_v)
                                   + mu_turb(i, j-1, k-1, EddyDiff::Mom_v) + mu_turb(i, j, k-1, EddyDiff::Mom_v) );
                Real mu_23  = mu_eff + 2.0*mu_bar;
                tau23(i,j,k) *= -mu_23;
            });
        } else {
            // Use one-sided strains if we have thin bodies
            if (thinbody.have_xfaces || thinbody.have_yfaces) {
                const auto& tb_xfaces = thinbody.xfacelist_d;
                const auto& tb_yfaces = thinbody.yfacelist_d;

                //ParallelFor(tbxxy,tbxxz,tbxyz,
                ParallelFor(tbxxy,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept {
                    const int diffidx = EddyDiff::Mom_h;
                    int next_to_xface = ( is_touching_thin_body_xface(i  , j  , k, tb_xfaces) ||
                                          is_touching_thin_body_xface(i  , j-1, k, tb_xfaces) );
                    int next_to_yface = ( is_touching_thin_body_yface(i  , j  , k, tb_yfaces) ||
                                          is_touching_thin_body_yface(i-1, j  , k, tb_yfaces) );
                    if ((next_to_xface==0) && (next_to_yface)==0) {
                        // default, no thin body
                        Real mu_bar = 0.25*( mu_turb(i-1, j  , k, diffidx) + mu_turb(i, j  , k, diffidx)
                                           + mu_turb(i-1, j-1, k, diffidx) + mu_turb(i, j-1, k, diffidx) );
                        tau12(i,j,k) *= -(mu_eff + 2.0*mu_bar);
                    } else if (next_to_xface < 0) {
                        // thin body on low x-face
                        Real mu_bar = 0.5*( mu_turb(i, j, k, diffidx) + mu_turb(i, j-1, k, diffidx) );
                        tau12(i,j,k) *= -(mu_eff + 2.0*mu_bar);
                    } else if (next_to_xface > 0) {
                        // thin body on high x-face
                        Real mu_bar = 0.5*( mu_turb(i, j, k, diffidx) + mu_turb(i, j-1, k, diffidx) );
                        tau12_op(i+1,j,k) *= -(mu_eff + 2.0*mu_bar);
                    } else if (next_to_yface < 0) {
                        // thin body on low y-face
                        Real mu_bar = 0.5*( mu_turb(i, j, k, diffidx) + mu_turb(i-1, j, k, diffidx) );
                        tau12(i,j,k) *= -(mu_eff + 2.0*mu_bar);
                    } else if (next_to_yface > 0) {
                        // thin body on high y-face
                        Real mu_bar = 0.5*( mu_turb(i, j, k, diffidx) + mu_turb(i-1, j, k, diffidx) );
                        tau12_op(i,j+1,k) *= -(mu_eff + 2.0*mu_bar);
                    }
                });
            }
        }
    }
}
