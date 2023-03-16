#include <Diffusion.H>
#include <EddyViscosity.H>
#include <ComputeQKESourceTerm.H>

using namespace amrex;

void
DiffusionSrcForState_N (const amrex::Box& bx, const amrex::Box& domain, int n_start, int n_end,
                        const Array4<const Real>& u,
                        const Array4<const Real>& v,
                        const Array4<const Real>& w,
                        const Array4<const Real>& cell_data,
                        const Array4<const Real>& cell_prim,
                        const Array4<Real>& cell_rhs,
                        const Array4<Real>& xflux,
                        const Array4<Real>& yflux,
                        const Array4<Real>& zflux,
                        const amrex::GpuArray<Real, AMREX_SPACEDIM>& cellSizeInv,
                        const Array4<const Real>& SmnSmn_a,
                        const Array4<const Real>& mf_m,
                        const Array4<const Real>& mf_u,
                        const Array4<const Real>& mf_v,
                              Array4<      Real>& hfx_x,
                              Array4<      Real>& hfx_y,
                              Array4<      Real>& hfx_z,
                              Array4<      Real>& diss,
                        const Array4<const Real>& mu_turb,
                        const SolverChoice &solverChoice,
                        const Array4<const Real>& tm_arr,
                        const amrex::GpuArray<Real,AMREX_SPACEDIM> grav_gpu,
                        const amrex::BCRec* bc_ptr)
{
    BL_PROFILE_VAR("DiffusionSrcForState_N()",DiffusionSrcForState_N);

    const Real dx_inv = cellSizeInv[0];
    const Real dy_inv = cellSizeInv[1];
    const Real dz_inv = cellSizeInv[2];

    const auto& dom_hi = amrex::ubound(domain);

    bool l_use_QKE       = solverChoice.use_QKE && solverChoice.advect_QKE;
    bool l_use_deardorff = (solverChoice.les_type == LESType::Deardorff);
    Real l_Delta         = std::pow(dx_inv * dy_inv * dz_inv,-1./3.);
    Real l_C_k           = solverChoice.Ck;
    Real l_C_e           = solverChoice.Ce;
    Real l_C_e_wall      = solverChoice.Ce_wall;
    Real Ce_lcoeff       = amrex::max(0.0, l_C_e - 1.9*l_C_k);
    Real l_inv_theta0    = 1.0 / solverChoice.theta_ref;
    Real l_abs_g         = std::abs(grav_gpu[2]);

    bool l_consA  = (solverChoice.molec_diff_type == MolecDiffType::ConstantAlpha);
    bool l_turb   = ( (solverChoice.les_type == LESType::Smagorinsky) ||
                      (solverChoice.les_type == LESType::Deardorff  ) ||
                      (solverChoice.pbl_type == PBLType::MYNN25     ) );

    const Box xbx = surroundingNodes(bx,0);
    const Box ybx = surroundingNodes(bx,1);
    const Box zbx = surroundingNodes(bx,2);

    const int ncomp      = n_end - n_start + 1;
    const int qty_offset = RhoTheta_comp;

    // Theta, KE, QKE, Scalar
    Vector<Real> alpha_eff(NUM_PRIM, 0.0);
    if (l_consA) {
        for (int i = 0; i < NUM_PRIM; ++i) {
           switch (i) {
               case PrimTheta_comp:
                    alpha_eff[PrimTheta_comp] = solverChoice.alpha_T;
                    break;
               case PrimScalar_comp:
                    alpha_eff[PrimScalar_comp] = solverChoice.alpha_C;
                    break;
#if defined(ERF_USE_MOISTURE)
               case PrimQt_comp:
                    alpha_eff[PrimQt_comp] = solverChoice.alpha_C;
                    break;
               case PrimQp_comp:
                    alpha_eff[PrimQp_comp] = solverChoice.alpha_C;
                    break;
#elif defined(ERF_USE_WARM_NO_PRECIP)
               case PrimQv_comp:
                    alpha_eff[PrimQv_comp] = solverChoice.alpha_C;
                    break;
               case PrimQc_comp:
                    alpha_eff[PrimQc_comp] = solverChoice.alpha_C;
                    break;
#endif
               default:
                    alpha_eff[i] = 0.0;
                    break;
          }
       }
    } else {
        for (int i = 0; i < NUM_PRIM; ++i) {
           switch (i) {
               case PrimTheta_comp:
                    alpha_eff[PrimTheta_comp] = solverChoice.rhoAlpha_T;
                    break;
               case PrimScalar_comp:
                    alpha_eff[PrimScalar_comp] = solverChoice.rhoAlpha_C;
                    break;
#if defined(ERF_USE_MOISTURE)
               case PrimQt_comp:
                    alpha_eff[PrimQt_comp] = solverChoice.rhoAlpha_C;
                    break;
               case PrimQp_comp:
                    alpha_eff[PrimQp_comp] = solverChoice.rhoAlpha_C;
                    break;
#elif defined(ERF_USE_WARM_NO_PRECIP)
               case PrimQv_comp:
                    alpha_eff[PrimQv_comp] = solverChoice.rhoAlpha_C;
                    break;
               case PrimQc_comp:
                    alpha_eff[PrimQc_comp] = solverChoice.rhoAlpha_C;
                    break;
#endif
               default:
                    alpha_eff[i] = 0.0;
                    break;
          }
       }
    }
#if defined(ERF_USE_MOISTURE)
    Vector<int> eddy_diff_idx{EddyDiff::Theta_h, EddyDiff::KE_h, EddyDiff::QKE_h, EddyDiff::Scalar_h, EddyDiff::Qt_h, EddyDiff::Qp_h};
    Vector<int> eddy_diff_idy{EddyDiff::Theta_h, EddyDiff::KE_h, EddyDiff::QKE_h, EddyDiff::Scalar_h, EddyDiff::Qt_h, EddyDiff::Qp_h};
    Vector<int> eddy_diff_idz{EddyDiff::Theta_v, EddyDiff::KE_v, EddyDiff::QKE_v, EddyDiff::Scalar_v, EddyDiff::Qt_v, EddyDiff::Qp_v};
#elif defined(ERF_USE_WARM_NO_PRECIP)
    Vector<int> eddy_diff_idx{EddyDiff::Theta_h, EddyDiff::KE_h, EddyDiff::QKE_h, EddyDiff::Scalar_h, EddyDiff::Qv_h, EddyDiff::Qc_h};
    Vector<int> eddy_diff_idy{EddyDiff::Theta_h, EddyDiff::KE_h, EddyDiff::QKE_h, EddyDiff::Scalar_h, EddyDiff::Qv_h, EddyDiff::Qc_h};
    Vector<int> eddy_diff_idz{EddyDiff::Theta_v, EddyDiff::KE_v, EddyDiff::QKE_v, EddyDiff::Scalar_v, EddyDiff::Qv_v, EddyDiff::Qc_v};
#else
    Vector<int> eddy_diff_idx{EddyDiff::Theta_h, EddyDiff::KE_h, EddyDiff::QKE_h, EddyDiff::Scalar_h};
    Vector<int> eddy_diff_idy{EddyDiff::Theta_h, EddyDiff::KE_h, EddyDiff::QKE_h, EddyDiff::Scalar_h};
    Vector<int> eddy_diff_idz{EddyDiff::Theta_v, EddyDiff::KE_v, EddyDiff::QKE_v, EddyDiff::Scalar_v};
#endif

    // Device vectors
    Gpu::AsyncVector<Real> alpha_eff_d;
    Gpu::AsyncVector<int>  eddy_diff_idx_d,eddy_diff_idy_d,eddy_diff_idz_d;
    alpha_eff_d.resize(alpha_eff.size());
    eddy_diff_idx_d.resize(eddy_diff_idx.size());
    eddy_diff_idy_d.resize(eddy_diff_idy.size());
    eddy_diff_idz_d.resize(eddy_diff_idz.size());

    Gpu::copy(Gpu::hostToDevice, alpha_eff.begin(), alpha_eff.end(), alpha_eff_d.begin());
    Gpu::copy(Gpu::hostToDevice, eddy_diff_idx.begin(), eddy_diff_idx.end(), eddy_diff_idx_d.begin());
    Gpu::copy(Gpu::hostToDevice, eddy_diff_idy.begin(), eddy_diff_idy.end(), eddy_diff_idy_d.begin());
    Gpu::copy(Gpu::hostToDevice, eddy_diff_idz.begin(), eddy_diff_idz.end(), eddy_diff_idz_d.begin());

    // Capture pointers for device code
    Real* d_alpha_eff     = alpha_eff_d.data();
    int*  d_eddy_diff_idx = eddy_diff_idx_d.data();
    int*  d_eddy_diff_idy = eddy_diff_idy_d.data();
    int*  d_eddy_diff_idz = eddy_diff_idz_d.data();

    // Compute fluxes at each face
    if (l_consA && l_turb) {
        amrex::ParallelFor(xbx, ncomp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            const int  qty_index = n_start + n;
            const int prim_index = qty_index - qty_offset;

            Real rhoFace  = 0.5 * ( cell_data(i, j, k, Rho_comp) + cell_data(i-1, j, k, Rho_comp) );
            Real rhoAlpha = rhoFace * d_alpha_eff[prim_index];
            rhoAlpha += 0.5 * ( mu_turb(i  , j, k, d_eddy_diff_idx[prim_index])
                              + mu_turb(i-1, j, k, d_eddy_diff_idx[prim_index]) );

            xflux(i,j,k,qty_index) = rhoAlpha * (cell_prim(i, j, k, prim_index) - cell_prim(i-1, j, k, prim_index)) * dx_inv * mf_u(i,j,0);
        });
        amrex::ParallelFor(ybx, ncomp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            const int  qty_index = n_start + n;
            const int prim_index = qty_index - qty_offset;

            Real rhoFace  = 0.5 * ( cell_data(i, j, k, Rho_comp) + cell_data(i, j-1, k, Rho_comp) );
            Real rhoAlpha = rhoFace * d_alpha_eff[prim_index];
            rhoAlpha += 0.5 * ( mu_turb(i, j  , k, d_eddy_diff_idy[prim_index])
                              + mu_turb(i, j-1, k, d_eddy_diff_idy[prim_index]) );

            yflux(i,j,k,qty_index) = rhoAlpha * (cell_prim(i, j, k, prim_index) - cell_prim(i, j-1, k, prim_index)) * dy_inv * mf_v(i,j,0);
        });
        amrex::ParallelFor(zbx, ncomp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            const int  qty_index = n_start + n;
            const int prim_index = qty_index - qty_offset;

            Real rhoFace  = 0.5 * ( cell_data(i, j, k, Rho_comp) + cell_data(i, j, k-1, Rho_comp) );
            Real rhoAlpha = rhoFace * d_alpha_eff[prim_index];
            rhoAlpha += 0.5 * ( mu_turb(i, j, k  , d_eddy_diff_idz[prim_index])
                              + mu_turb(i, j, k-1, d_eddy_diff_idz[prim_index]) );

            bool ext_dir_on_zlo = ( (bc_ptr[BCVars::cons_bc+qty_index].lo(2) == ERFBCType::ext_dir) && k == 0);
            bool ext_dir_on_zhi = ( (bc_ptr[BCVars::cons_bc+qty_index].lo(5) == ERFBCType::ext_dir) && k == dom_hi.z);
            if (ext_dir_on_zlo) {
                zflux(i,j,k,qty_index) = rhoAlpha * ( -(8./3.) * cell_prim(i, j, k-1, prim_index)
                                                          + 3. * cell_prim(i, j, k  , prim_index)
                                                     - (1./3.) * cell_prim(i, j, k+1, prim_index) ) * dz_inv;
            } else if (ext_dir_on_zhi) {
                zflux(i,j,k,qty_index) = rhoAlpha * (  (8./3.) * cell_prim(i, j, k-1, prim_index)
                                                          - 3. * cell_prim(i, j, k  , prim_index)
                                                     + (1./3.) * cell_prim(i, j, k+1, prim_index) ) * dz_inv;
            } else {
                zflux(i,j,k,qty_index) = rhoAlpha * (cell_prim(i, j, k, prim_index) - cell_prim(i, j, k-1, prim_index)) * dz_inv;
            }
        });
    } else if (l_turb) {
        amrex::ParallelFor(xbx, ncomp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            const int  qty_index = n_start + n;
            const int prim_index = qty_index - qty_offset;

            Real Alpha = d_alpha_eff[prim_index];
            Alpha += 0.5 * ( mu_turb(i  , j, k, d_eddy_diff_idx[prim_index])
                           + mu_turb(i-1, j, k, d_eddy_diff_idx[prim_index]) );

            xflux(i,j,k,qty_index) = Alpha * (cell_prim(i, j, k, prim_index) - cell_prim(i-1, j, k, prim_index)) * dx_inv * mf_u(i,j,0);
        });
        amrex::ParallelFor(ybx, ncomp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            const int  qty_index = n_start + n;
            const int prim_index = qty_index - qty_offset;

            Real Alpha = d_alpha_eff[prim_index];
            Alpha += 0.5 * ( mu_turb(i, j  , k, d_eddy_diff_idy[prim_index])
                           + mu_turb(i, j-1, k, d_eddy_diff_idy[prim_index]) );

            yflux(i,j,k,qty_index) = Alpha * (cell_prim(i, j, k, prim_index) - cell_prim(i, j-1, k, prim_index)) * dy_inv * mf_v(i,j,0);
        });
        amrex::ParallelFor(zbx, ncomp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            const int  qty_index = n_start + n;
            const int prim_index = qty_index - qty_offset;

            Real Alpha = d_alpha_eff[prim_index];
            Alpha += 0.5 * ( mu_turb(i, j, k  , d_eddy_diff_idz[prim_index])
                           + mu_turb(i, j, k-1, d_eddy_diff_idz[prim_index]) );

            bool ext_dir_on_zlo = ( (bc_ptr[BCVars::cons_bc+qty_index].lo(2) == ERFBCType::ext_dir) && k == 0);
            bool ext_dir_on_zhi = ( (bc_ptr[BCVars::cons_bc+qty_index].lo(5) == ERFBCType::ext_dir) && k == dom_hi.z);
            if (ext_dir_on_zlo) {
                zflux(i,j,k,qty_index) = Alpha * ( -(8./3.) * cell_prim(i, j, k-1, prim_index)
                                                       + 3. * cell_prim(i, j, k  , prim_index)
                                                  - (1./3.) * cell_prim(i, j, k+1, prim_index) ) * dz_inv;
            } else if (ext_dir_on_zhi) {
                zflux(i,j,k,qty_index) = Alpha * (  (8./3.) * cell_prim(i, j, k-1, prim_index)
                                                       - 3. * cell_prim(i, j, k  , prim_index)
                                                  + (1./3.) * cell_prim(i, j, k+1, prim_index) ) * dz_inv;
            } else {
                zflux(i,j,k,qty_index) = Alpha * (cell_prim(i, j, k, prim_index) - cell_prim(i, j, k-1, prim_index)) * dz_inv;
            }
        });
    } else if(l_consA) {
        amrex::ParallelFor(xbx, ncomp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            const int  qty_index = n_start + n;
            const int prim_index = qty_index - qty_offset;

            Real rhoFace  = 0.5 * ( cell_data(i, j, k, Rho_comp) + cell_data(i-1, j, k, Rho_comp) );
            Real rhoAlpha = rhoFace * d_alpha_eff[prim_index];

            xflux(i,j,k,qty_index) = rhoAlpha * (cell_prim(i, j, k, prim_index) - cell_prim(i-1, j, k, prim_index)) * dx_inv * mf_u(i,j,0);
        });
        amrex::ParallelFor(ybx, ncomp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            const int  qty_index = n_start + n;
            const int prim_index = qty_index - qty_offset;

            Real rhoFace  = 0.5 * ( cell_data(i, j, k, Rho_comp) + cell_data(i, j-1, k, Rho_comp) );
            Real rhoAlpha = rhoFace * d_alpha_eff[prim_index];

            yflux(i,j,k,qty_index) = rhoAlpha * (cell_prim(i, j, k, prim_index) - cell_prim(i, j-1, k, prim_index)) * dy_inv * mf_v(i,j,0);
        });
        amrex::ParallelFor(zbx, ncomp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            const int  qty_index = n_start + n;
            const int prim_index = qty_index - qty_offset;

            Real rhoFace  = 0.5 * ( cell_data(i, j, k, Rho_comp) + cell_data(i, j, k-1, Rho_comp) );
            Real rhoAlpha = rhoFace * d_alpha_eff[prim_index];

            bool ext_dir_on_zlo = ( (bc_ptr[BCVars::cons_bc+qty_index].lo(2) == ERFBCType::ext_dir) && k == 0);
            bool ext_dir_on_zhi = ( (bc_ptr[BCVars::cons_bc+qty_index].lo(5) == ERFBCType::ext_dir) && k == dom_hi.z);
            if (ext_dir_on_zlo) {
                zflux(i,j,k,qty_index) = rhoAlpha * ( -(8./3.) * cell_prim(i, j, k-1, prim_index)
                                                          + 3. * cell_prim(i, j, k  , prim_index)
                                                     - (1./3.) * cell_prim(i, j, k+1, prim_index) ) * dz_inv;
            } else if (ext_dir_on_zhi) {
                zflux(i,j,k,qty_index) = rhoAlpha * (  (8./3.) * cell_prim(i, j, k-1, prim_index)
                                                          - 3. * cell_prim(i, j, k  , prim_index)
                                                     + (1./3.) * cell_prim(i, j, k+1, prim_index) ) * dz_inv;
            } else {
                zflux(i,j,k,qty_index) = rhoAlpha * (cell_prim(i, j, k, prim_index) - cell_prim(i, j, k-1, prim_index)) * dz_inv;
            }
        });
    } else {
        amrex::ParallelFor(xbx, ncomp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            const int  qty_index = n_start + n;
            const int prim_index = qty_index - qty_offset;

            Real Alpha = d_alpha_eff[prim_index];

            xflux(i,j,k,qty_index) = Alpha * (cell_prim(i, j, k, prim_index) - cell_prim(i-1, j, k, prim_index)) * dx_inv * mf_u(i,j,0);
        });
        amrex::ParallelFor(ybx, ncomp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            const int  qty_index = n_start + n;
            const int prim_index = qty_index - qty_offset;

            Real Alpha = d_alpha_eff[prim_index];

            yflux(i,j,k,qty_index) = Alpha * (cell_prim(i, j, k, prim_index) - cell_prim(i, j-1, k, prim_index)) * dy_inv * mf_v(i,j,0);
        });
        amrex::ParallelFor(zbx, ncomp,[=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            const int  qty_index = n_start + n;
            const int prim_index = qty_index - qty_offset;

            Real Alpha = d_alpha_eff[prim_index];

            bool ext_dir_on_zlo = ( (bc_ptr[BCVars::cons_bc+qty_index].lo(2) == ERFBCType::ext_dir) && k == 0);
            bool ext_dir_on_zhi = ( (bc_ptr[BCVars::cons_bc+qty_index].lo(5) == ERFBCType::ext_dir) && k == dom_hi.z);
            if (ext_dir_on_zlo) {
                zflux(i,j,k,qty_index) = Alpha * ( -(8./3.) * cell_prim(i, j, k-1, prim_index)
                                                       + 3. * cell_prim(i, j, k  , prim_index)
                                                  - (1./3.) * cell_prim(i, j, k+1, prim_index) ) * dz_inv;
            } else if (ext_dir_on_zhi) {
                zflux(i,j,k,qty_index) = Alpha * (  (8./3.) * cell_prim(i, j, k-1, prim_index)
                                                       - 3. * cell_prim(i, j, k  , prim_index)
                                                  + (1./3.) * cell_prim(i, j, k+1, prim_index) ) * dz_inv;
            } else {
                zflux(i,j,k,qty_index) = Alpha * (cell_prim(i, j, k, prim_index) - cell_prim(i, j, k-1, prim_index)) * dz_inv;
            }
        });
    }



    // Use fluxes to compute RHS
    for (int qty_index = n_start; qty_index <= n_end; qty_index++)
    {
        amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {

            cell_rhs(i,j,k,qty_index) += (xflux(i+1,j  ,k  ,qty_index) - xflux(i, j, k, qty_index)) * dx_inv * mf_m(i,j,0)  // Diffusive flux in x-dir
                                        +(yflux(i  ,j+1,k  ,qty_index) - yflux(i, j, k, qty_index)) * dy_inv * mf_m(i,j,0)  // Diffusive flux in y-dir
                                        +(zflux(i  ,j  ,k+1,qty_index) - zflux(i, j, k, qty_index)) * dz_inv;  // Diffusive flux in z-dir
        });
    }

    // Using Deardorff
    if (l_use_deardorff && n_end >= RhoKE_comp) {
        int qty_index = RhoKE_comp;
        amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real cellVolMsf = 1.0 / (dx_inv * mf_u(i,j,0) * dy_inv * mf_v(i,j,0) * dz_inv);
            Real DeltaMsf   = std::pow(cellVolMsf,1.0/3.0);

            // calculate stratification-dependent mixing length (Deardorff 1980)
            Real eps       = std::numeric_limits<Real>::epsilon();
            Real dtheta_dz = 0.5*(  cell_data(i,j,k+1,RhoTheta_comp)/cell_data(i,j,k+1,Rho_comp)
                                  - cell_data(i,j,k-1,RhoTheta_comp)/cell_data(i,j,k-1,Rho_comp))*dz_inv;
            Real E         = amrex::max(cell_data(i,j,k,RhoKE_comp) / cell_data(i,j,k,Rho_comp), eps);
            Real strat     = l_abs_g * dtheta_dz * l_inv_theta0; // ==N^2 under stable conditions
            Real length;
            if (strat <= eps) {
                length = DeltaMsf;
            } else {
                length = amrex::min(DeltaMsf,
                                    0.76 * std::sqrt(E / strat));
            }

            // From eddy viscosity (or eddy diffusivity for momentum)
            //   mu_turb = rho * C_k * l * KE^(1/2)
            // and the eddy diffusivity for heat [kg m^-1 s^-1] is
            //   KH = (1 + 2*l/delta) * mu_turb
            // Note: mu_turb is fixed for all 3 RK stages, so recomputing the
            // eddy viscosity part of KH here would be inconsistent (since
            // l and KE evolve with each stage)
            Real KH = (1.+2.*length/DeltaMsf) * mu_turb(i,j,k,EddyDiff::Mom_v);

            // Add Buoyancy Source
            // where the SGS buoyancy flux tau_{theta,i} = -KH * dtheta/dx_i,
            // such that for dtheta/dz < 0, there is a positive (upward) heat flux;
            // flux; the TKE buoyancy production is then
            //   B = g/theta_0 * tau_{theta,w}
            // for a dry atmosphere (see, e.g., Sullivan et al 1994). To
            // account for moisture, the Brunt-Vaisala frequency,
            //   N^2 = g[1/theta * dtheta/dz + ...]
            // should be a function of the water vapor and total water mixing
            // ratios, depending on whether conditions are saturated or
            // not (see the WRF model description, Skamarock et al 2019).
            hfx_x(i,j,k) = 0.0;
            hfx_y(i,j,k) = 0.0;
            hfx_z(i,j,k) = -KH * dtheta_dz; // (rho*w)'theta' [kg m^-2 s^-1 K]
            cell_rhs(i,j,k,qty_index) += l_abs_g * l_inv_theta0 * hfx_z(i,j,k);

            // TKE shear production
            //   P = -tau_ij * S_ij = 2 * mu_turb * S_ij * S_ij
            // Note: This assumes that the horizontal and vertical diffusivities
            // of momentum are equal
            cell_rhs(i,j,k,qty_index) += 2.0*mu_turb(i,j,k,EddyDiff::Mom_h) * SmnSmn_a(i,j,k);

            // TKE dissipation
            diss(i,j,k) = 0.0;
            amrex::Real Ce;
            if ((l_C_e_wall > 0) && (k==0))
                Ce = l_C_e_wall;
            else
                Ce = 1.9*l_C_k + Ce_lcoeff*length / DeltaMsf;
            if (std::abs(E) > 0.) {
                diss(i,j,k) = cell_data(i,j,k,Rho_comp) * Ce *
                    std::pow(E,1.5) / length;
                cell_rhs(i,j,k,qty_index) -= diss(i,j,k);
            }
        });
    }

    // Using Deardorff
    if (l_use_QKE && n_end >= RhoQKE_comp) {
        int qty_index = RhoQKE_comp;
        amrex::ParallelFor(bx,[=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            cell_rhs(i, j, k, qty_index) += ComputeQKESourceTerms(i,j,k,u,v,cell_data,cell_prim,
                                                                  mu_turb,cellSizeInv,domain,solverChoice,tm_arr(i,j,0));
        });
    }

}
