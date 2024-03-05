#include <ABLMost.H>
#include <MOSTAverage.H>

using namespace amrex;

/**
 * Wrapper to update ustar and tstar for Monin Obukhov similarity theory.
 *
 * @param[in] lev Current level
 * @param[in] max_iters maximum iterations to use
 */
void
ABLMost::update_fluxes (const int& lev,
                        const Real& time,
                        int max_iters)
{
    // Update SST data if we have a valid pointer
    if (m_sst_lev[lev][0]) time_interp_sst(lev, time);

    // TODO: we want 0 index to always be theta?
    // Update land surface temp if we have a valid pointer
    if (m_lsm_data_lev[lev][0]) get_lsm_tsurf(lev);

    // Fill interior ghost cells
    t_surf[lev]->FillBoundary(m_geom[lev].periodicity());

    // Compute plane averages for all vars (regardless of flux type)
    m_ma.compute_averages(lev);

    // Iterate the fluxes if moeng type
    if (flux_type == FluxCalcType::MOENG) {
        if (theta_type == ThetaCalcType::HEAT_FLUX) {
            if (rough_type == RoughCalcType::CONSTANT) {
                surface_flux most_flux(m_ma.get_zref(), surf_temp_flux);
                compute_fluxes(lev, max_iters, most_flux);
            } else if (rough_type == RoughCalcType::CHARNOCK) {
                surface_flux_charnock most_flux(m_ma.get_zref(), surf_temp_flux, cnk_a);
                compute_fluxes(lev, max_iters, most_flux);
            } else {
                surface_flux_mod_charnock most_flux(m_ma.get_zref(), surf_temp_flux, depth);
                compute_fluxes(lev, max_iters, most_flux);
            }
        } else if (theta_type == ThetaCalcType::SURFACE_TEMPERATURE) {
            update_surf_temp(time);
            if (rough_type == RoughCalcType::CONSTANT) {
                surface_temp most_flux(m_ma.get_zref(), surf_temp_flux);
                compute_fluxes(lev, max_iters, most_flux);
            } else if (rough_type == RoughCalcType::CHARNOCK) {
                surface_temp_charnock most_flux(m_ma.get_zref(), surf_temp_flux, cnk_a);
                compute_fluxes(lev, max_iters, most_flux);
            } else {
                surface_temp_mod_charnock most_flux(m_ma.get_zref(), surf_temp_flux, depth);
                compute_fluxes(lev, max_iters, most_flux);
            }
        } else {
            if (rough_type == RoughCalcType::CONSTANT) {
                adiabatic most_flux(m_ma.get_zref(), surf_temp_flux);
                compute_fluxes(lev, max_iters, most_flux);
            } else if (rough_type == RoughCalcType::CHARNOCK) {
                adiabatic_charnock most_flux(m_ma.get_zref(), surf_temp_flux, cnk_a);
                compute_fluxes(lev, max_iters, most_flux);
            } else {
                adiabatic_mod_charnock most_flux(m_ma.get_zref(), surf_temp_flux, depth);
                compute_fluxes(lev, max_iters, most_flux);
            }
        } // theta flux
    } else if (flux_type == FluxCalcType::CUSTOM) {
        u_star[lev]->setVal(custom_ustar);
        t_star[lev]->setVal(custom_tstar);
        q_star[lev]->setVal(custom_qstar);
    }
}


/**
 * Function to compute the fluxes (u^star and t^star) for Monin Obukhov similarity theory.
 *
 * @param[in] lev Current level
 * @param[in] max_iters maximum iterations to use
 * @param[in] most_flux structure to iteratively compute ustar and tstar
 */
template <typename FluxIter>
void
ABLMost::compute_fluxes (const int& lev,
                         const int& max_iters,
                         const FluxIter& most_flux)
{
    // Pointers to the computed averages
    const auto *const tm_ptr  = m_ma.get_average(lev,2);
    const auto *const umm_ptr = m_ma.get_average(lev,4);

    for (MFIter mfi(*u_star[lev]); mfi.isValid(); ++mfi)
    {
        Box gtbx = mfi.growntilebox();

        auto u_star_arr = u_star[lev]->array(mfi);
        auto t_star_arr = t_star[lev]->array(mfi);
        auto t_surf_arr = t_surf[lev]->array(mfi);
        auto olen_arr   = olen[lev]->array(mfi);

        const auto tm_arr  = tm_ptr->array(mfi);
        const auto umm_arr = umm_ptr->array(mfi);
        const auto z0_arr  = z_0[lev].array();

        ParallelFor(gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            most_flux.iterate_flux(i, j, k, max_iters, z0_arr, umm_arr, tm_arr,
                                   u_star_arr, t_star_arr, t_surf_arr, olen_arr);
        });
    }
}


/**
 * Wrapper to impose Monin Obukhov similarity theory fluxes by populating ghost cells.
 *
 * @param[in] lev Current level
 * @param[in,out] mfs Multifabs to populate
 * @param[in] eddyDiffs Diffusion coefficients from turbulence model
 */
void
ABLMost::impose_most_bcs (const int& lev,
                          const Vector<MultiFab*>& mfs,
                          MultiFab* xzmom_flux, MultiFab* zxmom_flux,
                          MultiFab* yzmom_flux, MultiFab* zymom_flux,
                          MultiFab* heat_flux,  MultiFab* qv_flux,
                          MultiFab* eddyDiffs,
                          MultiFab* z_phys)
{
    Box domain    = m_geom[lev].Domain();
    const int klo = domain.smallEnd(2);
    if (flux_type == FluxCalcType::MOENG) {
        moeng_flux flux_comp(klo);
        compute_most_bcs(lev, mfs,
                         xzmom_flux, zxmom_flux,
                         yzmom_flux, zymom_flux,
                         heat_flux, qv_flux,
                         eddyDiffs,
                         z_phys, m_geom[lev].CellSize(2), flux_comp);
    } else if (flux_type == FluxCalcType::DONELAN) {
        donelan_flux flux_comp(klo);
        compute_most_bcs(lev, mfs,
                         xzmom_flux, zxmom_flux,
                         yzmom_flux, zymom_flux,
                         heat_flux, qv_flux,
                         eddyDiffs,
                         z_phys, m_geom[lev].CellSize(2), flux_comp);
    } else {
        custom_flux flux_comp(klo);
        compute_most_bcs(lev, mfs,
                         xzmom_flux, zxmom_flux,
                         yzmom_flux, zymom_flux,
                         heat_flux, qv_flux,
                         eddyDiffs,
                         z_phys, m_geom[lev].CellSize(2), flux_comp);
    }
}


/**
 * Function to calculate MOST fluxes for populating ghost cells.
 *
 * @param[in] lev Current level
 * @param[in,out] mfs Multifabs to populate
 * @param[in] eddyDiffs Diffusion coefficients from turbulence model
 * @param[in] flux_comp structure to compute fluxes
 */
template<typename FluxCalc>
void
ABLMost::compute_most_bcs (const int& lev,
                           const Vector<MultiFab*>& mfs,
                           MultiFab* xzmom_flux, MultiFab* zxmom_flux,
                           MultiFab* yzmom_flux, MultiFab* zymom_flux,
                           MultiFab* heat_flux , MultiFab* qv_flux,
                           MultiFab* eddyDiffs,
                           MultiFab* z_phys,
                           const Real& dz_no_terrain,
                           const FluxCalc& flux_comp)
{
    Box domain    = m_geom[lev].Domain();
    const int klo = domain.smallEnd(2);
    const bool store_fluxes = m_store_fluxes;
    const Real eta_eps = 1e-8;

    for (MFIter mfi(*mfs[0]); mfi.isValid(); ++mfi)
    {
        // TODO: No LSM lateral ghost cells, should this change?
        // Valid CC box
        Box vbx = mfi.validbox(); vbx.makeSlab(2,klo);
        Box vbxx = surroundingNodes(vbx,0);
        Box vbxy = surroundingNodes(vbx,1);

        // Get field arrays
        const auto cons_arr  = mfs[Vars::cons]->array(mfi);
        const auto velx_arr  = mfs[Vars::xvel]->array(mfi);
        const auto vely_arr  = mfs[Vars::yvel]->array(mfi);

        // Store diffusive fluxes
        auto t13_arr = xzmom_flux->array(mfi);
        auto t23_arr = yzmom_flux->array(mfi);
        auto t31_arr = (zxmom_flux) ? zxmom_flux->array(mfi) : Array4<Real>{};
        auto t32_arr = (zymom_flux) ? zymom_flux->array(mfi) : Array4<Real>{};
        auto hfx_arr = heat_flux->array(mfi);
        auto qfx_arr = (qv_flux) ? qv_flux->array(mfi) : Array4<Real>{};

        // Fill ghost cells
        const auto  eta_arr  = eddyDiffs->array(mfi);
        const auto zphys_arr = (z_phys) ? z_phys->const_array(mfi) : Array4<const Real>{};

        // Get average arrays
        const auto *const u_mean     = m_ma.get_average(lev,0);
        const auto *const v_mean     = m_ma.get_average(lev,1);
        const auto *const t_mean     = m_ma.get_average(lev,2);
        const auto *const q_mean     = m_ma.get_average(lev,3);
        const auto *const u_mag_mean = m_ma.get_average(lev,4);

        const auto um_arr  = u_mean->array(mfi);
        const auto vm_arr  = v_mean->array(mfi);
        const auto tm_arr  = t_mean->array(mfi);
        const auto umm_arr = u_mag_mean->array(mfi);

        // Get derived arrays
        const auto u_star_arr = u_star[lev]->array(mfi);
        const auto t_star_arr = t_star[lev]->array(mfi);
        const auto q_star_arr = q_star[lev]->array(mfi);
        const auto t_surf_arr = t_surf[lev]->array(mfi);

        // Get LSM fluxes
        auto lmask_arr    = (m_lmask_lev[lev][0])    ? m_lmask_lev[lev][0]->array(mfi) :
                                                       Array4<int> {};
        auto lsm_flux_arr = (m_lsm_flux_lev[lev][0]) ? m_lsm_flux_lev[lev][0]->array(mfi) :
                                                       Array4<Real> {};

        for (int var_idx = 0; var_idx < Vars::NumTypes; ++var_idx)
        {
            const Box& bx = (*mfs[var_idx])[mfi].box();
            auto dest_arr = (*mfs[var_idx])[mfi].array();

            if (var_idx == Vars::cons) {
                Box b2d = bx; int k_end = b2d.smallEnd(2); b2d.makeSlab(2,klo);
                int n   = RhoTheta_comp;
                ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    Real dz  = (zphys_arr) ? ( zphys_arr(i,j,klo  ) - zphys_arr(i,j,klo-1) ) : dz_no_terrain;
                    Real dz1 = (zphys_arr) ? ( zphys_arr(i,j,klo+1) - zphys_arr(i,j,klo  ) ) : dz_no_terrain;

                    // This is the _kinematic_ heat flux [K m/s]
                    Real Tflux = flux_comp.compute_t_flux(i, j, k, n,
                                                          cons_arr, velx_arr, vely_arr,
                                                          umm_arr, tm_arr,
                                                          u_star_arr, t_star_arr,
                                                          t_surf_arr);

                    // Pass heat flux to LSM model if needed
                    int is_land = (lmask_arr) ? lmask_arr(i,j,klo) : 1;
                    if (is_land && lsm_flux_arr && vbx.contains(i,j,k)) {
                        lsm_flux_arr(i,j,klo) = Tflux;
                    }

                    // Store flux or populate ghost cells
                    if (store_fluxes) {
                        if (vbx.contains(i,j,k)) hfx_arr(i,j,klo) = Tflux;
                        Real RT     =  cons_arr(i,j,klo  ,n);
                        Real RTgrad = (cons_arr(i,j,klo+1,n) - RT) / (0.5*(dz+dz1)); // surface gradient equal to gradient at first zface
                        for (int lk(klo-1); lk>=k_end; --lk) {
                            Real deltaz  = dz * (klo - lk);
                            dest_arr(i,j,k,n) = RT - RTgrad * deltaz;
                        }
                    } else {
                        Real rho   = cons_arr(i,j,klo,Rho_comp);
                        Real theta = cons_arr(i,j,klo,n) / rho;

                        int ie, je;
                        ie = i  < lbound(eta_arr).x ? lbound(eta_arr).x : i;
                        je = j  < lbound(eta_arr).y ? lbound(eta_arr).y : j;
                        ie = ie > ubound(eta_arr).x ? ubound(eta_arr).x : ie;
                        je = je > ubound(eta_arr).y ? ubound(eta_arr).y : je;
                        Real eta = eta_arr(ie,je,klo,EddyDiff::Theta_v);
                        eta   = amrex::max(eta,eta_eps);

                        for (int lk(klo-1); lk>=k_end; --lk) {
                            Real deltaz  = dz * (klo - lk);
                            dest_arr(i,j,k,n) = rho*(theta + Tflux*rho/eta*deltaz);
                        }
                    }

                });


                if ((flux_type == FluxCalcType::CUSTOM) && use_moisture) {
                    n = RhoQ1_comp;
                    ParallelFor(b2d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                    {
                        Real dz  = (zphys_arr) ? ( zphys_arr(i,j,klo  ) - zphys_arr(i,j,klo-1) ) : dz_no_terrain;
                        Real dz1 = (zphys_arr) ? ( zphys_arr(i,j,klo+1) - zphys_arr(i,j,klo  ) ) : dz_no_terrain;

                        Real Qflux = flux_comp.compute_q_flux(i, j, k, n,
                                                              cons_arr, velx_arr, vely_arr,
                                                              umm_arr, tm_arr,
                                                              u_star_arr, q_star_arr,
                                                              t_surf_arr);

                        // TODO: Generalize MOST q flux with MOENG & DONELAN flux types

                        // Store flux or populate ghost cells
                        if (store_fluxes) {
                            if (vbx.contains(i,j,k)) qfx_arr(i,j,klo) = Qflux;
                            Real RQ     =  cons_arr(i,j,klo  ,n);
                            Real RQgrad = (cons_arr(i,j,klo+1,n) - RQ) / (0.5*(dz+dz1)); // surface gradient equal to gradient at first zface
                            for (int lk(klo-1); lk>=k_end; --lk) {
                                Real deltaz  = dz * (klo - lk);
                                dest_arr(i,j,k,n) = RQ - RQgrad * deltaz;
                            }
                        } else {
                            Real rho = cons_arr(i,j,klo,Rho_comp);
                            Real qv  = cons_arr(i,j,klo,n) / rho;

                            int ie, je;
                            ie = std::max(lbound(eta_arr).x,i );
                            ie = std::min(ubound(eta_arr).x,ie);
                            je = std::max(lbound(eta_arr).y,j );
                            je = std::min(ubound(eta_arr).y,je);

                            Real eta = eta_arr(ie,je,klo,EddyDiff::Q_v);
                            eta = amrex::max(eta,eta_eps);

                            for (int lk(klo-1); lk>=k_end; --lk) {
                                Real deltaz  = dz * (klo - lk);
                                dest_arr(i,j,k,n) = rho*(qv + Qflux*rho/eta*deltaz);
                            }
                        }
                    });
                }

            } else if (var_idx == Vars::xvel) {
                Box xb2d = surroundingNodes(bx,0); int k_end = xb2d.smallEnd(2); xb2d.makeSlab(2,klo);
                ParallelFor(xb2d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    Real dz  = (zphys_arr) ? ( zphys_arr(i,j,klo  ) - zphys_arr(i,j,klo-1) ) : dz_no_terrain;
                    Real dz1 = (zphys_arr) ? ( zphys_arr(i,j,klo+1) - zphys_arr(i,j,klo  ) ) : dz_no_terrain;
                    Real stressx = flux_comp.compute_u_flux(i, j, k,
                                                            cons_arr, velx_arr, vely_arr,
                                                            umm_arr, um_arr, u_star_arr);

                    // Store flux or populate ghost cells
                    if (store_fluxes) {
                        if (vbxx.contains(i,j,k)) {
                            t13_arr(i,j,klo) = -stressx;
                            if (t31_arr) t31_arr(i,j,klo) = -stressx;
                        }
                        Real rho   = 0.5 * ( cons_arr(i  ,j,klo,Rho_comp)
                                           + cons_arr(i-1,j,klo,Rho_comp) );
                        Real ugrad = (velx_arr(i,j,klo+1) - velx_arr(i,j,klo)) / (0.5*(dz+dz1)); // surface gradient equal to gradient at first zface
                        for (int lk(klo-1); lk>=k_end; --lk) {
                            Real deltaz  = dz * (klo - lk);
                            dest_arr(i,j,k) = rho*(velx_arr(i,j,klo) - ugrad * deltaz);
                        }
                    } else {
                        int ie, je;
                        ie = std::max(lbound(eta_arr).x+1,i );
                        ie = std::min(ubound(eta_arr).x  ,ie);
                        je = std::max(lbound(eta_arr).y  ,j );
                        je = std::min(ubound(eta_arr).y  ,je);

                        Real eta   = 0.5 *(  eta_arr(ie  ,je,klo,EddyDiff::Mom_v)
                                           + eta_arr(ie-1,je,klo,EddyDiff::Mom_v) );
                        eta   = amrex::max(eta,eta_eps);

                        for (int lk(klo-1); lk>=k_end; --lk) {
                            Real deltaz  = dz * (klo - lk);
                            dest_arr(i,j,k) = dest_arr(i,j,klo) - stressx/eta*deltaz;
                        }
                    }
                });

            } else if (var_idx == Vars::yvel) {
                Box yb2d = surroundingNodes(bx,1); int k_end = yb2d.smallEnd(2); yb2d.makeSlab(2,klo);
                ParallelFor(yb2d, [=] AMREX_GPU_DEVICE (int i, int j, int k)
                {
                    Real dz  = (zphys_arr) ? ( zphys_arr(i,j,klo  ) - zphys_arr(i,j,klo-1) ) : dz_no_terrain;
                    Real dz1 = (zphys_arr) ? ( zphys_arr(i,j,klo+1) - zphys_arr(i,j,klo  ) ) : dz_no_terrain;
                    Real stressy = flux_comp.compute_v_flux(i, j, k,
                                                            cons_arr, velx_arr, vely_arr,
                                                            umm_arr, vm_arr, u_star_arr);

                    // Store flux or populate ghost cells
                    if (store_fluxes) {
                        if (vbxy.contains(i,j,k)) {
                            t23_arr(i,j,klo) = -stressy;
                            if (t32_arr) t32_arr(i,j,klo) = -stressy;
                        }
                        Real rho   = 0.5 * ( cons_arr(i,j  ,klo,Rho_comp)
                                           + cons_arr(i,j-1,klo,Rho_comp) );
                        Real vgrad = (vely_arr(i,j,klo+1) - vely_arr(i,j,klo)) / (0.5*(dz+dz1)); // surface gradient equal to gradient at first zface
                        for (int lk(klo-1); lk>=k_end; --lk) {
                            Real deltaz  = dz * (klo - lk);
                            dest_arr(i,j,k) = rho*(vely_arr(i,j,klo) - vgrad * deltaz);
                        }
                    } else {
                        int ie, je;
                        ie = std::max(lbound(eta_arr).x  ,i );
                        ie = std::min(ubound(eta_arr).x  ,ie);
                        je = std::max(lbound(eta_arr).y+1,j );
                        je = std::min(ubound(eta_arr).y  ,je);

                        Real eta   = 0.5 *(  eta_arr(ie,je  ,klo,EddyDiff::Mom_v)
                                           + eta_arr(ie,je-1,klo,EddyDiff::Mom_v) );
                        eta   = amrex::max(eta,eta_eps);

                        for (int lk(klo-1); lk>=k_end; --lk) {
                            Real deltaz  = dz * (klo - lk);
                            dest_arr(i,j,k) = dest_arr(i,j,klo) - stressy/eta*deltaz;
                        }
                    }
                });
            }
        } // var_idx
    } // mfiter
}

void
ABLMost::time_interp_sst (const int& lev,
                          const Real& time)
{
    // Time interpolation
    Real dT = m_bdy_time_interval;
    Real time_since_start = time - m_start_bdy_time;
    int n_time = static_cast<int>( time_since_start /  dT);
    amrex::Real alpha = (time_since_start - n_time * dT) / dT;
    AMREX_ALWAYS_ASSERT( alpha >= 0. && alpha <= 1.0);
    amrex::Real oma   = 1.0 - alpha;
    AMREX_ALWAYS_ASSERT( (n_time >= 0) && (n_time < (m_sst_lev[lev].size()-1)));

    // Populate t_surf
    for (MFIter mfi(*t_surf[lev]); mfi.isValid(); ++mfi)
    {
        Box gtbx = mfi.growntilebox();

        auto t_surf_arr = t_surf[lev]->array(mfi);
        const auto sst_hi_arr = m_sst_lev[lev][n_time+1]->const_array(mfi);
        const auto sst_lo_arr = m_sst_lev[lev][n_time  ]->const_array(mfi);
        auto lmask_arr  = (m_lmask_lev[lev][0]) ? m_lmask_lev[lev][0]->array(mfi) :
                                                  Array4<int> {};

        ParallelFor(gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            int is_land = (lmask_arr) ? lmask_arr(i,j,k) : 1;
            if (!is_land) {
                t_surf_arr(i,j,k) = oma   * sst_lo_arr(i,j,k)
                                  + alpha * sst_hi_arr(i,j,k);
            }
        });
    }
}

void
ABLMost::get_lsm_tsurf (const int& lev)
{
    for (MFIter mfi(*t_surf[lev]); mfi.isValid(); ++mfi)
    {
        Box gtbx = mfi.growntilebox();

        // TODO: LSM does not carry lateral ghost cells.
        //       This copies the valid box into the ghost cells.
        //       Fillboundary is called after this to pick up the
        //       interior ghost and periodic directions. Is there
        //       a better approach?
        Box vbx  = mfi.validbox();
        int i_lo = vbx.smallEnd(0); int i_hi = vbx.bigEnd(0);
        int j_lo = vbx.smallEnd(1); int j_hi = vbx.bigEnd(1);

        auto t_surf_arr = t_surf[lev]->array(mfi);
        auto lmask_arr  = (m_lmask_lev[lev][0]) ? m_lmask_lev[lev][0]->array(mfi) :
                                                  Array4<int> {};
        const auto lsm_arr = m_lsm_data_lev[lev][0]->const_array(mfi);

        ParallelFor(gtbx, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept
        {
            int is_land = (lmask_arr) ? lmask_arr(i,j,k) : 1;
            if (is_land) {
                int li = amrex::min(amrex::max(i, i_lo), i_hi);
                int lj = amrex::min(amrex::max(j, j_lo), j_hi);
                t_surf_arr(i,j,k) = lsm_arr(li,lj,k);
            }
        });
    }
}
