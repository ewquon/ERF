#ifndef MOSTStress_H
#define MOSTStress_H

#include <ERF_Constants.H>
#include <IndexDefines.H>

/**
 * Structure of plain old data relevant to MOST BCs
 */
struct most_data
{
public:
    amrex::Real zref{10.0};          ///< Reference height (m)
    amrex::Real z0_const{0.1};       ///< Roughness height -- default constant value(m)
    amrex::Real kappa{KAPPA};        ///< von Karman constant
    amrex::Real gravity{CONST_GRAV}; ///< Acceleration due to gravity (m/s^2)
    amrex::Real surf_temp_flux{0.0}; ///< Heat flux
    amrex::Real Cnk_a{0.0185};       ///< Standard Charnock constant https://doi.org/10.1175/JAMC-D-17-0137.1
    amrex::Real Cnk_b1{1.0/30.0};    ///< Modified Charnock Eq (4) https://doi.org/10.1175/JAMC-D-17-0137.1
    amrex::Real Cnk_b2{1260.0};      ///< Modified Charnock Eq (4) https://doi.org/10.1175/JAMC-D-17-0137.1
    amrex::Real Cnk_d{30.0};         ///< Modified Charnock Eq (4) https://doi.org/10.1175/JAMC-D-17-0137.1
    amrex::Real Cnk_b;
};


/**
 * Structure of similarity functions for Moeng formulation
 */
struct similarity_funs
{
    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real
    calc_psi_m (amrex::Real zeta) const
    {
        if (zeta > 0) {
            return -beta_m * zeta;
        } else {
            amrex::Real x = std::sqrt(std::sqrt(1.0 - gamma_m * zeta));
            return 2.0 * std::log(0.5 * (1.0 + x)) + log(0.5 * (1.0 + x * x)) -
                   2.0 * std::atan(x) + PIoTwo;
        }
    }

    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real
    calc_psi_h (amrex::Real zeta) const
    {
        if (zeta > 0) {
            return -beta_h * zeta;
        } else {
            amrex::Real x = std::sqrt(1.0 - gamma_h * zeta);
            return 2.0 * std::log(0.5 * (1.0 + x));
        }
    }

private:
    amrex::Real beta_m{5.0};         ///< Constants from Dyer, BLM, 1974
    amrex::Real beta_h{5.0};         ///< https://doi.org/10.1007/BF00240838
    amrex::Real gamma_m{16.0};
    amrex::Real gamma_h{16.0};
};


/**
 * Adiabatic with constant roughness
 */
struct adiabatic
{
    adiabatic (amrex::Real zref,
               amrex::Real flux)
    {
        mdata.zref = zref;
        mdata.surf_temp_flux = flux;
    }

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    void
    iterate_flux (const int& i,
                  const int& j,
                  const int& k,
                  const int& /*max_iters*/,
                  const amrex::Array4<const amrex::Real>& z0_arr,
                  const amrex::Array4<const amrex::Real>& umm_arr,
                  const amrex::Array4<const amrex::Real>& /*tm_arr*/,
                  const amrex::Array4<amrex::Real>& u_star_arr,
                  const amrex::Array4<amrex::Real>& t_star_arr,
                  const amrex::Array4<amrex::Real>& /*t_surf_arr*/,
                  const amrex::Array4<amrex::Real>& olen_arr) const
    {
        u_star_arr(i,j,k) = mdata.kappa * umm_arr(i,j,k) / std::log(mdata.zref / z0_arr(i,j,k));
        t_star_arr(i,j,k) = 0.0;
        olen_arr(i,j,k)   = 1.0e16;
    }

private:
    most_data mdata;
    similarity_funs sfuns;
};


/**
 * Adiabatic with charnock roughness
 */
struct adiabatic_charnock
{
    adiabatic_charnock (amrex::Real zref,
                        amrex::Real flux,
                        amrex::Real cnk_a)
    {
        mdata.zref = zref;
        mdata.surf_temp_flux = flux;
        mdata.Cnk_a = cnk_a;
    }

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    void
    iterate_flux (const int& i,
                  const int& j,
                  const int& k,
                  const int& max_iters,
                  const amrex::Array4<amrex::Real>& z0_arr,
                  const amrex::Array4<const amrex::Real>& umm_arr,
                  const amrex::Array4<const amrex::Real>& /*tm_arr*/,
                  const amrex::Array4<amrex::Real>& u_star_arr,
                  const amrex::Array4<amrex::Real>& t_star_arr,
                  const amrex::Array4<amrex::Real>& /*t_surf_arr*/,
                  const amrex::Array4<amrex::Real>& olen_arr) const
    {
        int iter = 0;
        amrex::Real ustar = 0.0;
        amrex::Real z0    = 0.0;
        u_star_arr(i,j,k) = mdata.kappa * umm_arr(i,j,k) / std::log(mdata.zref / z0_arr(i,j,k));
        do {
            ustar = u_star_arr(i,j,k);
            z0    = (mdata.Cnk_a / mdata.gravity) * ustar * ustar;
            u_star_arr(i,j,k) = mdata.kappa * umm_arr(i,j,k) / std::log(mdata.zref / z0);
            ++iter;
        } while ((std::abs(u_star_arr(i,j,k) - ustar) > tol) && iter <= max_iters);

        t_star_arr(i,j,k) = 0.0;
          olen_arr(i,j,k) = 1.0e16;
            z0_arr(i,j,k) = z0;
    }

private:
    most_data mdata;
    similarity_funs sfuns;
    const amrex::Real tol = 1.0e-5;
};


/**
 * Adiabatic with modified charnock roughness
 */
struct adiabatic_mod_charnock
{
    adiabatic_mod_charnock (amrex::Real zref,
                            amrex::Real flux,
                            amrex::Real depth)
    {
        mdata.zref = zref;
        mdata.surf_temp_flux = flux;
        mdata.Cnk_d = depth;
        mdata.Cnk_b = mdata.Cnk_b1 * std::log(mdata.Cnk_b2 / mdata.Cnk_d);
    }

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    void
    iterate_flux (const int& i,
                  const int& j,
                  const int& k,
                  const int& max_iters,
                  const amrex::Array4<amrex::Real>& z0_arr,
                  const amrex::Array4<const amrex::Real>& umm_arr,
                  const amrex::Array4<const amrex::Real>& /*tm_arr*/,
                  const amrex::Array4<amrex::Real>& u_star_arr,
                  const amrex::Array4<amrex::Real>& t_star_arr,
                  const amrex::Array4<amrex::Real>& /*t_surf_arr*/,
                  const amrex::Array4<amrex::Real>& olen_arr) const
    {
        int iter = 0;
        amrex::Real ustar = 0.0;
        amrex::Real z0    = 0.0;
        u_star_arr(i,j,k) = mdata.kappa * umm_arr(i,j,k) / std::log(mdata.zref / z0_arr(i,j,k));
        do {
            ustar = u_star_arr(i,j,k);
            z0    = std::exp( (2.7*ustar - 1.8/mdata.Cnk_b) / (ustar + 0.17/mdata.Cnk_b) );
            u_star_arr(i,j,k) = mdata.kappa * umm_arr(i,j,k) / std::log(mdata.zref / z0);
            ++iter;
        } while ((std::abs(u_star_arr(i,j,k) - ustar) > tol) && iter <= max_iters);

        t_star_arr(i,j,k) = 0.0;
          olen_arr(i,j,k) = 1.0e16;
            z0_arr(i,j,k) = z0;
    }

private:
    most_data mdata;
    similarity_funs sfuns;
    const amrex::Real tol = 1.0e-5;
};


/**
 * Surface flux with constant roughness
 */
struct surface_flux
{
    surface_flux (amrex::Real zref,
                  amrex::Real flux)
    {
        mdata.zref = zref;
        mdata.surf_temp_flux = flux;
    }

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    void
    iterate_flux (const int& i,
                  const int& j,
                  const int& k,
                  const int& max_iters,
                  const amrex::Array4<const amrex::Real>& z0_arr,
                  const amrex::Array4<const amrex::Real>& umm_arr,
                  const amrex::Array4<const amrex::Real>& tm_arr,
                  const amrex::Array4<amrex::Real>& u_star_arr,
                  const amrex::Array4<amrex::Real>& t_star_arr,
                  const amrex::Array4<amrex::Real>& t_surf_arr,
                  const amrex::Array4<amrex::Real>& olen_arr) const
    {
        int iter = 0;
        amrex::Real ustar = 0.0;
        amrex::Real zeta  = 0.0;
        amrex::Real psi_m = 0.0;
        amrex::Real psi_h = 0.0;
        amrex::Real Olen  = 0.0;
        u_star_arr(i,j,k) = mdata.kappa * umm_arr(i,j,k) / std::log(mdata.zref / z0_arr(i,j,k));
        do {
            ustar = u_star_arr(i,j,k);
            Olen = -ustar * ustar * ustar * tm_arr(i,j,k) /
                   (mdata.kappa * mdata.gravity * mdata.surf_temp_flux);
            zeta  = mdata.zref / Olen;
            psi_m = sfuns.calc_psi_m(zeta);
            psi_h = sfuns.calc_psi_h(zeta);
            u_star_arr(i,j,k) = mdata.kappa * umm_arr(i,j,k) / (std::log(mdata.zref / z0_arr(i,j,k)) - psi_m);
            ++iter;
        } while ((std::abs(u_star_arr(i,j,k) - ustar) > tol) && iter <= max_iters);

        t_surf_arr(i,j,k) = mdata.surf_temp_flux * (std::log(mdata.zref / z0_arr(i,j,k)) - psi_h) /
                            (u_star_arr(i,j,k) * mdata.kappa) + tm_arr(i,j,k);
        t_star_arr(i,j,k) = -mdata.surf_temp_flux / u_star_arr(i,j,k);
        olen_arr(i,j,k)   = Olen;
    }

private:
    most_data mdata;
    similarity_funs sfuns;
    const amrex::Real tol = 1.0e-5;
};


/**
 * Surface flux with charnock roughness
 */
struct surface_flux_charnock
{
    surface_flux_charnock (amrex::Real zref,
                           amrex::Real flux,
                           amrex::Real cnk_a)
    {
        mdata.zref = zref;
        mdata.surf_temp_flux = flux;
        mdata.Cnk_a = cnk_a;
    }

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    void
    iterate_flux (const int& i,
                  const int& j,
                  const int& k,
                  const int& max_iters,
                  const amrex::Array4<amrex::Real>& z0_arr,
                  const amrex::Array4<const amrex::Real>& umm_arr,
                  const amrex::Array4<const amrex::Real>& tm_arr,
                  const amrex::Array4<amrex::Real>& u_star_arr,
                  const amrex::Array4<amrex::Real>& t_star_arr,
                  const amrex::Array4<amrex::Real>& t_surf_arr,
                  const amrex::Array4<amrex::Real>& olen_arr) const
    {
        int iter = 0;
        amrex::Real ustar = 0.0;
        amrex::Real z0    = 0.0;
        amrex::Real zeta  = 0.0;
        amrex::Real psi_m = 0.0;
        amrex::Real psi_h = 0.0;
        amrex::Real Olen  = 0.0;
        u_star_arr(i,j,k) = mdata.kappa * umm_arr(i,j,k) / std::log(mdata.zref / z0_arr(i,j,k));
        do {
            ustar = u_star_arr(i,j,k);
            z0    = (mdata.Cnk_a / mdata.gravity) * ustar * ustar;
            Olen = -ustar * ustar * ustar * tm_arr(i,j,k) /
                   (mdata.kappa * mdata.gravity * mdata.surf_temp_flux);
            zeta  = mdata.zref / Olen;
            psi_m = sfuns.calc_psi_m(zeta);
            psi_h = sfuns.calc_psi_h(zeta);
            u_star_arr(i,j,k) = mdata.kappa * umm_arr(i,j,k) / (std::log(mdata.zref / z0) - psi_m);
            ++iter;
        } while ((std::abs(u_star_arr(i,j,k) - ustar) > tol) && iter <= max_iters);

        t_surf_arr(i,j,k) = mdata.surf_temp_flux * (std::log(mdata.zref / z0) - psi_h) /
                            (u_star_arr(i,j,k) * mdata.kappa) + tm_arr(i,j,k);
        t_star_arr(i,j,k) = -mdata.surf_temp_flux / u_star_arr(i,j,k);
          olen_arr(i,j,k) = Olen;
           z0_arr(i,j,k)  = z0;
    }

private:
    most_data mdata;
    similarity_funs sfuns;
    const amrex::Real tol = 1.0e-5;
};


/**
 * Surface flux with modified charnock roughness
 */
struct surface_flux_mod_charnock
{
    surface_flux_mod_charnock (amrex::Real zref,
                               amrex::Real flux,
                               amrex::Real depth)
    {
        mdata.zref = zref;
        mdata.surf_temp_flux = flux;
        mdata.Cnk_d = depth;
        mdata.Cnk_b = mdata.Cnk_b1 * std::log(mdata.Cnk_b2 / mdata.Cnk_d);
    }

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    void
    iterate_flux (const int& i,
                  const int& j,
                  const int& k,
                  const int& max_iters,
                  const amrex::Array4<amrex::Real>& z0_arr,
                  const amrex::Array4<const amrex::Real>& umm_arr,
                  const amrex::Array4<const amrex::Real>& tm_arr,
                  const amrex::Array4<amrex::Real>& u_star_arr,
                  const amrex::Array4<amrex::Real>& t_star_arr,
                  const amrex::Array4<amrex::Real>& t_surf_arr,
                  const amrex::Array4<amrex::Real>& olen_arr) const
    {
        int iter = 0;
        amrex::Real ustar = 0.0;
        amrex::Real z0    = 0.0;
        amrex::Real zeta  = 0.0;
        amrex::Real psi_m = 0.0;
        amrex::Real psi_h = 0.0;
        amrex::Real Olen  = 0.0;
        u_star_arr(i,j,k) = mdata.kappa * umm_arr(i,j,k) / std::log(mdata.zref / z0_arr(i,j,k));
        do {
            ustar = u_star_arr(i,j,k);
            z0    = std::exp( (2.7*ustar - 1.8/mdata.Cnk_b) / (ustar + 0.17/mdata.Cnk_b) );
            Olen = -ustar * ustar * ustar * tm_arr(i,j,k) /
                   (mdata.kappa * mdata.gravity * mdata.surf_temp_flux);
            zeta  = mdata.zref / Olen;
            psi_m = sfuns.calc_psi_m(zeta);
            psi_h = sfuns.calc_psi_h(zeta);
            u_star_arr(i,j,k) = mdata.kappa * umm_arr(i,j,k) / (std::log(mdata.zref / z0) - psi_m);
            ++iter;
        } while ((std::abs(u_star_arr(i,j,k) - ustar) > tol) && iter <= max_iters);

        t_surf_arr(i,j,k) = mdata.surf_temp_flux * (std::log(mdata.zref / z0) - psi_h) /
                            (u_star_arr(i,j,k) * mdata.kappa) + tm_arr(i,j,k);
        t_star_arr(i,j,k) = -mdata.surf_temp_flux / u_star_arr(i,j,k);
          olen_arr(i,j,k) = Olen;
           z0_arr(i,j,k)  = z0;
    }

private:
    most_data mdata;
    similarity_funs sfuns;
    const amrex::Real tol = 1.0e-5;
};


/**
 * Surface temperature with constant roughness
 */
struct surface_temp
{
    surface_temp (amrex::Real zref,
                  amrex::Real flux)
    {
        mdata.zref = zref;
        mdata.surf_temp_flux = flux;
    }

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    void
    iterate_flux (const int& i,
                  const int& j,
                  const int& k,
                  const int& max_iters,
                  const amrex::Array4<const amrex::Real>& z0_arr,
                  const amrex::Array4<const amrex::Real>& umm_arr,
                  const amrex::Array4<const amrex::Real>& tm_arr,
                  const amrex::Array4<amrex::Real>& u_star_arr,
                  const amrex::Array4<amrex::Real>& t_star_arr,
                  const amrex::Array4<amrex::Real>& t_surf_arr,
                  const amrex::Array4<amrex::Real>& olen_arr) const
    {
        int iter = 0;
        amrex::Real ustar = 0.0;
        amrex::Real tflux = 0.0;
        amrex::Real zeta  = 0.0;
        amrex::Real psi_m = 0.0;
        amrex::Real psi_h = 0.0;
        amrex::Real Olen  = 0.0;
        u_star_arr(i,j,k) = mdata.kappa * umm_arr(i,j,k) / std::log(mdata.zref / z0_arr(i,j,k));
        do {
            ustar = u_star_arr(i,j,k);
            tflux = -(tm_arr(i,j,k) - t_surf_arr(i,j,k)) * ustar * mdata.kappa /
                     (std::log(mdata.zref / z0_arr(i,j,k)) - psi_h);
            Olen = -ustar * ustar * ustar * tm_arr(i,j,k) /
                    (mdata.kappa * mdata.gravity * tflux);
            zeta  = mdata.zref / Olen;
            psi_m = sfuns.calc_psi_m(zeta);
            psi_h = sfuns.calc_psi_h(zeta);
            u_star_arr(i,j,k) = mdata.kappa * umm_arr(i,j,k) / (std::log(mdata.zref / z0_arr(i,j,k)) - psi_m);
            ++iter;
        } while ((std::abs(u_star_arr(i,j,k) - ustar) > tol) && iter <= max_iters);

        t_star_arr(i,j,k) = mdata.kappa * (tm_arr(i,j,k) - t_surf_arr(i,j,k)) /
                            (std::log(mdata.zref / z0_arr(i,j,k)) - psi_h);
        olen_arr(i,j,k)   = Olen;
    }

private:
    most_data mdata;
    similarity_funs sfuns;
    const amrex::Real tol = 1.0e-5;
};


/**
 * Surface temperature with charnock roughness
 */
struct surface_temp_charnock
{
    surface_temp_charnock (amrex::Real zref,
                           amrex::Real flux,
                           amrex::Real cnk_a)
    {
        mdata.zref = zref;
        mdata.surf_temp_flux = flux;
        mdata.Cnk_a = cnk_a;
    }

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    void
    iterate_flux (const int& i,
                  const int& j,
                  const int& k,
                  const int& max_iters,
                  const amrex::Array4<amrex::Real>& z0_arr,
                  const amrex::Array4<const amrex::Real>& umm_arr,
                  const amrex::Array4<const amrex::Real>& tm_arr,
                  const amrex::Array4<amrex::Real>& u_star_arr,
                  const amrex::Array4<amrex::Real>& t_star_arr,
                  const amrex::Array4<amrex::Real>& t_surf_arr,
                  const amrex::Array4<amrex::Real>& olen_arr) const
    {
        int iter = 0;
        amrex::Real ustar = 0.0;
        amrex::Real z0    = 0.0;
        amrex::Real tflux = 0.0;
        amrex::Real zeta  = 0.0;
        amrex::Real psi_m = 0.0;
        amrex::Real psi_h = 0.0;
        amrex::Real Olen  = 0.0;
        u_star_arr(i,j,k) = mdata.kappa * umm_arr(i,j,k) / std::log(mdata.zref / z0_arr(i,j,k));
        do {
            ustar = u_star_arr(i,j,k);
            z0    = (mdata.Cnk_a / mdata.gravity) * ustar * ustar;
            tflux = -(tm_arr(i,j,k) - t_surf_arr(i,j,k)) * ustar * mdata.kappa /
                     (std::log(mdata.zref / z0) - psi_h);
            Olen = -ustar * ustar * ustar * tm_arr(i,j,k) /
                    (mdata.kappa * mdata.gravity * tflux);
            zeta  = mdata.zref / Olen;
            psi_m = sfuns.calc_psi_m(zeta);
            psi_h = sfuns.calc_psi_h(zeta);
            u_star_arr(i,j,k) = mdata.kappa * umm_arr(i,j,k) / (std::log(mdata.zref / z0) - psi_m);
            ++iter;
        } while ((std::abs(u_star_arr(i,j,k) - ustar) > tol) && iter <= max_iters);

        t_star_arr(i,j,k) = mdata.kappa * (tm_arr(i,j,k) - t_surf_arr(i,j,k)) /
                            (std::log(mdata.zref / z0) - psi_h);
          olen_arr(i,j,k) = Olen;
            z0_arr(i,j,k) = z0;
    }

private:
    most_data mdata;
    similarity_funs sfuns;
    const amrex::Real tol = 1.0e-5;
};


/**
 * Surface temperature with modified charnock roughness
 */
struct surface_temp_mod_charnock
{
    surface_temp_mod_charnock (amrex::Real zref,
                               amrex::Real flux,
                               amrex::Real depth)
    {
        mdata.zref = zref;
        mdata.surf_temp_flux = flux;
        mdata.Cnk_d = depth;
        mdata.Cnk_b = mdata.Cnk_b1 * std::log(mdata.Cnk_b2 / mdata.Cnk_d);
    }

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    void
    iterate_flux (const int& i,
                  const int& j,
                  const int& k,
                  const int& max_iters,
                  const amrex::Array4<amrex::Real>& z0_arr,
                  const amrex::Array4<const amrex::Real>& umm_arr,
                  const amrex::Array4<const amrex::Real>& tm_arr,
                  const amrex::Array4<amrex::Real>& u_star_arr,
                  const amrex::Array4<amrex::Real>& t_star_arr,
                  const amrex::Array4<amrex::Real>& t_surf_arr,
                  const amrex::Array4<amrex::Real>& olen_arr) const
    {
        int iter = 0;
        amrex::Real ustar = 0.0;
        amrex::Real z0    = 0.0;
        amrex::Real tflux = 0.0;
        amrex::Real zeta  = 0.0;
        amrex::Real psi_m = 0.0;
        amrex::Real psi_h = 0.0;
        amrex::Real Olen  = 0.0;
        u_star_arr(i,j,k) = mdata.kappa * umm_arr(i,j,k) / std::log(mdata.zref / z0_arr(i,j,k));
        do {
            ustar = u_star_arr(i,j,k);
            z0    = std::exp( (2.7*ustar - 1.8/mdata.Cnk_b) / (ustar + 0.17/mdata.Cnk_b) );
            tflux = -(tm_arr(i,j,k) - t_surf_arr(i,j,k)) * ustar * mdata.kappa /
                     (std::log(mdata.zref / z0) - psi_h);
            Olen = -ustar * ustar * ustar * tm_arr(i,j,k) /
                    (mdata.kappa * mdata.gravity * tflux);
            zeta  = mdata.zref / Olen;
            psi_m = sfuns.calc_psi_m(zeta);
            psi_h = sfuns.calc_psi_h(zeta);
            u_star_arr(i,j,k) = mdata.kappa * umm_arr(i,j,k) / (std::log(mdata.zref / z0) - psi_m);
            ++iter;
        } while ((std::abs(u_star_arr(i,j,k) - ustar) > tol) && iter <= max_iters);

        t_star_arr(i,j,k) = mdata.kappa * (tm_arr(i,j,k) - t_surf_arr(i,j,k)) /
                            (std::log(mdata.zref / z0) - psi_h);
          olen_arr(i,j,k) = Olen;
            z0_arr(i,j,k) = z0;
    }

private:
    most_data mdata;
    similarity_funs sfuns;
    const amrex::Real tol = 1.0e-5;
};


/**
 * Moeng flux formulation
 */
struct moeng_flux
{
    moeng_flux (int l_klo)
      :  klo(l_klo) {}


    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real
    compute_q_flux (const int& /*i*/,
                    const int& /*j*/,
                    const int& /*k*/,
                    const int& /*n*/,
                    const amrex::Array4<const amrex::Real>& /*cons_arr*/,
                    const amrex::Array4<const amrex::Real>& /*velx_arr*/,
                    const amrex::Array4<const amrex::Real>& /*vely_arr*/,
                    const amrex::Array4<const amrex::Real>& /*umm_arr*/,
                    const amrex::Array4<const amrex::Real>& /*tm_arr*/,
                    const amrex::Array4<const amrex::Real>& /*u_star_arr*/,
                    const amrex::Array4<const amrex::Real>& /*q_star_arr*/,
                    const amrex::Array4<const amrex::Real>& /*t_surf_arr*/) const
    {
        // TODO: Integrate MOST with moisture and DONELAN FLUX type
        amrex::Real moflux  = 0.0;

        return moflux;
    }

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real
    compute_t_flux (const int& i,
                    const int& j,
                    const int& /*k*/,
                    const int& n,
                    const amrex::Array4<const amrex::Real>& cons_arr,
                    const amrex::Array4<const amrex::Real>& velx_arr,
                    const amrex::Array4<const amrex::Real>& vely_arr,
                    const amrex::Array4<const amrex::Real>& umm_arr,
                    const amrex::Array4<const amrex::Real>& tm_arr,
                    const amrex::Array4<const amrex::Real>& u_star_arr,
                    const amrex::Array4<const amrex::Real>& t_star_arr,
                    const amrex::Array4<const amrex::Real>& t_surf_arr) const
    {
        int ix, jx;
        ix = i < lbound(velx_arr).x    ? lbound(velx_arr).x   : i;
        jx = j < lbound(velx_arr).y    ? lbound(velx_arr).y   : j;
        ix = ix > ubound(velx_arr).x-1 ? ubound(velx_arr).x-1 : ix;
        jx = jx > ubound(velx_arr).y   ? ubound(velx_arr).y   : jx;

        int iy, jy;
        iy = i  < lbound(vely_arr).x   ? lbound(vely_arr).x   : i;
        jy = j  < lbound(vely_arr).y   ? lbound(vely_arr).y   : j;
        iy = iy > ubound(vely_arr).x   ? ubound(vely_arr).x   : iy;
        jy = jy > ubound(vely_arr).y-1 ? ubound(vely_arr).y-1 : jy;

        int ic, jc;
        ic = i  < lbound(cons_arr).x ? lbound(cons_arr).x : i;
        jc = j  < lbound(cons_arr).y ? lbound(cons_arr).y : j;
        ic = ic > ubound(cons_arr).x ? ubound(cons_arr).x : ic;
        jc = jc > ubound(cons_arr).y ? ubound(cons_arr).y : jc;

        amrex::Real velx  = 0.5 *( velx_arr(ix,jx,klo) + velx_arr(ix+1,jx  ,klo) );
        amrex::Real vely  = 0.5 *( vely_arr(iy,jy,klo) + vely_arr(iy  ,jy+1,klo) );
        amrex::Real rho   = cons_arr(ic,jc,klo,Rho_comp);
        amrex::Real theta = cons_arr(ic,jc,klo,n) / rho;

        amrex::Real theta_mean  = tm_arr(ic,jc,klo);
        amrex::Real wsp_mean    = umm_arr(ic,jc,klo);
        amrex::Real ustar       = u_star_arr(ic,jc,klo);
        amrex::Real tstar       = t_star_arr(ic,jc,klo);
        amrex::Real theta_surf  = t_surf_arr(ic,jc,klo);

        amrex::Real wsp     = sqrt(velx*velx+vely*vely);
        amrex::Real num1    = wsp * (theta_mean-theta_surf);
        amrex::Real num2    = wsp_mean * (theta-theta_mean);
        amrex::Real moflux  = (std::abs(tstar) > eps) ?
                              -tstar*ustar*(num1+num2)/((theta_mean-theta_surf)*wsp_mean) : 0.0;

        return moflux;
    }

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real
    compute_u_flux (const int& i,
                    const int& j,
                    const int& /*k*/,
                    const amrex::Array4<const amrex::Real>& cons_arr,
                    const amrex::Array4<const amrex::Real>& velx_arr,
                    const amrex::Array4<const amrex::Real>& vely_arr,
                    const amrex::Array4<const amrex::Real>& umm_arr,
                    const amrex::Array4<const amrex::Real>& um_arr,
                    const amrex::Array4<const amrex::Real>& u_star_arr) const
    {
        int iylo = i <= lbound(vely_arr).x ? lbound(vely_arr).x : i-1;
        int iyhi = i >  ubound(vely_arr).x ? ubound(vely_arr).x : i;

        int jy;
        jy = j  < lbound(vely_arr).y   ? lbound(vely_arr).y   : j;
        jy = jy > ubound(vely_arr).y-1 ? ubound(vely_arr).y-1 : jy;

        int ic, jc;
        ic = i  < lbound(cons_arr).x+1 ? lbound(cons_arr).x+1 : i;
        jc = j  < lbound(cons_arr).y   ? lbound(cons_arr).y   : j;
        ic = ic > ubound(cons_arr).x   ? ubound(cons_arr).x   : ic;
        jc = jc > ubound(cons_arr).y   ? ubound(cons_arr).y   : jc;

        amrex::Real velx  = velx_arr(i,j,klo);
        amrex::Real vely  = 0.25*( vely_arr(iyhi,jy,klo)+vely_arr(iyhi,jy+1,klo)
                                 + vely_arr(iylo,jy,klo)+vely_arr(iylo,jy+1,klo) );
        amrex::Real rho   = 0.5 *( cons_arr(ic-1,jc,klo,Rho_comp)
                                 + cons_arr(ic  ,jc,klo,Rho_comp) );

        amrex::Real umean    = um_arr(i,j,klo);
        amrex::Real wsp_mean = 0.5 * ( umm_arr(ic-1,jc,klo) + umm_arr(ic,jc,klo) );
        amrex::Real ustar    = 0.5 * ( u_star_arr(ic-1,jc,klo) + u_star_arr(ic,jc,klo) );

        // Note: The surface mean shear stress is decomposed into tau_xz by
        //       multiplying the modeled shear stress (rho*ustar^2) with
        //       a factor of umean/wsp_mean for directionality; this factor
        //       modifies the demoninator from what is in Moeng 1984.
        amrex::Real wsp     = sqrt(velx*velx+vely*vely);
        amrex::Real num1    = wsp * umean;
        amrex::Real num2    = wsp_mean * (velx-umean);
        amrex::Real stressx = rho*ustar*ustar * (num1+num2)/(wsp_mean*wsp_mean);

        return stressx;
    }

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real
    compute_v_flux (const int& i,
                    const int& j,
                    const int& /*k*/,
                    const amrex::Array4<const amrex::Real>& cons_arr,
                    const amrex::Array4<const amrex::Real>& velx_arr,
                    const amrex::Array4<const amrex::Real>& vely_arr,
                    const amrex::Array4<const amrex::Real>& umm_arr,
                    const amrex::Array4<const amrex::Real>& vm_arr,
                    const amrex::Array4<const amrex::Real>& u_star_arr) const
    {
        int jxlo = j <= lbound(velx_arr).y ? lbound(velx_arr).y : j-1;
        int jxhi = j >  ubound(velx_arr).y ? ubound(velx_arr).y : j;

        int ix;
        ix = i  < lbound(velx_arr).x ? lbound(velx_arr).x : i;
        ix = ix > ubound(velx_arr).x ? ubound(velx_arr).x : ix;

        int ic, jc;
        ic = i  < lbound(cons_arr).x   ? lbound(cons_arr).x   : i;
        jc = j  < lbound(cons_arr).y+1 ? lbound(cons_arr).y+1 : j;
        ic = ic > ubound(cons_arr).x   ? ubound(cons_arr).x   : ic;
        jc = jc > ubound(cons_arr).y   ? ubound(cons_arr).y   : jc;

        amrex::Real velx  = 0.25*( velx_arr(ix,jxhi,klo)+velx_arr(ix+1,jxhi,klo)
                                 + velx_arr(ix,jxlo,klo)+velx_arr(ix+1,jxlo,klo) );
        amrex::Real vely  = vely_arr(i,j,klo);
        amrex::Real rho   = 0.5*( cons_arr(ic,jc-1,klo,Rho_comp)
                                + cons_arr(ic,jc  ,klo,Rho_comp) );

        amrex::Real vmean     = vm_arr(i,j,klo);
        amrex::Real wsp_mean  = 0.5 * ( umm_arr(ic,jc-1,klo) + umm_arr(ic,jc,klo) );
        amrex::Real ustar     = 0.5 * ( u_star_arr(ic,jc-1,klo) + u_star_arr(ic,jc,klo) );

        // Note: The surface mean shear stress is decomposed into tau_yz by
        //       multiplying the modeled shear stress (rho*ustar^2) with
        //       a factor of vmean/wsp_mean for directionality; this factor
        //       modifies the demoninator from what is in Moeng 1984.
        amrex::Real wsp     = sqrt(velx*velx+vely*vely);
        amrex::Real num1    = wsp * vmean;
        amrex::Real num2    = wsp_mean * (vely-vmean);
        amrex::Real stressy = rho*ustar*ustar * (num1+num2)/(wsp_mean*wsp_mean);

        return stressy;
    }

private:
    int klo;
    const amrex::Real     eps = 1e-15;
};


/**
 * Donelan flux formulation
 */
struct donelan_flux
{
    donelan_flux (int l_klo)
      :  klo(l_klo) {}


    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real
    compute_q_flux (const int& /*i*/,
                    const int& /*j*/,
                    const int& /*k*/,
                    const int& /*n*/,
                    const amrex::Array4<const amrex::Real>& /*cons_arr*/,
                    const amrex::Array4<const amrex::Real>& /*velx_arr*/,
                    const amrex::Array4<const amrex::Real>& /*vely_arr*/,
                    const amrex::Array4<const amrex::Real>& /*umm_arr*/,
                    const amrex::Array4<const amrex::Real>& /*tm_arr*/,
                    const amrex::Array4<const amrex::Real>& /*u_star_arr*/,
                    const amrex::Array4<const amrex::Real>& /*q_star_arr*/,
                    const amrex::Array4<const amrex::Real>& /*t_surf_arr*/) const
    {
        // TODO: Integrate MOST with moisture and DONELAN FLUX type
        amrex::Real moflux  = 0.0;

        return moflux;
    }

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real
    compute_t_flux (const int& i,
                    const int& j,
                    const int& /*k*/,
                    const int& /*n*/,
                    const amrex::Array4<const amrex::Real>& cons_arr,
                    const amrex::Array4<const amrex::Real>& /*velx_arr*/,
                    const amrex::Array4<const amrex::Real>& /*vely_arr*/,
                    const amrex::Array4<const amrex::Real>& umm_arr,
                    const amrex::Array4<const amrex::Real>& tm_arr,
                    const amrex::Array4<const amrex::Real>& /*u_star_arr*/,
                    const amrex::Array4<const amrex::Real>& /*t_star_arr*/,
                    const amrex::Array4<const amrex::Real>& t_surf_arr) const
    {
        int ic, jc;
        ic = i  < lbound(cons_arr).x ? lbound(cons_arr).x : i;
        jc = j  < lbound(cons_arr).y ? lbound(cons_arr).y : j;
        ic = ic > ubound(cons_arr).x ? ubound(cons_arr).x : ic;
        jc = jc > ubound(cons_arr).y ? ubound(cons_arr).y : jc;

        amrex::Real Cd = 0.0012;
        amrex::Real wsp_mean    = umm_arr(ic,jc,klo);
        amrex::Real theta_surf  = t_surf_arr(ic,jc,klo);
        amrex::Real theta_mean  = tm_arr(ic,jc,klo);
        amrex::Real moflux      = Cd * wsp_mean * (theta_surf - theta_mean);

        return moflux;
    }

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real
    compute_u_flux (const int& i,
                    const int& j,
                    const int& /*k*/,
                    const amrex::Array4<const amrex::Real>& cons_arr,
                    const amrex::Array4<const amrex::Real>& velx_arr,
                    const amrex::Array4<const amrex::Real>& vely_arr,
                    const amrex::Array4<const amrex::Real>& umm_arr,
                    const amrex::Array4<const amrex::Real>& /*um_arr*/,
                    const amrex::Array4<const amrex::Real>& /*u_star_arr*/) const
    {
        int iylo = i <= lbound(vely_arr).x ? lbound(vely_arr).x : i-1;
        int iyhi = i >  ubound(vely_arr).x ? ubound(vely_arr).x : i;

        int jy;
        jy = j  < lbound(vely_arr).y   ? lbound(vely_arr).y   : j;
        jy = jy > ubound(vely_arr).y-1 ? ubound(vely_arr).y-1 : jy;

        int ic, jc;
        ic = i  < lbound(cons_arr).x+1 ? lbound(cons_arr).x+1 : i;
        jc = j  < lbound(cons_arr).y   ? lbound(cons_arr).y   : j;
        ic = ic > ubound(cons_arr).x   ? ubound(cons_arr).x   : ic;
        jc = jc > ubound(cons_arr).y   ? ubound(cons_arr).y   : jc;

        amrex::Real velx  = velx_arr(i,j,klo);
        amrex::Real vely  = 0.25*( vely_arr(iyhi,jy,klo)+vely_arr(iyhi,jy+1,klo)
                                 + vely_arr(iylo,jy,klo)+vely_arr(iylo,jy+1,klo) );
        amrex::Real rho   = 0.5 *( cons_arr(ic-1,jc,klo,Rho_comp)
                                 + cons_arr(ic  ,jc,klo,Rho_comp) );

        amrex::Real Cd       = 0.001;
        const amrex::Real c  = 7e-5;
        amrex::Real wsp      = sqrt(velx*velx+vely*vely);
        amrex::Real wsp_mean = 0.5 * ( umm_arr(ic-1,jc,klo) + umm_arr(ic,jc,klo) );
        if (wsp_mean <= 5.0) {
            Cd = 0.001;
        } else if (wsp_mean < 25.0 && wsp_mean > 5.0) {
            Cd = 0.001 + c * (wsp_mean - 5.0);
        } else {
            Cd = 0.0024;
        }
        amrex::Real stressx = rho * Cd * velx * wsp;

        return stressx;
    }

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real
    compute_v_flux (const int& i,
                    const int& j,
                    const int& /*k*/,
                    const amrex::Array4<const amrex::Real>& cons_arr,
                    const amrex::Array4<const amrex::Real>& velx_arr,
                    const amrex::Array4<const amrex::Real>& vely_arr,
                    const amrex::Array4<const amrex::Real>& umm_arr,
                    const amrex::Array4<const amrex::Real>& /*vm_arr*/,
                    const amrex::Array4<const amrex::Real>& /*u_star_arr*/) const
    {
        int jxlo = j <= lbound(velx_arr).y ? lbound(velx_arr).y : j-1;
        int jxhi = j >  ubound(velx_arr).y ? ubound(velx_arr).y : j;

        int ix;
        ix = i  < lbound(velx_arr).x ? lbound(velx_arr).x : i;
        ix = ix > ubound(velx_arr).x ? ubound(velx_arr).x : ix;

        int ic, jc;
        ic = i  < lbound(cons_arr).x   ? lbound(cons_arr).x   : i;
        jc = j  < lbound(cons_arr).y+1 ? lbound(cons_arr).y+1 : j;
        ic = ic > ubound(cons_arr).x   ? ubound(cons_arr).x   : ic;
        jc = jc > ubound(cons_arr).y   ? ubound(cons_arr).y   : jc;

        amrex::Real velx  = 0.25*( velx_arr(ix,jxhi,klo)+velx_arr(ix+1,jxhi,klo)
                                 + velx_arr(ix,jxlo,klo)+velx_arr(ix+1,jxlo,klo) );
        amrex::Real vely  = vely_arr(i,j,klo);
        amrex::Real rho   = 0.5*( cons_arr(ic,jc-1,klo,Rho_comp)
                                + cons_arr(ic,jc  ,klo,Rho_comp) );

        amrex::Real Cd       = 0.001;
        const amrex::Real c  = 7e-5;
        amrex::Real wsp      = sqrt(velx*velx+vely*vely);
        amrex::Real wsp_mean = 0.5 * ( umm_arr(ic,jc-1,klo) + umm_arr(ic,jc,klo) );
        if (wsp_mean <= 5.0) {
            Cd = 0.001;
        } else if (wsp_mean < 25.0 && wsp_mean > 5.0) {
            Cd = 0.001 + c * (wsp_mean - 5.0);
        } else {
            Cd = 0.0024;
        }
        amrex::Real stressy = rho * Cd * vely * wsp;

        return stressy;
    }

private:
    int klo;
};


/**
 * Custom flux formulation
 */
struct custom_flux
{
    custom_flux (int l_klo)
      :  klo(l_klo) {}


    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real
    compute_q_flux (const int& i,
                    const int& j,
                    const int& /*k*/,
                    const int& /*n*/,
                    const amrex::Array4<const amrex::Real>& cons_arr,
                    const amrex::Array4<const amrex::Real>& /*velx_arr*/,
                    const amrex::Array4<const amrex::Real>& /*vely_arr*/,
                    const amrex::Array4<const amrex::Real>& /*umm_arr*/,
                    const amrex::Array4<const amrex::Real>& /*tm_arr*/,
                    const amrex::Array4<const amrex::Real>& /*u_star_arr*/,
                    const amrex::Array4<const amrex::Real>& q_star_arr,
                    const amrex::Array4<const amrex::Real>& /*t_surf_arr*/) const
    {
        int ic, jc;
        ic = i  < lbound(cons_arr).x ? lbound(cons_arr).x : i;
        jc = j  < lbound(cons_arr).y ? lbound(cons_arr).y : j;
        ic = ic > ubound(cons_arr).x ? ubound(cons_arr).x : ic;
        jc = jc > ubound(cons_arr).y ? ubound(cons_arr).y : jc;

        amrex::Real rho = cons_arr(ic,jc,klo,Rho_comp);

        amrex::Real qstar   = q_star_arr(ic,jc,klo);
        amrex::Real moflux  = (std::abs(qstar) > eps) ? -rho * qstar : 0.0;

        return moflux;
    }

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real
    compute_t_flux (const int& i,
                    const int& j,
                    const int& /*k*/,
                    const int& /*n*/,
                    const amrex::Array4<const amrex::Real>& cons_arr,
                    const amrex::Array4<const amrex::Real>& /*velx_arr*/,
                    const amrex::Array4<const amrex::Real>& /*vely_arr*/,
                    const amrex::Array4<const amrex::Real>& /*umm_arr*/,
                    const amrex::Array4<const amrex::Real>& /*tm_arr*/,
                    const amrex::Array4<const amrex::Real>& u_star_arr,
                    const amrex::Array4<const amrex::Real>& t_star_arr,
                    const amrex::Array4<const amrex::Real>& /*t_surf_arr*/) const
    {
        int ic, jc;
        ic = i  < lbound(cons_arr).x ? lbound(cons_arr).x : i;
        jc = j  < lbound(cons_arr).y ? lbound(cons_arr).y : j;
        ic = ic > ubound(cons_arr).x ? ubound(cons_arr).x : ic;
        jc = jc > ubound(cons_arr).y ? ubound(cons_arr).y : jc;

        amrex::Real ustar   = u_star_arr(ic,jc,klo);
        amrex::Real tstar   = t_star_arr(ic,jc,klo);
        amrex::Real moflux  = (std::abs(tstar) > eps) ? -tstar*ustar : 0.0;

        return moflux;
    }

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real
    compute_u_flux (const int& i,
                    const int& j,
                    const int& /*k*/,
                    const amrex::Array4<const amrex::Real>& cons_arr,
                    const amrex::Array4<const amrex::Real>& velx_arr,
                    const amrex::Array4<const amrex::Real>& vely_arr,
                    const amrex::Array4<const amrex::Real>& /*umm_arr*/,
                    const amrex::Array4<const amrex::Real>& /*um_arr*/,
                    const amrex::Array4<const amrex::Real>& u_star_arr) const
    {
        int iylo = i <= lbound(vely_arr).x ? lbound(vely_arr).x : i-1;
        int iyhi = i >  ubound(vely_arr).x ? ubound(vely_arr).x : i;

        int jy;
        jy = j  < lbound(vely_arr).y   ? lbound(vely_arr).y   : j;
        jy = jy > ubound(vely_arr).y-1 ? ubound(vely_arr).y-1 : jy;

        int ic, jc;
        ic = i  < lbound(cons_arr).x+1 ? lbound(cons_arr).x+1 : i;
        jc = j  < lbound(cons_arr).y   ? lbound(cons_arr).y   : j;
        ic = ic > ubound(cons_arr).x   ? ubound(cons_arr).x   : ic;
        jc = jc > ubound(cons_arr).y   ? ubound(cons_arr).y   : jc;

        amrex::Real velx  = velx_arr(i,j,klo);
        amrex::Real vely  = 0.25*( vely_arr(iyhi,jy,klo)+vely_arr(iyhi,jy+1,klo)
                                 + vely_arr(iylo,jy,klo)+vely_arr(iylo,jy+1,klo) );
        amrex::Real rho   = 0.5 *( cons_arr(ic-1,jc,klo,Rho_comp)
                                 + cons_arr(ic  ,jc,klo,Rho_comp) );

        amrex::Real ustar   = 0.5 * ( u_star_arr(ic-1,jc,klo) + u_star_arr(ic,jc,klo) );
        amrex::Real wsp     = sqrt(velx*velx+vely*vely);
        amrex::Real stressx = rho * ustar * ustar * velx / wsp;

        return stressx;
    }

    AMREX_GPU_DEVICE
    AMREX_FORCE_INLINE
    amrex::Real
    compute_v_flux (const int& i,
                    const int& j,
                    const int& /*k*/,
                    const amrex::Array4<const amrex::Real>& cons_arr,
                    const amrex::Array4<const amrex::Real>& velx_arr,
                    const amrex::Array4<const amrex::Real>& vely_arr,
                    const amrex::Array4<const amrex::Real>& /*umm_arr*/,
                    const amrex::Array4<const amrex::Real>& /*vm_arr*/,
                    const amrex::Array4<const amrex::Real>& u_star_arr) const
    {
        int jxlo = j <= lbound(velx_arr).y ? lbound(velx_arr).y : j-1;
        int jxhi = j >  ubound(velx_arr).y ? ubound(velx_arr).y : j;

        int ix;
        ix = i  < lbound(velx_arr).x ? lbound(velx_arr).x : i;
        ix = ix > ubound(velx_arr).x ? ubound(velx_arr).x : ix;

        int ic, jc;
        ic = i  < lbound(cons_arr).x   ? lbound(cons_arr).x   : i;
        jc = j  < lbound(cons_arr).y+1 ? lbound(cons_arr).y+1 : j;
        ic = ic > ubound(cons_arr).x   ? ubound(cons_arr).x   : ic;
        jc = jc > ubound(cons_arr).y   ? ubound(cons_arr).y   : jc;

        amrex::Real velx  = 0.25*( velx_arr(ix,jxhi,klo)+velx_arr(ix+1,jxhi,klo)
                                 + velx_arr(ix,jxlo,klo)+velx_arr(ix+1,jxlo,klo) );
        amrex::Real vely  = vely_arr(i,j,klo);
        amrex::Real rho   = 0.5*( cons_arr(ic,jc-1,klo,Rho_comp)
                                + cons_arr(ic,jc  ,klo,Rho_comp) );

        amrex::Real ustar   = 0.5 * ( u_star_arr(ic,jc-1,klo) + u_star_arr(ic,jc,klo) );
        amrex::Real wsp     = sqrt(velx*velx+vely*vely);
        amrex::Real stressy = rho * ustar * ustar * vely / wsp;

        return stressy;
    }

private:
    int klo;
    const amrex::Real eps = 1e-15;
};
#endif
