#ifndef ERF_THINBODY_STRUCT_H_
#define ERF_THINBODY_STRUCT_H_

struct ThinImmersedBody {
    /*
     * Hold data for modeling thin immersed bodies
     */
  public:

    void init_params ()
    {
        amrex::ParmParse pp(pp_prefix);

        // Faces marked as belonging to a thin body
        pp.queryarr("zero_flux_xfaces", zero_xflux);
        pp.queryarr("zero_flux_yfaces", zero_yflux);
        pp.queryarr("zero_flux_zfaces", zero_zflux);

        have_zero_flux_faces = ((zero_xflux.size() > 0) || (zero_yflux.size() > 0) || (zero_zflux.size() > 0));
    }

    void resize_arrays (int nlevs_max)
    {
        xflux_imask.resize(nlevs_max);
        yflux_imask.resize(nlevs_max);
        zflux_imask.resize(nlevs_max);
        //overset_imask.resize(nlevs_max);
        fx.resize(nlevs_max);
        fy.resize(nlevs_max);
        fz.resize(nlevs_max);
    }

    explicit operator bool() const {
        return have_zero_flux_faces;
    }

    // Default prefix
    std::string pp_prefix {"erf"};

    // Lists of faces (indices are staggered in x, y, or )
    amrex::Vector<amrex::IntVect> zero_xflux;
    amrex::Vector<amrex::IntVect> zero_yflux;
    amrex::Vector<amrex::IntVect> zero_zflux;
    bool have_zero_flux_faces = false;

    // Masks used to zero the flux on selected faces
    amrex::Vector<std::unique_ptr<amrex::iMultiFab>> xflux_imask;
    amrex::Vector<std::unique_ptr<amrex::iMultiFab>> yflux_imask;
    amrex::Vector<std::unique_ptr<amrex::iMultiFab>> zflux_imask;
    //amrex::Vector<std::unique_ptr<amrex::iMultiFab>> overset_imask;

    // Body forces that result from the thin immersed body
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> fx;
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> fy;
    amrex::Vector<std::unique_ptr<amrex::MultiFab>> fz;
};

#endif
