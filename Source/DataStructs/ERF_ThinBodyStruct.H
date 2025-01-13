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

        // Faces marked as belonging to a thin body -- used to create a mask
        // (on staggered grids), applied to momentum sources to enforce no
        // penetration
        pp.queryarr("zero_flux_xfaces", zero_xflux);
        pp.queryarr("zero_flux_yfaces", zero_yflux);
        pp.queryarr("zero_flux_zfaces", zero_zflux);
        pp.query("extend_tb_faces", extend_faces);

        xfacelist_d.resize(zero_xflux.size());
        yfacelist_d.resize(zero_yflux.size());
        zfacelist_d.resize(zero_zflux.size());

        have_xfaces = (zero_xflux.size() > 0);
        have_yfaces = (zero_yflux.size() > 0);
        have_zfaces = (zero_zflux.size() > 0);
        have_zero_flux_faces = (have_xfaces || have_yfaces || have_zfaces);
    }

    void create_extended_sets (const amrex::Box dom)
    {
        if (!extend_faces)
            return;

        if (zero_xflux.size() > 0) {
            for (int iface=0; iface < zero_xflux.size(); ++iface) {
                const auto& iv = zero_xflux[iface];
                const int jlo = amrex::max(iv[1] - 1, dom.smallEnd(1));
                const int jhi = amrex::min(iv[1] + 1,   dom.bigEnd(1));
                const int klo = amrex::max(iv[2] - 1, dom.smallEnd(2));
                const int khi = amrex::min(iv[2] + 1,   dom.bigEnd(2));
                for (int j=jlo; j <= jhi; ++j) {
                    for (int k=klo; k <= khi; ++k) {
                        extended_xfaces.insert(amrex::IntVect(iv[0], j, k));
                    }
                }
            }
        }
        if (zero_yflux.size() > 0) {
            for (int iface=0; iface < zero_yflux.size(); ++iface) {
                const auto& iv = zero_yflux[iface];
                const int ilo = amrex::max(iv[0] - 1, dom.smallEnd(0));
                const int ihi = amrex::min(iv[0] + 1,   dom.bigEnd(0));
                const int klo = amrex::max(iv[2] - 1, dom.smallEnd(2));
                const int khi = amrex::min(iv[2] + 1,   dom.bigEnd(2));
                for (int i=ilo; i <= ihi; ++i) {
                    for (int k=klo; k <= khi; ++k) {
                        extended_yfaces.insert(amrex::IntVect(i, iv[1], k));
                    }
                }
            }
        }
        if (zero_zflux.size() > 0) {
            for (int iface=0; iface < zero_zflux.size(); ++iface) {
                const auto& iv = zero_zflux[iface];
                const int ilo = amrex::max(iv[0] - 1, dom.smallEnd(0));
                const int ihi = amrex::min(iv[0] + 1,   dom.bigEnd(0));
                const int jlo = amrex::max(iv[1] - 1, dom.smallEnd(1));
                const int jhi = amrex::min(iv[1] + 1,   dom.bigEnd(1));
                for (int i=ilo; i <= ihi; ++i) {
                    for (int j=jlo; j <= jhi; ++j) {
                        extended_zfaces.insert(amrex::IntVect(i, j, iv[2]));
                    }
                }
            }
        }
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
    bool have_xfaces = false;
    bool have_yfaces = false;
    bool have_zfaces = false;
    bool have_zero_flux_faces = false;

    // Extended sets
    bool extend_faces = false;
    std::set<amrex::IntVect> extended_xfaces;
    std::set<amrex::IntVect> extended_yfaces;
    std::set<amrex::IntVect> extended_zfaces;

    amrex::Gpu::DeviceVector<amrex::IntVect> xfacelist_d;
    amrex::Gpu::DeviceVector<amrex::IntVect> yfacelist_d;
    amrex::Gpu::DeviceVector<amrex::IntVect> zfacelist_d;

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
