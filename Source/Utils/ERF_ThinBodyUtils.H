#ifndef ERF_THINBODY_UTILS_H_
#define ERF_THINBODY_UTILS_H_

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
thin_body_faces_to_vec(amrex::Gpu::DeviceVector<amrex::IntVect>& xfaces,
                       amrex::Gpu::DeviceVector<amrex::IntVect>& yfaces,
                       amrex::Gpu::DeviceVector<amrex::IntVect>& zfaces,
                       const std::set<amrex::IntVect>& xset,
                       const std::set<amrex::IntVect>& yset,
                       const std::set<amrex::IntVect>& zset)
{
    xfaces.resize(xset.size());
    yfaces.resize(yset.size());
    zfaces.resize(zset.size());

    if (xfaces.size() > 0) {
        amrex::Vector<amrex::IntVect> tmp(xset.begin(), xset.end());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         tmp.begin(),
                         tmp.end(),
                         xfaces.begin());
    }
    if (yfaces.size() > 0) {
        amrex::Vector<amrex::IntVect> tmp(yset.begin(), yset.end());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         tmp.begin(),
                         tmp.end(),
                         yfaces.begin());
    }
    if (zfaces.size() > 0) {
        amrex::Vector<amrex::IntVect> tmp(zset.begin(), zset.end());
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         tmp.begin(),
                         tmp.end(),
                         zfaces.begin());
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
thin_body_faces_to_vec(amrex::Gpu::DeviceVector<amrex::IntVect>& xfaces,
                       amrex::Gpu::DeviceVector<amrex::IntVect>& yfaces,
                       amrex::Gpu::DeviceVector<amrex::IntVect>& zfaces,
                       const amrex::Vector<amrex::IntVect>& xset,
                       const amrex::Vector<amrex::IntVect>& yset,
                       const amrex::Vector<amrex::IntVect>& zset)
{
    xfaces.resize(xset.size());
    yfaces.resize(yset.size());
    zfaces.resize(zset.size());

    if (xfaces.size() > 0) {
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         xset.begin(),
                         xset.end(),
                         xfaces.begin());
    }
    if (yfaces.size() > 0) {
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         yset.begin(),
                         yset.end(),
                         yfaces.begin());
    }
    if (zfaces.size() > 0) {
        amrex::Gpu::copy(amrex::Gpu::hostToDevice,
                         zset.begin(),
                         zset.end(),
                         zfaces.begin());
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
bool
is_touching_thin_body(int i, int j, int k,
                      const amrex::Gpu::DeviceVector<amrex::IntVect>& xfaces,
                      const amrex::Gpu::DeviceVector<amrex::IntVect>& yfaces,
                      const amrex::Gpu::DeviceVector<amrex::IntVect>& zfaces)
{
    // Check adjacent faces (staggered indices) relative to specified cell indices
    amrex::IntVect faceslo(i  ,j  ,k  );
    amrex::IntVect xfacehi(i+1,j  ,k  );
    amrex::IntVect yfacehi(i  ,j+1,k  );
    amrex::IntVect zfacehi(i  ,j  ,k+1);
    return ( std::find(xfaces.begin(), xfaces.end(), faceslo) != xfaces.end() ||
             std::find(yfaces.begin(), yfaces.end(), faceslo) != yfaces.end() ||
             std::find(zfaces.begin(), zfaces.end(), faceslo) != zfaces.end() ||
             std::find(xfaces.begin(), xfaces.end(), xfacehi) != xfaces.end() ||
             std::find(yfaces.begin(), yfaces.end(), yfacehi) != yfaces.end() ||
             std::find(zfaces.begin(), zfaces.end(), zfacehi) != zfaces.end() );
}

#endif
