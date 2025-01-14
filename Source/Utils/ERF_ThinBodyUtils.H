#ifndef ERF_THINBODY_UTILS_H_
#define ERF_THINBODY_UTILS_H_

/* Check if a thin body face exists on the low side (< 0) or high side (> 0) of
   the given set of indices; return 0 if there is not an adjacent thin body.
   If stag >= 0, then the input indices correspond to a staggered quantity in
   dimension `stag`.
 */
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
int
is_touching_thin_body_xface(int i, int j, int k,
                            const amrex::Gpu::DeviceVector<amrex::IntVect>& xfaces,
                            int stag=-1)
{
    amrex::IntVect xfacelo(i  ,j,k);
    amrex::IntVect xfacehi(i+1,j,k);
    if (std::find(xfaces.begin(), xfaces.end(), xfacelo) != xfaces.end()) {
        return -1;
    } else if (std::find(xfaces.begin(), xfaces.end(), xfacehi) != xfaces.end()) {
        return 1;
    } else if (stag < 0) {
        return 0;
    }
    xfacelo[stag] -= 1;
    xfacehi[stag] -= 1;
    if (std::find(xfaces.begin(), xfaces.end(), xfacelo) != xfaces.end()) {
        return -1;
    } else if (std::find(xfaces.begin(), xfaces.end(), xfacehi) != xfaces.end()) {
        return 1;
    } else {
        return 0;
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
int
is_touching_thin_body_yface(int i, int j, int k,
                            const amrex::Gpu::DeviceVector<amrex::IntVect>& yfaces,
                            int stag=-1)
{
    amrex::IntVect yfacelo(i,j  ,k);
    amrex::IntVect yfacehi(i,j+1,k);
    if (std::find(yfaces.begin(), yfaces.end(), yfacelo) != yfaces.end()) {
        return -1;
    } else if (std::find(yfaces.begin(), yfaces.end(), yfacehi) != yfaces.end()) {
        return 1;
    } else if (stag < 0) {
        return 0;
    }
    yfacelo[stag] -= 1;
    yfacehi[stag] -= 1;
    if (std::find(yfaces.begin(), yfaces.end(), yfacelo) != yfaces.end()) {
        return -1;
    } else if (std::find(yfaces.begin(), yfaces.end(), yfacehi) != yfaces.end()) {
        return 1;
    } else {
        return 0;
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
int
is_touching_thin_body_zface(int i, int j, int k,
                            const amrex::Gpu::DeviceVector<amrex::IntVect>& zfaces,
                            int stag=-1)
{
    amrex::IntVect zfacelo(i,j,k  );
    amrex::IntVect zfacehi(i,j,k+1);
    if (std::find(zfaces.begin(), zfaces.end(), zfacelo) != zfaces.end()) {
        return -1;
    } else if (std::find(zfaces.begin(), zfaces.end(), zfacehi) != zfaces.end()) {
        return 1;
    } else if (stag < 0) {
        return 0;
    }
    zfacelo[stag] -= 1;
    zfacehi[stag] -= 1;
    if (std::find(zfaces.begin(), zfaces.end(), zfacelo) != zfaces.end()) {
        return -1;
    } else if (std::find(zfaces.begin(), zfaces.end(), zfacehi) != zfaces.end()) {
        return 1;
    } else {
        return 0;
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
