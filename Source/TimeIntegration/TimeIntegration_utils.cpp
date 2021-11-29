#include <TimeIntegration.H>

using namespace amrex;

int
ComputeGhostCells(const int& spatial_order) {
  int nGhostCells;

  //TODO: Make sure we have correct number of ghost cells for different spatial orders.
  // As of Oct. 27, 2021 we haven't played around with number of ghost cells as a function of spatial order
  switch (spatial_order) {
    case 1:
      nGhostCells = 1;
      break;
    case 2:
      nGhostCells = 1;
      break;
    case 3:
      nGhostCells = 1;
      break;
    case 4:
      nGhostCells = 1;
      break;
    case 5:
      nGhostCells = 1;
      break;
    case 6:
      nGhostCells = 1;
      break;
    default:
      nGhostCells = 1;
  }

  return nGhostCells;
}
