// Port of reference/tinynav/core/perception_node.py's [ISAM Processing] block —
// the no-GTSAM counterpart of refine.cpp: the component calls
// make_gtsam_refine() at construction and keeps the v1 chained-PnP pipeline
// when it gets nullptr (see refine.hpp and CMakeLists's tinynav_gtsam).
#include "tinynav_cpp/gtsam/refine.hpp"

namespace tinynav::gtsam
{

std::shared_ptr<Refine> make_gtsam_refine()
{
  return nullptr;
}

}  // namespace tinynav::gtsam
