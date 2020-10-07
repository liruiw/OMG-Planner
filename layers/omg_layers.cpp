#include <torch/torch.h>

#include <vector>

#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda())
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous())
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


/************************************************************
 sdf matching loss layer
*************************************************************/

std::vector<at::Tensor> sdf_loss_cuda_forward(
    at::Tensor pose_init,
    at::Tensor sdf_grids,
    at::Tensor sdf_limits,
    at::Tensor points,
    at::Tensor epsilons,
    at::Tensor padding_scales,
    at::Tensor clearances,
    at::Tensor disables);

std::vector<at::Tensor> sdf_loss_forward(
    at::Tensor pose_init,
    at::Tensor sdf_grids,
    at::Tensor sdf_limits,
    at::Tensor points,
    at::Tensor epsilons,
    at::Tensor padding_scales,
    at::Tensor clearances,
    at::Tensor disables)
{
  CHECK_INPUT(pose_init);
  CHECK_INPUT(sdf_grids);
  CHECK_INPUT(sdf_limits);
  CHECK_INPUT(points);
  CHECK_INPUT(epsilons);
  CHECK_INPUT(padding_scales);
  CHECK_INPUT(clearances);
  CHECK_INPUT(disables);

  return sdf_loss_cuda_forward(pose_init, sdf_grids, sdf_limits, points, epsilons, padding_scales, clearances, disables);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sdf_loss_forward", &sdf_loss_forward, "SDF loss forward (CUDA)");
}
