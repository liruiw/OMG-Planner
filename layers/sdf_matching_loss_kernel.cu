#include <ATen/ATen.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <vector>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

inline __device__ __host__ float lerp(float a, float b, float t)
{
    return a + t*(b-a);
}

__device__ __host__ float3 operator+(const float3 &a, const float3 &b)
{
  return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}

__device__ __host__ float3 operator-(const float3 &a, const float3 &b)
{
  return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}

template <typename Dtype>
inline __device__ __host__ const Dtype & getValue(const int3 & v, const int3 & dim, const Dtype* sdf_grids)
{
  return sdf_grids[v.x * dim.y * dim.z + v.y * dim.z + v.z];
}

template <typename Dtype>
inline __device__ __host__ Dtype getValueInterpolated(const float3 & pGrid, const int3 & dim, const Dtype* sdf_grids)
{
  const int x0 = (int)(pGrid.x - 0.5); const float fx = (pGrid.x - 0.5) - x0;
  const int y0 = (int)(pGrid.y - 0.5); const float fy = (pGrid.y - 0.5) - y0;
  const int z0 = (int)(pGrid.z - 0.5); const float fz = (pGrid.z - 0.5) - z0;

  const int x1 = x0 + 1;
  const int y1 = y0 + 1;
  const int z1 = z0 + 1;

  if ( !(x0 >= 0 && x1 < dim.x && y0 >= 0 && y1 < dim.y && z0 >=0 && z1 < dim.z) )
    return 1.0;

  const float dx00 = lerp( getValue(make_int3(x0,y0,z0), dim, sdf_grids), getValue(make_int3(x1,y0,z0), dim, sdf_grids), fx);
  const float dx01 = lerp( getValue(make_int3(x0,y0,z1), dim, sdf_grids), getValue(make_int3(x1,y0,z1), dim, sdf_grids), fx);
  const float dx10 = lerp( getValue(make_int3(x0,y1,z0), dim, sdf_grids), getValue(make_int3(x1,y1,z0), dim, sdf_grids), fx);
  const float dx11 = lerp( getValue(make_int3(x0,y1,z1), dim, sdf_grids), getValue(make_int3(x1,y1,z1), dim, sdf_grids), fx);

  const float dxy0 = lerp( dx00, dx10, fy );
  const float dxy1 = lerp( dx01, dx11, fy );
  float dxyz = lerp( dxy0, dxy1, fz );

  // penalize inside objects
  // if (dxyz < 0)
  //  dxyz *= 10;

  return dxyz;
}

template <typename Dtype>
inline __device__ __host__ float3 getGradientInterpolated(const float3 & pGrid, const int3 & dim, const Dtype* sdf_grids, float delta)
{
  const float3 delta_x = make_float3(1,0,0);
  const float3 delta_y = make_float3(0,1,0);
  const float3 delta_z = make_float3(0,0,1);

  Dtype f_px = getValueInterpolated(pGrid + delta_x, dim, sdf_grids);
  Dtype f_py = getValueInterpolated(pGrid + delta_y, dim, sdf_grids);
  Dtype f_pz = getValueInterpolated(pGrid + delta_z, dim, sdf_grids);

  Dtype f_mx = getValueInterpolated(pGrid - delta_x, dim, sdf_grids);
  Dtype f_my = getValueInterpolated(pGrid - delta_y, dim, sdf_grids);
  Dtype f_mz = getValueInterpolated(pGrid - delta_z, dim, sdf_grids);

  float3 grad;
  grad.x = 0.5*(f_px - f_mx) / delta;
  grad.y = 0.5*(f_py - f_my) / delta;
  grad.z = 0.5*(f_pz - f_mz) / delta;
  return grad;
}


/*******************************************/
/* nthreads: num_points x num_objects      */
/* pose_init: num_objects x 4 x 4          */
/* sdf_grid: num_objects x c x h x w       */
/* sdf_limits: num_objects x 10             */
/* points: num_points x 3                  */
/*******************************************/
template <typename Dtype>
__global__ void SDFdistanceForward(const int nthreads, const Dtype* pose_init,
    const Dtype* sdf_grids, const Dtype* sdf_limits, const Dtype* points, 
    const Dtype* epsilons, const Dtype* padding_scales, const Dtype* clearances, const Dtype* disables,
    const int num_points, const int num_objects, Dtype* potentials, Dtype* collides, Dtype* potential_grads) 
{
  typedef Sophus::SE3<Dtype> SE3;
  typedef Sophus::SO3<Dtype> SO3;
  typedef Eigen::Matrix<Dtype,3,1,Eigen::DontAlign> Vec3;
  typedef Eigen::Matrix<Dtype,3,3,Eigen::DontAlign> Mat3;

  // index is the index of point
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    // batch index
    int pindex = index / num_objects;
    int obj_index = index % num_objects;
    int start_index;

    if (disables[obj_index] > 0)
        continue;

    // convert initial pose
    Eigen::Matrix<Dtype,4,4> initialPose;
    start_index = 16 * obj_index;
    initialPose << pose_init[start_index + 0], pose_init[start_index + 1], pose_init[start_index + 2], pose_init[start_index + 3],
                   pose_init[start_index + 4], pose_init[start_index + 5], pose_init[start_index + 6], pose_init[start_index + 7],
                   pose_init[start_index + 8], pose_init[start_index + 9], pose_init[start_index + 10], pose_init[start_index + 11],
                   pose_init[start_index + 12], pose_init[start_index + 13], pose_init[start_index + 14], pose_init[start_index + 15];
    SE3 initialPoseMatrix = SE3(initialPose);
    Mat3 rotationMatrix = initialPoseMatrix.so3().matrix();

    // convert point
    Vec3 point;
    point << points[3 * pindex], points[3 * pindex + 1], points[3 * pindex + 2];

    // transform the point
    const Vec3 updatedPoint = initialPoseMatrix * point;

    // obtain sdf value
    start_index = 10 * obj_index;
    int d0 = int(sdf_limits[start_index + 6]);
    int d1 = int(sdf_limits[start_index + 7]);
    int d2 = int(sdf_limits[start_index + 8]);
    float px = (updatedPoint(0) - sdf_limits[start_index + 0]) / (sdf_limits[start_index + 3] - sdf_limits[start_index + 0]) * d0;
    float py = (updatedPoint(1) - sdf_limits[start_index + 1]) / (sdf_limits[start_index + 4] - sdf_limits[start_index + 1]) * d1;
    float pz = (updatedPoint(2) - sdf_limits[start_index + 2]) / (sdf_limits[start_index + 5] - sdf_limits[start_index + 2]) * d2;
    float delta = sdf_limits[start_index + 9];

    float3 pGrid = make_float3(px, py, pz);
    int3 dim = make_int3(d0, d1, d2);
    Dtype value = getValueInterpolated(pGrid, dim, sdf_grids + obj_index * d0 * d1 * d2);

    // collision
    if (value < clearances[obj_index])
      collides[index] = 1;

    // compute gradient
    float3 grad = getGradientInterpolated(pGrid, dim, sdf_grids + obj_index * d0 * d1 * d2, delta);
    Dtype epsilon = epsilons[obj_index];
    Dtype padding_scale = padding_scales[obj_index];
    Vec3 vgrad;
    if (value <= 0)
    {
      potentials[index] = -value + 0.5 * epsilon;
      vgrad(0) = -grad.x;
      vgrad(1) = -grad.y;
      vgrad(2) = -grad.z;
    }
    else if (value > 0 && value <= epsilon)
    {
      potentials[index] = 1 / (2 * epsilon) * (value - epsilon) * (value - epsilon) * padding_scale;
      vgrad(0) = 1 / epsilon * grad.x * (value - epsilon) * padding_scale;
      vgrad(1) = 1 / epsilon * grad.y * (value - epsilon) * padding_scale;
      vgrad(2) = 1 / epsilon * grad.z * (value - epsilon) * padding_scale;
    }
    else
      continue;

    // map to robot coordinate
    const Vec3 updatedGrad = rotationMatrix.transpose() * vgrad;
    potential_grads[3 * index + 0] = updatedGrad(0);
    potential_grads[3 * index + 1] = updatedGrad(1);
    potential_grads[3 * index + 2] = updatedGrad(2);
  }
}

/* diffs: num_points x num_objects x num_channels */
/* bottom_diff: num_points x num_channels */
template <typename Dtype>
__global__ void sum_gradients(const int nthreads, const Dtype* diffs, const int num_objects, const int num_channels, Dtype* bottom_diff) 
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) 
  {
    int p = index / (num_objects * num_channels);
    int n = index % (num_objects * num_channels);
    int c = n % num_channels;
    atomicAdd(bottom_diff + p * num_channels + c, diffs[index]);
  }
}


/*******************************************/
/* pose_init: num_objects x 4 x 4          */
/* sdf_grid: num_objects x c x h x w       */
/* sdf_limits: num_objects x 9             */
/* points: num_points x 3                  */
/*******************************************/
std::vector<at::Tensor> sdf_loss_cuda_forward(
    at::Tensor pose_init,
    at::Tensor sdf_grids,
    at::Tensor sdf_limits,
    at::Tensor points,
    at::Tensor epsilons,
    at::Tensor padding_scales,
    at::Tensor clearances,
    at::Tensor disables) 
{
  // run kernels
  cudaError_t err;
  const int kThreadsPerBlock = 512;
  const int num_channels = 3;
  int output_size;

  // sizes
  const int num_objects = pose_init.size(0);
  const int num_points = points.size(0);

  // outputs
  auto potentials = at::zeros({num_points, num_objects}, points.options());
  auto collides = at::zeros({num_points, num_objects}, points.options());
  auto potential_grads = at::zeros({num_points, num_objects, num_channels}, points.options());

  auto top_potentials = at::zeros({num_points}, points.options());
  auto top_collides = at::zeros({num_points}, points.options());
  auto top_potential_grads = at::zeros({num_points, num_channels}, points.options());

  // compute the potentials and gradients
  output_size = num_points * num_objects;
  SDFdistanceForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      output_size, pose_init.data<float>(), sdf_grids.data<float>(), sdf_limits.data<float>(),
      points.data<float>(), epsilons.data<float>(), padding_scales.data<float>(), clearances.data<float>(), disables.data<float>(),
      num_points, num_objects, potentials.data<float>(), collides.data<float>(), potential_grads.data<float>());
  cudaDeviceSynchronize();

  err = cudaGetLastError();
  if(cudaSuccess != err)
  {
    fprintf( stderr, "cudaCheckError() failed: %s\n", cudaGetErrorString( err ) );
    exit( -1 );
  }

  // sum the potentials
  output_size = num_points * num_objects;
  sum_gradients<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      output_size, potentials.data<float>(), num_objects, 1, top_potentials.data<float>());

  sum_gradients<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      output_size, collides.data<float>(), num_objects, 1, top_collides.data<float>());

  output_size = num_points * num_objects * num_channels;
  sum_gradients<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock, kThreadsPerBlock>>>(
      output_size, potential_grads.data<float>(), num_objects, num_channels, top_potential_grads.data<float>());
  cudaDeviceSynchronize();

  return {top_potentials, top_potential_grads, top_collides};
}
