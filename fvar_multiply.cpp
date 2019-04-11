#define EIGEN_MATRIXBASE_PLUGIN "matrix_addons.h"
#include <stan/math/fwd/mat.hpp>
#include <benchmark/benchmark.h>

static void fvarDbl_MatMultiply_Old(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using stan::math::matrix_fd;
  using stan::math::multiply;
  
  matrix_fd mat1(1000,1000);
  matrix_fd mat2(1000,1000);

  mat1.val_() = MatrixXd::Random(1000, 1000);
  mat2.val_() = MatrixXd::Random(1000, 1000);
  mat1.d_() = MatrixXd::Random(1000, 1000);
  mat2.d_() = MatrixXd::Random(1000, 1000);

  for (auto _ : state) {
    matrix_fd out = multiply(mat1,mat2);
  }
}
BENCHMARK(fvarDbl_MatMultiply_Old);

static void fvarDbl_MatMultiply_New(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using stan::math::matrix_fd;
  using stan::math::matrix_d;
  using stan::math::multiply;
  
  matrix_fd mat1(1000,1000);
  matrix_fd mat2(1000,1000);

  mat1.val_() = MatrixXd::Random(1000, 1000);
  mat2.val_() = MatrixXd::Random(1000, 1000);
  mat1.d_() = MatrixXd::Random(1000, 1000);
  mat2.d_() = MatrixXd::Random(1000, 1000);

  for (auto _ : state) {
    matrix_fd out(1000, 1000);

    matrix_d m1val = mat1.val_();
    matrix_d m2val = mat2.val_();

    out.val_() = m1val * m2val;
    out.d_() = m1val * mat2.d_() + mat1.d_() * m2val;
  }
}
BENCHMARK(fvarDbl_MatMultiply_New);

static void fvarfvarDbl_MatMultiply_Old(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using stan::math::matrix_ffd;
  using stan::math::multiply;
  
  matrix_ffd mat1(1000,1000);
  matrix_ffd mat2(1000,1000);

  mat1.val_().val_() = MatrixXd::Random(1000, 1000);
  mat2.val_().val_() = MatrixXd::Random(1000, 1000);
  mat1.d_().val_() = MatrixXd::Random(1000, 1000);
  mat2.d_().val_() = MatrixXd::Random(1000, 1000);

  for (auto _ : state) {
    matrix_ffd out = multiply(mat1,mat2);
  }
}
BENCHMARK(fvarfvarDbl_MatMultiply_Old);

static void fvarfvarDbl_MatMultiply_New(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using stan::math::matrix_ffd;
  using stan::math::matrix_d;
  using stan::math::multiply;
  
  matrix_ffd mat1(1000,1000);
  matrix_ffd mat2(1000,1000);

  mat1.val_().val_() = MatrixXd::Random(1000, 1000);
  mat2.val_().val_() = MatrixXd::Random(1000, 1000);
  mat1.d_().val_() = MatrixXd::Random(1000, 1000);
  mat2.d_().val_() = MatrixXd::Random(1000, 1000);

  for (auto _ : state) {
    matrix_ffd out(1000, 1000);

    matrix_d m1val = mat1.val_().val_();
    matrix_d m2val = mat2.val_().val_();

    out.val_().val_() = m1val * m2val;
    out.d_().val_() = m1val * mat2.d_().val_() + mat1.d_().val_() * m2val;
  }
}
BENCHMARK(fvarfvarDbl_MatMultiply_New);

BENCHMARK_MAIN();