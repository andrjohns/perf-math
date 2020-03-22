#include <stan/math.hpp>
#include <benchmark/benchmark.h>
#include <Eigen/Dense>

namespace stan {
namespace math {

namespace internal {
template <typename Container>
class exp_matrix_vari : public vari {
 public:
  int A_rows_;
  int A_cols_;
  int A_size_;
  double* Ad_;
  vari** variRefA_;
  vari** variRefExp_;

  explicit exp_matrix_vari(const Container& A)
      : vari(0.0),
        A_rows_(A.rows()),
        A_cols_(A.cols()),
        A_size_(A.size()),
        Ad_(ChainableStack::instance_->memalloc_.alloc_array<double>(A_size_)),
        variRefA_(
            ChainableStack::instance_->memalloc_.alloc_array<vari*>(A_size_)),
        variRefExp_(
            ChainableStack::instance_->memalloc_.alloc_array<vari*>(A_size_)) {
    using Eigen::Map;
    Map<matrix_vi>(variRefA_, A_rows_, A_cols_) = A.vi();
    Map<matrix_d> Ad(Ad_, A_rows_, A_cols_);
    Ad = A.val();
    Map<matrix_vi>(variRefExp_, A_rows_, A_cols_).array()
        = Ad.array().exp().unaryExpr(
            [](double x) { return new vari(x, false); });
  }

  virtual void chain() {
    using Eigen::Map;
    Map<matrix_vi> RefExp(variRefExp_, A_rows_, A_cols_);
    Map<matrix_vi>(variRefA_, A_rows_, A_cols_).adj()
        += RefExp.adj().cwiseProduct(RefExp.val());
  }
};

}  // namespace internal

template <typename Container,
          require_container_st<is_container, is_var, Container>...>
inline auto exp_new(const Container& x) {
  return apply_vector_unary<Container>::apply(x, [](const auto& v) {
    using T_plain = plain_type_t<decltype(v)>;
    using T_ref = Eigen::Ref<const T_plain>;

    const T_ref& v_ref = v;
    auto* baseVari = new internal::exp_matrix_vari<T_ref>(v_ref);
    T_plain result(v_ref.rows(), v_ref.cols());
    result.vi() = Eigen::Map<matrix_vi>(baseVari->variRefExp_, v_ref.rows(),
                                        v_ref.cols());

    return result;
  });
}

}  // namespace math
}  // namespace stan

const int R = -1;
const int C = -1;

static void Exp_Old(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using namespace stan::math;
  Matrix<var,R, C> m_d(1000, 1000);
  Matrix<var,R, C> result(1000, 1000);
  m_d = MatrixXd::Random(1000, 1000);
  m_d.adj() = MatrixXd::Random(1000, 1000);


  for (auto _ : state) {
    result = exp(m_d);
  }
}
BENCHMARK(Exp_Old);

static void Exp_New(benchmark::State& state) {
  using Eigen::Matrix;
  using Eigen::MatrixXd;
  using namespace stan::math;
  Matrix<var,R, C> m_d(1000, 1000);
  Matrix<var,R, C> result(1000, 1000);
  m_d = MatrixXd::Random(1000, 1000);
  m_d.adj() = MatrixXd::Random(1000, 1000);


  for (auto _ : state) {
    result = exp_new(m_d);
  }
}
BENCHMARK(Exp_New);

BENCHMARK_MAIN();
