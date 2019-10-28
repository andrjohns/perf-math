#include <stan/math.hpp>
#include <benchmark/benchmark.h>
#include <vector>

template <typename T>
const auto as_eigen(const std::vector<T>& v) {
  return Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(v.data(),
                                                               v.size());
}

static void LogSumExp_Old(benchmark::State& state) {
  std::vector<std::vector<double>> y(500);
  Eigen::VectorXd gen;

  for (auto _ : state) {
    std::vector<double> results(500);
    for(int i = 0; i < 500; i++){
      gen = Eigen::VectorXd::Random(1000);
      y[i] = std::vector<double>(gen.data(), gen.data() + gen.size());
      results[i] = stan::math::log_sum_exp(y[i]);
    }
  }
}
BENCHMARK(LogSumExp_Old);

static void LogSumExp_New(benchmark::State& state) {
  std::vector<std::vector<double>> y(500);
  Eigen::VectorXd gen;

  for (auto _ : state) {
    std::vector<double> results(500);
    for(int i = 0; i < 500; i++){
      gen = Eigen::VectorXd::Random(1000);
      y[i] = std::vector<double>(gen.data(), gen.data() + gen.size());
      results[i] = stan::math::log_sum_exp(as_eigen(y[i]));
    }
  }
}
BENCHMARK(LogSumExp_New);

static void DotSelf_Old(benchmark::State& state) {
  std::vector<std::vector<double>> y(500);
  Eigen::VectorXd gen;

  for (auto _ : state) {
    std::vector<double> results(500);
    for(int i = 0; i < 500; i++){
      gen = Eigen::VectorXd::Random(1000);
      y[i] = std::vector<double>(gen.data(), gen.data() + gen.size());
      results[i] = stan::math::dot_self(y[i]);
    }
  }
}
BENCHMARK(DotSelf_Old);

static void DotSelf_New(benchmark::State& state) {
  std::vector<std::vector<double>> y(500);
  Eigen::VectorXd gen;

  for (auto _ : state) {
    std::vector<double> results(500);
    for(int i = 0; i < 500; i++){
      gen = Eigen::VectorXd::Random(1000);
      y[i] = std::vector<double>(gen.data(), gen.data() + gen.size());
      results[i] = stan::math::dot_self(as_eigen(y[i]));
    }
  }
}
BENCHMARK(DotSelf_New);

static void Sum_Old(benchmark::State& state) {
  std::vector<std::vector<double>> y(500);
  Eigen::VectorXd gen;

  for (auto _ : state) {
    std::vector<double> results(500);
    for(int i = 0; i < 500; i++){
      gen = Eigen::VectorXd::Random(1000);
      y[i] = std::vector<double>(gen.data(), gen.data() + gen.size());
      results[i] = stan::math::sum(y[i]);
    }
  }
}
BENCHMARK(Sum_Old);

static void Sum_New(benchmark::State& state) {
  std::vector<std::vector<double>> y(500);
  Eigen::VectorXd gen;

  for (auto _ : state) {
    std::vector<double> results(500);
    for(int i = 0; i < 500; i++){
      gen = Eigen::VectorXd::Random(1000);
      y[i] = std::vector<double>(gen.data(), gen.data() + gen.size());
      results[i] = stan::math::sum(as_eigen(y[i]));
    }
  }
}
BENCHMARK(Sum_New);

BENCHMARK_MAIN();
