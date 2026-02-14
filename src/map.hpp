#ifndef _MAP_H
#define _MAP_H

#include "llh.hpp"
#include "lshf.hpp"
#include "rqseq.hpp"
#include "sketch.hpp"
#include "hm.hpp"
#include "enc.hpp"
#include "types.hpp"
#include "exthash.hpp"

#define SIMDE_ENABLE_NATIVE_ALIASES
#include <simde/x86/avx512.h>

class LLH;

template<size_t N>
class DIM
{
  static_assert(N == 1 || N == 8, "DIM supports only N=1 or N=8");
  
public:
  DIM(const std::array<llh_sptr_t, N>& llhf_arr, uint64_t en_mers);
  
  std::array<double, N> get_fdt() const { return fdt; }
  std::array<double, N> get_sdt() const { return sdt; }
  
  void inclusive_scan();
  void optimize_loglikelihood();
  void extract_intervals(uint64_t tau, size_t idx);
  uint64_t expand_intervals(double chisq_th, size_t idx);
  void report_intervals(std::ostream& output_stream, const std::string& identifer, size_t idx);
  void aggregate_mer(sketch_sptr_t sketch, uint32_t rix, enc_t enc_lr, uint64_t i);

private:
  std::array<llh_sptr_t, N> llhf_arr;
  const uint64_t en_mers;
  std::array<bool, N> opposite;
  std::array<uint32_t, N> hdist_th;
  
  std::array<uint64_t, N> merhit_count;
  std::array<uint64_t, N> mermiss_count;
  std::array<std::vector<uint64_t>, N> hdisthist_v;
  
  std::vector<std::array<double, N>> fdc_v;
  std::vector<std::array<double, N>> sdc_v;
  std::vector<std::array<double, N>> fdps_v;
  std::vector<std::array<double, N>> sdps_v;
  std::vector<std::array<double, N>> fdpmax_v;
  std::vector<std::array<double, N>> fdsmin_v;
  
  std::array<vec<interval_t>, N> rintervals_v;
  std::array<vec<interval_t>, N> eintervals_v;
  std::array<vec<double>, N> chisq_v;
  
  std::array<double, N> fdt;
  std::array<double, N> sdt;
  std::array<double, N> d_llh;
  std::array<double, N> v_llh;
};

class QIE
{
public:
  QIE(sketch_sptr_t sketch, lshf_sptr_t lshf, qseq_sptr_t qs, const std::vector<params_t>& params_vec);
  void map_sequences(std::ostream& output_stream);

private:
  template<size_t N>
  void map_sequences_impl(std::ostream& output_stream);
  
  template<size_t N>
  void search_mers(const char* cseq, uint64_t len, DIM<N>& dim_or, DIM<N>& dim_rc);
  
  const sketch_sptr_t sketch;
  const lshf_sptr_t lshf;
  const uint64_t batch_size;
  const uint32_t k;
  const uint32_t h;
  const uint32_t m;
  const double rho;
  const uint64_t mask_bp;
  const uint64_t mask_lr;
  
  std::vector<params_t> params_vec;
  const size_t num_params;
  
  uint64_t onmers;
  uint64_t en_mers;
  uint64_t bix;
  vec<std::string> seq_batch;
  vec<std::string> identifer_batch;
};

#endif
