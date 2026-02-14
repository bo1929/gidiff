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

class LLH;

class DIM
{
public:
  DIM(llh_sptr_t llhf, uint64_t en_mers);
  double get_fdt() const { return fdt; }
  double get_sdt() const { return sdt; }
  double fdt_at(uint64_t i) const { return fdc_v[i]; }
  double sdt_at(uint64_t i) const { return sdc_v[i]; }
  void inclusive_scan();
  void optimize_loglikelihood();
  void extract_intervals(uint64_t tau);
  uint64_t expand_intervals(double chisq_th);
  void report_intervals(std::ostream& output_stream, const std::string& identifer);
  void aggregate_mer(sketch_sptr_t sketch, uint32_t rix, enc_t enc_lr, uint64_t i);
  // void skip_mer(uint64_t i);

private:
  llh_sptr_t llhf;
  const uint64_t en_mers;
  const bool opposite;
  const uint32_t hdist_th;
  uint64_t merhit_count = 0;
  uint64_t merna_count = 0;
  uint64_t mermiss_count = 0;
  std::vector<uint64_t> hdisthist_v;
  vec<double> fdc_v;
  vec<double> sdc_v;
  vec<double> fdps_v;
  vec<double> sdps_v;
  vec<double> fdpmax_v;
  vec<double> fdsmin_v;
  vec<interval_t> rintervals_v;
  vec<interval_t> eintervals_v;
  vec<double> chisq_v;
  double fdt = 0;
  double sdt = 0;
  double d_llh = std::numeric_limits<double>::quiet_NaN();
  double v_llh = std::numeric_limits<double>::quiet_NaN();
};

class QIE
{
public:
  QIE(sketch_sptr_t sketch, lshf_sptr_t lshf, qseq_sptr_t qs, params_t params);
  void map_sequences(std::ostream& output_stream);
  void search_mers(const char* cseq, uint64_t len, DIM& or_summary, DIM& rc_summary);

private:
  const sketch_sptr_t sketch;
  const lshf_sptr_t lshf;
  const uint64_t batch_size;
  const uint32_t k;
  const uint32_t h;
  const uint32_t m;
  const double rho;
  const double dist_th;
  const uint32_t hdist_th;
  const uint64_t min_length;
  const double chisq; // 3.841; // 95%
  uint64_t mask_bp;
  uint64_t mask_lr;
  llh_sptr_t llhf;
  uint64_t onmers;
  uint64_t en_mers;
  uint64_t bix;
  vec<std::string> seq_batch;
  vec<std::string> identifer_batch;
};

#endif
