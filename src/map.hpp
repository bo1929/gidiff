#ifndef _MAP_H
#define _MAP_H

#include <unordered_map>
#include <simde/x86/avx512.h>
#include "llh.hpp"
#include "lshf.hpp"
#include "rqseq.hpp"
#include "sketch.hpp"

static constexpr uint64_t NULL_SAMPLES_PER_LENGTH = 200;
static constexpr uint64_t NULL_MIN_SAMPLES = 30;
static constexpr std::array<double, 3> GAMMA_FIT_PROBS = {0.25, 0.50, 0.75};
static constexpr double SUBSAMPLE_FACTOR = 1.0;
static constexpr double GRID_GROWTH = 1.25;

struct qrec_t
{
  uint64_t nbins;
  uint64_t nmers;
  vec<uint64_t> hdisthist_v; // subsampled prefix-sum rows, flattened [(nbins/G + 1) × W]
};

template<typename T>
class QIE;

struct segment_t
{
  uint64_t start;
  uint64_t end;
  double d_s;
  uint8_t mask; // bitmask: bit i set iff threshold i hits this segment
  char sign;    // '<' for positive thresholds, '>' for negative thresholds
};

struct output_record_t
{
  uint64_t bix;
  uint64_t n, a, b;
  uint64_t nbins_s;
  char strand;
  double d_s, d_q;
  double percentile, fold;
  uint8_t mask;
  char sign;
};

template<typename T>
class DIM
{
  static constexpr size_t WIDTH = std::is_same_v<T, double> ? 1 : RWIDTH;

public:
  DIM(llh_sptr_t<T> llhf, uint32_t hdist_th, uint64_t nbins, uint64_t nmers, bool enum_only = true);
  // T get_fdt() const { return fdt; }
  // T get_sdt() const { return sdt; }
  static inline double at(T v, size_t idx);
  void release_accumulators() noexcept;
  void inclusive_scan();
  void extrema_scan();
  void compute_prefhistsum();
  // void skip_mer(uint64_t i); // TODO: Anything better than ignoring?
  void aggregate_mer(uint32_t hdist_min, uint64_t i);
  void extract_intervals_mx(uint64_t tau, size_t idx = 0);
  void extract_intervals_sx(uint64_t tau, size_t idx = 0);
  void expand_intervals(double chisq_th, size_t idx = 0);
  [[nodiscard]] interval_t get_interval(uint64_t i, size_t idx = 0) const;
  T fdc_at(uint64_t i) const { return fdc_v[i]; } // per-bin f'  contribution
  T sdc_at(uint64_t i) const { return sdc_v[i]; } // per-bin f'' contribution
  const vec<uint64_t>& get_hdisthist() const { return hdisthist_v; }
  const vec<segment_t>& get_segments() const { return segments_v; }
  uint64_t get_nbins() const { return nbins; }
  uint64_t get_nmers() const { return nmers; }
  void map_contiguous_segments(uint64_t bin_shift, uint8_t th_bv, char sign);
  double estimate_interval_distance(uint64_t a, uint64_t b, uint64_t bin_shift);
  void extract_histogram(uint64_t a, uint64_t b, uint64_t bin_shift, vec<uint64_t>& v, uint64_t& u, uint64_t& t) const;
  static inline void add_to(T& dest, const T& src)
  {
    if constexpr (std::is_same_v<T, double>) {
      dest += src;
    } else {
      simde__m512d vd = simde_mm512_loadu_pd(dest.data());
      simde__m512d vs = simde_mm512_loadu_pd(src.data());
      vd = simde_mm512_add_pd(vd, vs);
      simde_mm512_storeu_pd(dest.data(), vd);
    }
  }

private:
  const llh_sptr_t<T> llhf;
  const bool enum_only;
  const uint32_t hdist_th;
  const uint64_t nbins; // number of bins
  const uint64_t nmers; // number of k-mers in query (for per-k-mer hdist tracking)
  uint64_t t_q = 0;
  uint64_t u_q = 0;
  // T fdt; // To keep the total in case fw/rc decision is needed.
  // T sdt; // Not sure if this is needeed even for fw/rc decision.
  vec<uint64_t> hdisthist_v; // [(nbins+1) × (hdist_th+1)] row-major, row 0 = zeros, compute_prefhistsum() converts in-place
  vec<T> fdc_v;              // The f' contribution c_i of the k-mer (bin) starting at i
  vec<T> sdc_v;              // The f'' contribution s_i of the k-mer (bin) starting at i
  vec<T> fdps_v;             // C[i] = sum(c_0, ..., c_{i}), C[0] = 0 (length n) (shifted by 1 w.r.t. fdc_v)
  vec<T> sdps_v;             // S[i] = sum(s_0, ..., s_{i}), S[0] = 0 (length n) (shifted by 1 w.r.t. sdc_v)
  vec<T> fdpmax_v;           // H[i] = max(C_1, ..., C_{i}), H_0 = -inf, H_{n+1} = inf (length n+1)
  vec<T> fdsmin_v;           // L[i] = min(C_{i}, ..., C_n), L_0 = inf, L_{n+1}= -inf (length n+1)
  arr<vec<interval_t>, WIDTH> rintervals_v;
  arr<vec<interval_t>, WIDTH> eintervals_v;
  vec<segment_t> segments_v;
  // arr<vec<double>, WIDTH> chisq_v;
};

template<typename T>
class QIE
{
  static constexpr size_t WIDTH = std::is_same_v<T, double> ? 1 : RWIDTH;

public:
  QIE(sketch_sptr_t sketch, lshf_sptr_t lshf, const vec<str>& seq_batch, const vec<str>& qid_batch, params_t<T> params);
  void map_sequences(std::ostream& sout, const str& rid);

private:
  static inline double at(T v, size_t idx);
  void search_mers(const char* cseq, uint64_t len, DIM<T>& dim_fw, DIM<T>& dim_rc);
  void report_intervals(std::ostream& sout, const str& rid, DIM<T>& dim, bool rc, size_t idx = 0);
  void collect_segments(const DIM<T>& dim, bool rc, double d_q);
  void store_qrec(const DIM<T>& dim);
  void build_length_grid();
  void fit_gamma_significance();
  void emit_segments(std::ostream& sout, const str& rid) const;
  double compute_mle_dist(const vec<uint64_t>& v, uint64_t u);

  const sketch_sptr_t sketch;
  const lshf_sptr_t lshf;
  const uint64_t batch_size;
  const uint32_t k;
  const uint32_t h;
  const uint32_t m;
  const params_t<T> params;
  const uint64_t tau;
  const uint64_t btau;
  const uint64_t bin_shift;
  const uint64_t bin_size;
  const double chisq; // 3.841; // 95%
  uint64_t mask_bp;
  uint64_t mask_lr;
  uint64_t onmers; // number of observed k-mers in current query (e.g., due to Ns)
  uint64_t enmers; // number of expected k-mers in current query (= len - k + 1)
  uint64_t nbins;  // number of bins  (= ceil(enmers / bin_len))
  uint64_t bix;    // Index of the current query in the this batch
  llh_sptr_t<T> llhf;

  const vec<str>& seq_batch;
  const vec<str>& qid_batch;

  double d_acc = std::numeric_limits<double>::quiet_NaN();
  uint64_t u_acc = 0;
  vec<uint64_t> v_acc;
  uint64_t gstride = 1;
  vec<uint64_t> length_grid;
  vec<qrec_t> qrecs;
  vec<output_record_t> output_records;
};

#define WRITE_CINTERVAL(qid, n, a, b, strand, rid, dist_th)                                                                 \
  qid << '\t' << n << '\t' << a << '\t' << b << '\t' << strand << '\t' << rid << '\t' << dist_th

#define WRITE_SEGMENT(qid, n, a, b, strand, rid, dist, mask, sign, d_q, d_acc, pctl, fold)                                  \
  qid << '\t' << n << '\t' << a << '\t' << b << '\t' << strand << '\t' << rid << '\t' << dist << '\t' << mask << '\t'       \
      << sign << '\t' << d_q << '\t' << d_acc << '\t' << pctl << '\t' << fold

#endif
