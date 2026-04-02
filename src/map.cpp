#include "map.hpp"
#include <boost/math/tools/minima.hpp>

#define EPS 1e-10
#define MAXHD 32

// TODO: Further optimize, and polish.
// TODO: A better output format with a header? For both modes...
// TODO: If we will compute the distances, perhaps have an option to merge all overlapping intervals per distance.
// TODO: Merge aggressively as per above reasons.
// TODO: Something interesting with the second derivatives? Is the minimum second-derivative interval interesting?

template<typename T>
DIM<T>::DIM(llh_sptr_t<T> llhf, uint32_t hdist_th, uint64_t nbins, uint64_t nmers, bool enum_only)
  : llhf(llhf)
  , hdist_th(hdist_th)
  , nbins(nbins)
  , nmers(nmers)
  , enum_only(enum_only)
{
  fdc_v.resize(nbins); // Alternative?: fdc_v.reserve(nbins);
  sdc_v.resize(nbins); // Alternative?: sdc_v.reserve(nbins);
  if (!enum_only) {
    // Index 0 = zeros (sentinel);
    // aggregate_mer writes into rows from 1 to nbins;
    // compute_prefhistsum() converts in-place
    hdisthist_v.assign((nbins + 1) * (hdist_th + 1), 0);
  }

  // fdps_v.reserve(nbins + 1);
  // sdps_v.reserve(nbins + 1);

  // This is for deciding between fw and rc.
  // if constexpr (std::is_same_v<T, double>) {
  //   fdt = 0.0;
  //   sdt = 0.0;
  // } else {
  //   fdt.fill(0.0);
  //   sdt.fill(0.0);
  // }
}

template<typename T>
inline double DIM<T>::at(T v, const size_t idx)
{
  if constexpr (std::is_same_v<T, double>) {
    return v;
  } else {
    return v[idx];
  }
}

template<typename T>
void DIM<T>::aggregate_mer(uint32_t hdist_min, uint64_t i)
{
  // i is the bin index; multiple k-mers in the same bin accumulate here.
  if (hdist_min <= hdist_th) {
    merhit_count++;
    if (!enum_only) hdisthist_v[(i + 1) * (hdist_th + 1) + hdist_min]++;
    add_to(sdc_v[i], llhf->get_sdc(hdist_min));
    add_to(fdc_v[i], llhf->get_fdc(hdist_min));
  } else {
    mermiss_count++;
    add_to(sdc_v[i], llhf->get_sdc());
    add_to(fdc_v[i], llhf->get_fdc());
  }

  // This is for deciding between fw and rc.
  // Not needed if we want to find intervals for both.
  // if constexpr (std::is_same_v<T, double>) {
  //   // fdt += fdc_v.back();
  //   // sdt += sdc_v.back();
  //   fdt += fdc_v[i];
  //   sdt += sdc_v[i];
  // } else {
  //   simde__m512d vfdt = simde_mm512_load_pd(fdt.data());
  //   simde__m512d vsdt = simde_mm512_load_pd(sdt.data());
  //   simde__m512d fdc = simde_mm512_load_pd(fdc_v[i].data());
  //   simde__m512d sdc = simde_mm512_load_pd(sdc_v[i].data());
  //   vfdt = simde_mm512_add_pd(vfdt, fdc);
  //   vsdt = simde_mm512_add_pd(vsdt, sdc);
  //   simde_mm512_store_pd(fdt.data(), vfdt);
  //   simde_mm512_store_pd(sdt.data(), vsdt);
  // }
}

template<typename T>
void DIM<T>::compute_prefhistsum()
{
  if (enum_only) return;
  const uint32_t W = hdist_th + 1;
  // Row 0 is zeros; add each row to the previous in-place to get prefix sums
  for (uint64_t i = 0; i < nbins; ++i) {
    for (uint32_t d = 0; d < W; ++d) {
      hdisthist_v[(i + 1) * W + d] += hdisthist_v[i * W + d];
    }
  }
}

template<typename T>
void DIM<T>::release_accumulators() noexcept
{
  fdc_v.clear();
  fdc_v.shrink_to_fit();
  sdc_v.clear();
  sdc_v.shrink_to_fit();
}

template<typename T>
void DIM<T>::extrema_scan(const uint64_t tau, const size_t idx)
{
  const uint64_t s = nbins + 1;
  fdpmax_v.resize(s + 1);
  fdsmin_v.resize(s + 1);

  if constexpr (std::is_same_v<T, double>) {
    fdpmax_v[0] = -std::numeric_limits<double>::max();
    fdsmin_v[0] = std::numeric_limits<double>::max();
    std::inclusive_scan(
      fdps_v.begin() + 1, fdps_v.end(), fdpmax_v.begin() + 1, [](double a, double b) { return std::max(a, b); });
    std::inclusive_scan(
      fdps_v.rbegin(), fdps_v.rend() - 1, fdsmin_v.rbegin() + 1, [](double a, double b) { return std::min(a, b); });
    fdpmax_v[s] = std::numeric_limits<double>::max();
    fdsmin_v[s] = -std::numeric_limits<double>::max();
  } else {
    fdpmax_v[0].fill(-std::numeric_limits<double>::max());
    fdsmin_v[0].fill(std::numeric_limits<double>::max());
    simde__m512d fdpmax_acc = simde_mm512_loadu_pd(fdpmax_v[0].data());
    simde__m512d fdsmin_acc = simde_mm512_loadu_pd(fdsmin_v[0].data());
    for (uint64_t i = 1; i < s; ++i) {
      const simde__m512d fdps_front = simde_mm512_loadu_pd(fdps_v[i].data());
      const simde__m512d fdps_back = simde_mm512_loadu_pd(fdps_v[s - i].data());
      fdpmax_acc = simde_mm512_max_pd(fdpmax_acc, fdps_front);
      fdsmin_acc = simde_mm512_min_pd(fdsmin_acc, fdps_back);
      simde_mm512_storeu_pd(fdpmax_v[i].data(), fdpmax_acc);
      simde_mm512_storeu_pd(fdsmin_v[s - i].data(), fdsmin_acc);
    }
    fdpmax_v[s].fill(std::numeric_limits<double>::max());
    fdsmin_v[s].fill(-std::numeric_limits<double>::max());
  }
}

template<typename T>
void DIM<T>::extract_intervals_mx(const uint64_t tau, const size_t idx)
{
  uint64_t b_curr = 1;
  uint64_t b_prev = std::numeric_limits<uint64_t>::max(); // no interval yet

  for (uint64_t a = 1; a <= nbins; ++a) {
    const double fdpmax_a = at(fdpmax_v[a - 1], idx);
    const double fdps_a = at(fdps_v[a], idx);

    if (fdpmax_a >= fdps_a) {
      continue; // a is a prefix maxima of the prefix sum
    }

    while ((b_curr + 1) <= nbins && (at(fdsmin_v[b_curr + 1], idx) < fdps_a)) {
      ++b_curr; // Right maximal
    } // We increment b_curr to the last b with fdsmin_v[b] < fdps_a
    // There is no other a*>a where b*<b, so no valid (a, b) is missed

    const uint64_t b_star = b_curr;
    if (b_star < (a + tau)) {
      continue; // No valid right endpoint in [a+tau, n]
    }
    if (at(fdps_v[b_star], idx) >= fdps_a) {
      continue; // Negative-sum check
    }
    // if (__builtin_expect(b_star == b_prev, 0)) {
    if (b_star == b_prev) {
      continue; // An earlier (leftmost) a already claimed this b*
    }

    if (at(fdps_v[b_star], idx) >= fdpmax_a) { // Left maximal
      rintervals_v[idx].emplace_back(a, b_star);
      b_prev = b_star;
    }
  }
}

template<typename T>
void DIM<T>::extract_intervals_sx(const uint64_t tau, const size_t idx)
{
  // O(n + k) where k = number of suffix minimum positions of fdps_v (k << n).
  // Every valid right endpoint b* is a suffix minimum of fdps_v.
  // Suffix minimum values are strictly increasing left-to-right,
  // so the pointer into the list is monotone across record highs which is O(k) total.
  struct sufmin_t
  {
    uint64_t pos;
    double val;
  };
  vec<sufmin_t> smins;
  {
    double cur_min = std::numeric_limits<double>::max();
    for (uint64_t j = nbins; j >= 1; --j) {
      const double v = at(fdps_v[j], idx);
      if (v < cur_min) {
        cur_min = v;
        smins.push_back({j, v});
      }
    }
    std::reverse(smins.begin(), smins.end());
  }
  if (smins.empty()) return;

  size_t smin_ptr = 0;
  double running_max = -std::numeric_limits<double>::max();
  uint64_t b_prev = std::numeric_limits<uint64_t>::max();

  for (uint64_t a = 1; a <= nbins; ++a) {
    const double fdps_a = at(fdps_v[a], idx);
    const double fdpmax_a = running_max;

    const double fdps_prev = at(fdps_v[a - 1], idx);
    if (fdps_prev > running_max) running_max = fdps_prev;

    if (fdpmax_a >= fdps_a) continue; // a not a record high

    // Advance smin_ptr to the last suffix minimum with val < fdps_a.
    while (smin_ptr + 1 < smins.size() && smins[smin_ptr + 1].val < fdps_a) {
      ++smin_ptr;
    }
    if (smins[smin_ptr].val >= fdps_a) continue;

    const uint64_t b_star = smins[smin_ptr].pos;
    const double fdps_bstar = smins[smin_ptr].val;

    if (b_star < a + tau) continue;
    if (b_star == b_prev) continue;

    if (fdps_bstar >= fdpmax_a) { // Left maximal
      rintervals_v[idx].emplace_back(a, b_star);
      b_prev = b_star;
    }
  }
}

template<typename T>
void DIM<T>::inclusive_scan()
{
  assert(nbins > 0);
  const uint64_t s = nbins + 1;

  fdps_v.resize(s);
  sdps_v.resize(s);

  if constexpr (std::is_same_v<T, double>) {
    fdps_v[0] = 0.0;
    sdps_v[0] = 0.0;
    for (uint64_t i = 1; i < s; ++i) {
      fdps_v[i] = fdps_v[i - 1] + fdc_v[i - 1];
      sdps_v[i] = sdps_v[i - 1] + sdc_v[i - 1];
    }
  } else {
    fdps_v[0].fill(0.0);
    sdps_v[0].fill(0.0);
    simde__m512d fdps_acc = simde_mm512_setzero_pd();
    simde__m512d sdps_acc = simde_mm512_setzero_pd();
    for (uint64_t i = 1; i < s; ++i) {
      const simde__m512d fdc = simde_mm512_loadu_pd(fdc_v[i - 1].data());
      const simde__m512d sdc = simde_mm512_loadu_pd(sdc_v[i - 1].data());
      fdps_acc = simde_mm512_add_pd(fdps_acc, fdc);
      sdps_acc = simde_mm512_add_pd(sdps_acc, sdc);
      simde_mm512_storeu_pd(fdps_v[i].data(), fdps_acc);
      simde_mm512_storeu_pd(sdps_v[i].data(), sdps_acc);
    }
  }
}

template<typename T>
uint64_t DIM<T>::expand_intervals(const double chisq_th, const size_t idx)
{
  if (rintervals_v[idx].empty()) {
    return 0;
  }

  double fdiff, sdiff, chisq_val;
  uint64_t a, ap, b, bp;
  ap = rintervals_v[idx][0].first;
  bp = rintervals_v[idx][0].second;

  for (uint64_t i = 1; i < rintervals_v[idx].size(); ++i) {
    a = rintervals_v[idx][i].first;
    b = rintervals_v[idx][i].second;
    fdiff = at(fdps_v[b], idx) - at(fdps_v[ap], idx);
    sdiff = at(sdps_v[ap], idx) - at(sdps_v[b], idx);
    // chisq_val = (sdiff > 0.0) ? (fdiff * fdiff) / sdiff : std::numeric_limits<double>::infinity();
    chisq_val = (fdiff * fdiff) / (sdiff + EPS);

    if ((chisq_val < chisq_th) && (a < bp)) {
      a = ap;
    } else {
      eintervals_v[idx].emplace_back(ap, bp);
      // chisq_v[idx].push_back(chisq_val);
    }

    ap = a;
    bp = b;
  }

  fdiff = at(fdps_v[bp], idx) - at(fdps_v[ap], idx);
  sdiff = at(sdps_v[ap], idx) - at(sdps_v[bp], idx);
  // chisq_val = (sdiff > 0.0) ? (fdiff * fdiff) / sdiff : std::numeric_limits<double>::infinity();
  chisq_val = (fdiff * fdiff) / (sdiff + EPS);
  eintervals_v[idx].emplace_back(ap, bp);
  // chisq_v[idx].push_back(chisq_val);
  rintervals_v[idx].clear();
  rintervals_v[idx].shrink_to_fit();
  return eintervals_v[idx].size();
}

template<typename T>
interval_t DIM<T>::get_interval(uint64_t i, size_t idx) const
{
  if ((idx < eintervals_v.size()) && (i < eintervals_v[idx].size())) {
    return eintervals_v[idx][i];
  } else {
    return {nbins, nbins};
  }
}

template<typename T>
QIE<T>::QIE(sketch_sptr_t sketch, lshf_sptr_t lshf, const vec<str>& seq_batch, const vec<str>& qid_batch, params_t<T> params)
  : sketch(sketch)
  , lshf(lshf)
  , k(lshf->get_k())
  , h(lshf->get_h())
  , m(lshf->get_m())
  , batch_size(seq_batch.size())
  , params(params)
  , min_length(params.min_length)
  , bin_shift(params.bin_shift)
  , bin_size(1 << params.bin_shift)
  , chisq(params.chisq)
  , seq_batch(seq_batch)
  , qid_batch(qid_batch)
{
  llhf = std::make_shared<LLH<T>>(k, h, sketch->get_rho(), params.hdist_th, params.dist_th);
  const uint64_t u64m = std::numeric_limits<uint64_t>::max();
  mask_lr = ((u64m >> (64 - k)) << 32) + ((u64m << 32) >> (64 - k));
  mask_bp = u64m >> ((32 - k) * 2);
}

template<typename T>
void QIE<T>::map_sequences(std::ostream& sout, const str& rid)
{
  for (bix = 0; bix < batch_size; ++bix) {
    const char* cseq = seq_batch[bix].data();
    const uint64_t len = seq_batch[bix].size();
    onmers = 0;
    if (len < static_cast<uint64_t>(k)) {
      // TODO: Do we want to have a more verbose mode with this reported?
      continue;
    }
    enmers = len - k + 1;
    nbins = (enmers + bin_size - 1) >> bin_shift;

    DIM<T> dim_fw(llhf, params.hdist_th, nbins, enmers, params.enum_only);
    DIM<T> dim_rc(llhf, params.hdist_th, nbins, enmers, params.enum_only);
    search_mers(cseq, len, dim_fw, dim_rc);

    // Convert min_length (in base-pairs / k-mer units) to bin units for tau
    const uint64_t tau_bins = (min_length + bin_size - 1) >> bin_shift;

    // Extract intervals for all thresholds
    dim_fw.inclusive_scan();
    // dim_fw.release_accumulators();
    dim_rc.inclusive_scan();
    // dim_rc.release_accumulators();
    if constexpr (std::is_same_v<T, double>) {
      dim_fw.extract_intervals_sx(std::min(tau_bins, nbins) - 1);
      dim_rc.extract_intervals_sx(std::min(tau_bins, nbins) - 1);
      dim_fw.expand_intervals(chisq);
      dim_rc.expand_intervals(chisq);
    } else {
      for (size_t i = 0; i < WIDTH; ++i) {
        dim_fw.extract_intervals_sx(std::min(tau_bins, nbins) - 1, i);
        dim_rc.extract_intervals_sx(std::min(tau_bins, nbins) - 1, i);
        dim_fw.expand_intervals(chisq, i);
        dim_rc.expand_intervals(chisq, i);
      }
    }

    if (params.enum_only) {
      if constexpr (std::is_same_v<T, double>) {
        report_intervals(sout, rid, dim_fw, false);
        report_intervals(sout, rid, dim_rc, true);
      } else {
        for (size_t i = 0; i < WIDTH; ++i) {
          report_intervals(sout, rid, dim_fw, false, i);
          report_intervals(sout, rid, dim_rc, true, i);
        }
      }
    } else {
      dim_fw.compute_prefhistsum();
      dim_rc.compute_prefhistsum();
      dim_fw.map_contiguous_segments(bin_shift);
      dim_rc.map_contiguous_segments(bin_shift);
      report_segments(sout, rid, dim_fw, false);
      report_segments(sout, rid, dim_rc, true);
    }
  }
}

template<typename T>
void QIE<T>::search_mers(const char* cseq, uint64_t len, DIM<T>& dim_fw, DIM<T>& dim_rc)
{
  uint32_t i = 0, j = 0, l = 0;
  uint32_t orrix, rcrix;
  uint64_t orenc64_bp, orenc64_lr, rcenc64_bp;
  for (; i < len; ++i) {
    if (__builtin_expect(SEQ_NT4_TABLE[cseq[i]] >= 4, 0)) {
      l = 0; // TODO: What to do for missing ones?
      continue;
    }
    ++l;
    if (l < k) {
      continue; // TODO: How to propagate the missing ones?
    }
    j = i - k + 1;
    if (l == k) {
      compute_encoding(cseq + j, cseq + i + 1, orenc64_lr, orenc64_bp);
    } else {
      update_encoding(cseq + i, orenc64_lr, orenc64_bp);
    }
    orenc64_bp &= mask_bp;
    orenc64_lr &= mask_lr;
    rcenc64_bp = revcomp_bp64(orenc64_bp, k);
    onmers++;
    const uint64_t bix_j = j >> bin_shift; // The bin index for this k-mer position
#ifdef CANONICAL
    if (rcenc64_bp < orenc64_bp) {
      orrix = lshf->compute_hash(orenc64_bp);
      const uint32_t off_fw = sketch->partial_offset(orrix);
      sketch->prefetch_offset_inc(off_fw);
      const enc_t enc_lr_fw = lshf->drop_ppos_lr(orenc64_lr);
      sketch->prefetch_offset_enc(off_fw);
      uint32_t hdist_fw;
      if (sketch->scan_bucket(off_fw, enc_lr_fw, hdist_fw)) {
        dim_fw.aggregate_mer(hdist_fw, bix_j);
      }
    } else {
      rcrix = lshf->compute_hash(rcenc64_bp);
      const uint32_t off_rc = sketch->partial_offset(rcrix);
      sketch->prefetch_offset_inc(off_rc);                                  // Phase 1
      const enc_t enc_lr_rc = lshf->drop_ppos_lr(bp64_to_lr64(rcenc64_bp)); // Phase 2
      sketch->prefetch_offset_enc(off_rc);                                  // Phase 3
      uint32_t hdist_rc;
      if (sketch->scan_bucket(off_rc, enc_lr_rc, hdist_rc)) {
        dim_rc.aggregate_mer(hdist_rc, bix_j);
      }
    }
#else
    orrix = lshf->compute_hash(orenc64_bp);
    rcrix = lshf->compute_hash(rcenc64_bp);
    const uint32_t off_fw = sketch->partial_offset(orrix);
    const uint32_t off_rc = sketch->partial_offset(rcrix);
    sketch->prefetch_offset_inc(off_fw);
    sketch->prefetch_offset_inc(off_rc);
    const enc_t enc_lr_fw = lshf->drop_ppos_lr(orenc64_lr);
    const enc_t enc_lr_rc = lshf->drop_ppos_lr(bp64_to_lr64(rcenc64_bp));
    sketch->prefetch_offset_enc(off_fw);
    sketch->prefetch_offset_enc(off_rc);
    uint32_t hdist_fw;
    if (sketch->scan_bucket(off_fw, enc_lr_fw, hdist_fw)) {
      dim_fw.aggregate_mer(hdist_fw, bix_j);
    }
    uint32_t hdist_rc;
    if (sketch->scan_bucket(off_rc, enc_lr_rc, hdist_rc)) {
      dim_rc.aggregate_mer(hdist_rc, bix_j);
    }
#endif /* CANONICAL */
  }
}

template<typename T>
void QIE<T>::report_intervals(std::ostream& sout, const str& rid, DIM<T>& dim, bool rc, size_t idx)
{ // TODO: Revisit?
  const str strand = rc ? "-" : "+";
  const double dist_th = at(params.dist_th, idx);
  const uint64_t nbins = dim.get_nbins();
  const uint64_t n = enmers + k - 1;
  uint64_t i = 0;
  interval_t ab = dim.get_interval(i, idx);
  while (ab.first < nbins) {
    const uint64_t a = ab.first << bin_shift;
    const uint64_t b = std::min(((ab.second + 1) << bin_shift) - 1, enmers - 1) + k - 1;
    sout << WRITE_CINTERVAL(qid_batch[bix], n, a, b, strand, rid, dist_th) << '\n';
    ab = dim.get_interval(++i, idx);
  }
}

template<typename T>
inline double QIE<T>::at(T v, const size_t idx)
{
  if constexpr (std::is_same_v<T, double>) {
    return v;
  } else {
    return v[idx];
  }
}

template<typename T>
double DIM<T>::estimate_interval_distance(uint64_t a, uint64_t b, uint64_t bin_shift)
{
  const uint32_t W = hdist_th + 1;
  const uint64_t pa = a * W;
  const uint64_t pb = (b + 1) * W;

  assert(W <= MAXHD+1);
  uint64_t hist[MAXHD+1];
  uint64_t total_merhit = 0;
  for (uint32_t d = 0; d < W; ++d) {
    hist[d] = hdisthist_v[pb + d] - hdisthist_v[pa + d];
    total_merhit += hist[d];
  }

  if (total_merhit == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }

  // TODO: this is not necessarily correct, Ns must be taken care of?
  const uint64_t total_nmers = std::min((b + 1) << bin_shift, nmers) - (a << bin_shift);

  llhf->set_counts(hist, total_nmers - total_merhit);
  auto f = [&](const double& D) { return (*llhf)(D); };
  return boost::math::tools::brent_find_minima(f, EPS, 0.5, 16).first;
}

template<typename T>
void DIM<T>::map_contiguous_segments(uint64_t bin_shift)
{
  segments_v.clear();

  // k-way merge of already-sorted per-threshold breakpoints into a unique sorted list.
  // Each interval [first, second] contributes two breakpoints: first and second+1.
  // Per-threshold breakpoints are already sorted, so a simple merge suffices.
  vec<uint64_t> pts;
  arr<size_t, WIDTH> ix = {};
  for (;;) {
    uint64_t smallest = std::numeric_limits<uint64_t>::max();
    for (size_t ti = 0; ti < WIDTH; ++ti) {
      const auto& ev = eintervals_v[ti];
      if (ix[ti] < ev.size() * 2) {
        const uint64_t v = (ix[ti] & 1) ? ev[ix[ti] >> 1].second + 1 : ev[ix[ti] >> 1].first;
        if (v < smallest) smallest = v;
      }
    }
    if (smallest == std::numeric_limits<uint64_t>::max()) break;
    if (pts.empty() || pts.back() != smallest) pts.push_back(smallest);
    for (size_t ti = 0; ti < WIDTH; ++ti) {
      const auto& ev = eintervals_v[ti];
      while (ix[ti] < ev.size() * 2) {
        const uint64_t v = (ix[ti] & 1) ? ev[ix[ti] >> 1].second + 1 : ev[ix[ti] >> 1].first;
        if (v != smallest) break;
        ++ix[ti];
      }
    }
  }
  if (pts.size() < 2) return;

  // Each consecutive pair of breakpoints defines an contiguos segment [a, b] where
  // the set of thresholds satisfied is constant.
  arr<size_t, WIDTH> ti_ix = {};
  for (size_t pi = 0; pi + 1 < pts.size(); ++pi) {
    const uint64_t a = pts[pi];
    const uint64_t b = pts[pi + 1] - 1;
    uint8_t mask = 0;
    for (size_t ti = 0; ti < WIDTH; ++ti) {
      // Advance past any interval whose right endpoint is before a.
      while (ti_ix[ti] < eintervals_v[ti].size() && eintervals_v[ti][ti_ix[ti]].second < a) {
        ++ti_ix[ti];
      }
      // By breakpoint construction [a,b] is either fully inside or fully
      // outside the current interval, so first <= a is a sufficient check.
      if (ti_ix[ti] < eintervals_v[ti].size() && eintervals_v[ti][ti_ix[ti]].first <= a) {
        mask |= static_cast<uint8_t>(1u << ti);
      }
    }
    if (mask == 0) continue;
    segments_v.push_back({a, b, estimate_interval_distance(a, b, bin_shift), mask});
  }
}

template<typename T>
void QIE<T>::report_segments(std::ostream& sout, const str& rid, const DIM<T>& dim, bool rc)
{
  const str strand = rc ? "-" : "+";
  const uint64_t n = enmers + k - 1;
  const auto& segments_v = dim.get_segments();

  for (const auto& ab : segments_v) {
    if (std::isnan(ab.d_llh)) continue;
    const uint64_t a = ab.start << bin_shift;
    const uint64_t b = std::min(((ab.end + 1) << bin_shift) - 1, enmers - 1) + k - 1;
    // TODO: Revisit how we report (rc/fw and other nuances)?
    sout << WRITE_SEGMENT(qid_batch[bix], n, a, b, strand, rid, ab.d_llh, static_cast<uint32_t>(ab.mask)) << '\n';
  }
}

template class QIE<double>;
template class QIE<cm512_t>;

template class DIM<double>;
template class DIM<cm512_t>;

template class LLH<double>;
template class LLH<cm512_t>;
