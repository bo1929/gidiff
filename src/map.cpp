#include "map.hpp"
#include <boost/math/tools/minima.hpp>

#define EPS 1e-10

// TODO: Further optimize, and polish.
// TODO: A better output format with a header? For both modes...
// TODO: Figure out the default mode.
// TODO: If we will compute the distances, perhaps have an option to merge all overlapping intervals per distance.
// TODO: Merge aggressively as per above reasons.
// TODO: Something interesting with the second derivatives? Is the minimum second-derivative interval interesting?

template<typename T>
DIM<T>::DIM(llh_sptr_t<T> llhf, uint32_t hdist_th, uint64_t nbins, uint64_t nmers, bool segment_mode)
  : llhf(llhf)
  , hdist_th(hdist_th)
  , nbins(nbins)
  , nmers(nmers)
  , segment_mode(segment_mode)
{
  fdc_v.resize(nbins); // ?: fdc_v.reserve(nbins);
  sdc_v.resize(nbins); // ?: sdc_v.reserve(nbins);

  fdps_v.reserve(nbins + 1);
  sdps_v.reserve(nbins + 1);

  fdpmax_v.reserve(nbins + 2); // TODO: Remove?
  fdsmin_v.reserve(nbins + 2); // TODO: Remove?

  if (segment_mode) {
    // Row 0 = zeros (sentinel); aggregate_mer writes into rows from 1 to nbins.
    // compute_prefhistsum() converts in-place
    hdisthist_v.assign((nbins + 1) * (hdist_th + 1), 0);
  }

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
    if (segment_mode) hdisthist_v[(i + 1) * (hdist_th + 1) + hdist_min]++;
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
  if (!segment_mode) return;
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
void DIM<T>::extract_intervals(const uint64_t tau, const size_t idx)
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

// template<typename T> // O(n); makes sense, space concern, no prefmax suffmin
// void DIM<T>::extract_intervals(const uint64_t tau, const size_t idx)
// {
//   // Clear previous intervals for this lane
//   rintervals_v[idx].clear();

//   // -----------------------------------------------------------------
//   // 1. Identify all record highs (indices where fdps[i] is strictly
//   //    greater than all previous prefix sums). These are the only
//   //    possible left endpoints of maximal intervals.
//   // -----------------------------------------------------------------
//   std::vector<uint64_t> rec_idx; // positions of record highs
//   std::vector<double> rec_val;   // corresponding fdps values
//   double max_so_far = -std::numeric_limits<double>::max();
//   for (uint64_t i = 1; i <= n; ++i) {
//     double val = at(fdps_v[i], idx);
//     if (val > max_so_far) {
//       rec_idx.push_back(i);
//       rec_val.push_back(val);
//       max_so_far = val;
//     }
//   }
//   size_t m = rec_idx.size();
//   if (m == 0) return;

//   // -----------------------------------------------------------------
//   // 2. For each record high v, find the largest index b such that
//   //    fdps[b] < v.  This b is the only possible right endpoint for
//   //    a maximal interval starting at that record high (because any
//   //    later b with fdps[b] < v would violate right‑maximality).
//   //    We compute these in a single right‑to‑left scan.
//   // -----------------------------------------------------------------
//   std::vector<uint64_t> last_less(m, 0);
//   int k = static_cast<int>(m) - 1; // start with the largest v
//   for (uint64_t b = n; b >= 1; --b) {
//     double val_b = at(fdps_v[b], idx);
//     while (k >= 0 && rec_val[k] > val_b) {
//       last_less[k] = b; // this b is the largest so far for this v
//       k--;
//     }
//     if (k < 0) break; // all thresholds already have a candidate
//   }

//   // -----------------------------------------------------------------
//   // 3. For each record high, check whether the candidate b satisfies
//   //    the minimum length and left‑maximality conditions.
//   // -----------------------------------------------------------------
//   for (size_t k = 0; k < m; ++k) {
//     uint64_t b = last_less[k];
//     if (b == 0) continue; // no value < v exists

//     // Minimum length: need b - a >= tau  (since tau = min_length_in_bins - 1)
//     if (b >= rec_idx[k] + tau - 1) {
//       double val_b = at(fdps_v[b], idx);
//       double prev = (k == 0) ? -std::numeric_limits<double>::max() : rec_val[k - 1];
//       if (val_b >= prev) { // left‑maximal condition
//         rintervals_v[idx].emplace_back(rec_idx[k], b);
//       }
//     }
//   }
// }

// template<typename T>
// void DIM<T>::extract_intervals(const uint64_t tau, const size_t idx)
// {
//   // O(n + k) algorithm where k = number of suffix minimum positions of fdps_v.
//   // k << n in practice (e.g. k ≈ n/28 for genomic walks with positive bias).
//   //
//   // Eliminates fdpmax_v and fdsmin_v arrays entirely:
//   //   - fdpmax_v[a-1] is replaced by a running scalar (prefix max updated per step)
//   //   - fdsmin_v is replaced by a compact list of suffix minimum positions
//   //
//   // KEY FACT: every valid right endpoint b* is a suffix minimum of fdps_v.
//   // Proof: b* = last j with fdsmin_v[j] < fdps_a.  At b*, fdsmin_v[b*+1] >= fdps_a
//   // but fdsmin_v[b*] < fdps_a.  So min(fdps_v[b*..n]) < min(fdps_v[b*+1..n]),
//   // which means fdps_v[b*] < all of fdps_v[b*+1..n], i.e. b* IS a suffix minimum.
//   //
//   // Suffix minimum positions in left-to-right order have STRICTLY INCREASING values
//   // (proof: if j1 < j2 are both suffix minima, fdps_v[j1] < min(fdps_v[j1+1..n])
//   // <= fdps_v[j2]).  Since fdps_v[a] is strictly increasing across L (the set of
//   // running-maximum positions), the pointer into the suffix minimum list is
//   // monotone non-decreasing: total pointer movement is O(k).
//   //
//   // === Suffix minimum list (built once per call, O(n)) ===
//   // Each entry: (position, scalar value for component idx).
//   // Positions increase left-to-right; values increase left-to-right.
//   struct SufMin
//   {
//     uint64_t pos;
//     double val;
//   };
//   vec<SufMin> smins;
//   {
//     double cur_min = std::numeric_limits<double>::max();
//     for (uint64_t j = n; j >= 1; --j) { // right-to-left; exclude j=0 (b* >= 1 always)
//       const double v = at(fdps_v[j], idx);
//       if (v < cur_min) {
//         cur_min = v;
//         smins.push_back({j, v});
//       }
//     }
//     // Collected in descending-j order (descending values).
//     // Reverse → ascending positions, ascending values.
//     std::reverse(smins.begin(), smins.end());
//   }
//   if (smins.empty()) return;

//   uint64_t smin_ptr = 0;
//   // fdpmax_v[a-1] = max(fdps_v[0..a-2]).  Track as a running scalar.
//   // Before a=1: max of empty set = -inf.
//   // Update rule after inspecting a, before a+1:
//   //   running_max_new = max(running_max_old, fdps_v[a-1])
//   // For a=1: fdps_v[0] = 0, so running_max becomes 0. ✓
//   double running_max = -std::numeric_limits<double>::max();
//   uint64_t b_prev = std::numeric_limits<uint64_t>::max();

//   for (uint64_t a = 1; a <= n; ++a) {
//     const double fdps_a = at(fdps_v[a], idx);
//     const double fdpmax_a = running_max; // = fdpmax_v[a-1] = max(fdps_v[0..a-2])

//     // Update running_max BEFORE any continue: fdpmax_v[a] = max(fdps_v[0..a-1])
//     const double fdps_prev = at(fdps_v[a - 1], idx);
//     if (fdps_prev > running_max) running_max = fdps_prev;

//     if (fdpmax_a >= fdps_a) continue; // a not in L

//     // Advance smin_ptr to the last suffix minimum with val < fdps_a.
//     // smins values are increasing left-to-right, so entries qualifying (val < fdps_a)
//     // form a prefix; the last qualifying entry is at smin_ptr after advancement.
//     // smin_ptr is monotone across all a in L (O(k) total advances).
//     while (smin_ptr + 1 < smins.size() && smins[smin_ptr + 1].val < fdps_a) {
//       ++smin_ptr;
//     }
//     if (smins[smin_ptr].val >= fdps_a) continue; // no suffix minimum < fdps_a

//     const uint64_t b_star = smins[smin_ptr].pos;
//     const double fdps_bstar = smins[smin_ptr].val;

//     if (b_star < a + tau) continue; // minimum length not satisfied

//     if (b_star == b_prev) continue; // non-maximal: earlier a already claimed this b*

//     if (fdps_bstar >= fdpmax_a) { // left-maximal: can't improve by extending left
//       rintervals_v[idx].emplace_back(a, b_star);
//       b_prev = b_star;
//     }
//   }
// }

template<typename T>
void DIM<T>::inclusive_scan()
{
  assert(nbins > 0);
  const uint64_t s = nbins + 1;

  fdps_v.resize(s);
  sdps_v.resize(s);
  fdpmax_v.resize(s + 1); // TODO: Remove?
  fdsmin_v.resize(s + 1); // TODO: Remove?

  if constexpr (std::is_same_v<T, double>) {
    fdps_v[0] = 0.0;
    sdps_v[0] = 0.0;
    for (uint64_t i = 1; i < s; ++i) {
      fdps_v[i] = fdps_v[i - 1] + fdc_v[i - 1];
      sdps_v[i] = sdps_v[i - 1] + sdc_v[i - 1];
    }

    // TODO: Remove both?
    fdpmax_v[0] = -std::numeric_limits<double>::max();
    fdsmin_v[0] = std::numeric_limits<double>::max();
    std::inclusive_scan(
      fdps_v.begin() + 1, fdps_v.end(), fdpmax_v.begin() + 1, [](double a, double b) { return std::max(a, b); });
    std::inclusive_scan(
      fdps_v.rbegin(), fdps_v.rend() - 1, fdsmin_v.rbegin() + 1, [](double a, double b) { return std::min(a, b); });
    fdpmax_v[s] = std::numeric_limits<double>::max();
    fdsmin_v[s] = -std::numeric_limits<double>::max();
    // Until here?
  } else {
    fdps_v[0].fill(0.0);
    sdps_v[0].fill(0.0);
    simde__m512d fdps_acc = simde_mm512_setzero_pd();
    simde__m512d sdps_acc = simde_mm512_setzero_pd();
    for (uint64_t i = 1; i < s; ++i) {
      const simde__m512d fdc = simde_mm512_load_pd(fdc_v[i - 1].data());
      const simde__m512d sdc = simde_mm512_load_pd(sdc_v[i - 1].data());
      fdps_acc = simde_mm512_add_pd(fdps_acc, fdc);
      sdps_acc = simde_mm512_add_pd(sdps_acc, sdc);
      simde_mm512_store_pd(fdps_v[i].data(), fdps_acc);
      simde_mm512_store_pd(sdps_v[i].data(), sdps_acc);
    }

    // TODO: Remove too?
    fdpmax_v[0].fill(-std::numeric_limits<double>::max());
    fdsmin_v[0].fill(std::numeric_limits<double>::max());
    simde__m512d fdpmax_acc = simde_mm512_load_pd(fdpmax_v[0].data());
    simde__m512d fdsmin_acc = simde_mm512_load_pd(fdsmin_v[0].data());
    for (uint64_t i = 1; i < s; ++i) {
      const simde__m512d fdps_front = simde_mm512_load_pd(fdps_v[i].data());
      const simde__m512d fdps_back = simde_mm512_load_pd(fdps_v[s - i].data());
      fdpmax_acc = simde_mm512_max_pd(fdpmax_acc, fdps_front);
      fdsmin_acc = simde_mm512_min_pd(fdsmin_acc, fdps_back);
      simde_mm512_store_pd(fdpmax_v[i].data(), fdpmax_acc);
      simde_mm512_store_pd(fdsmin_v[s - i].data(), fdsmin_acc);
    }
    fdpmax_v[s].fill(std::numeric_limits<double>::max());
    fdsmin_v[s].fill(-std::numeric_limits<double>::max());
    // Until here.
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
  // TODO: Adding these deallocations back may make sense.
  // rintervals_v[idx].clear();
  // rintervals_v[idx].shrink_to_fit();
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

    DIM<T> dim_fw(llhf, params.hdist_th, nbins, enmers, params.segment_mode);
    DIM<T> dim_rc(llhf, params.hdist_th, nbins, enmers, params.segment_mode);
    search_mers(cseq, len, dim_fw, dim_rc);

    // Convert min_length (in base-pairs / k-mer units) to bin units for tau
    const uint64_t tau_bins = (min_length + bin_size - 1) >> bin_shift;

    // Extract intervals for all thresholds
    dim_fw.inclusive_scan();
    // dim_fw.release_accumulators();
    dim_rc.inclusive_scan();
    // dim_rc.release_accumulators();
    if constexpr (std::is_same_v<T, double>) {
      dim_fw.extract_intervals(std::min(tau_bins, nbins) - 1);
      dim_rc.extract_intervals(std::min(tau_bins, nbins) - 1);
      dim_fw.expand_intervals(chisq);
      dim_rc.expand_intervals(chisq);
    } else {
      for (size_t i = 0; i < WIDTH; ++i) {
        dim_fw.extract_intervals(std::min(tau_bins, nbins) - 1, i);
        dim_rc.extract_intervals(std::min(tau_bins, nbins) - 1, i);
        dim_fw.expand_intervals(chisq, i);
        dim_rc.expand_intervals(chisq, i);
      }
    }

    if (params.segment_mode) {
      dim_fw.compute_prefhistsum();
      dim_rc.compute_prefhistsum();
      dim_fw.map_contiguous_segments(bin_shift);
      dim_rc.map_contiguous_segments(bin_shift);
      report_segments(sout, rid, dim_fw, false);
      report_segments(sout, rid, dim_rc, true);
    } else {
      if constexpr (std::is_same_v<T, double>) {
        report_intervals(sout, rid, dim_fw, false);
        report_intervals(sout, rid, dim_rc, true);
      } else {
        for (size_t i = 0; i < WIDTH; ++i) {
          report_intervals(sout, rid, dim_fw, false, i);
          report_intervals(sout, rid, dim_rc, true, i);
        }
      }
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

  vec<uint64_t> hist(W);
  uint64_t total_merhit = 0;
  for (uint32_t d = 0; d < W; ++d) {
    hist[d] = hdisthist_v[pb + d] - hdisthist_v[pa + d];
    total_merhit += hist[d];
  }

  if (total_merhit == 0) {
    return std::numeric_limits<double>::quiet_NaN();
  } // Not sure about this one. TODO: still report MLE, rho might affect this?

  // TODO: this is not necessarily correct, Ns must be taken care of?
  const uint64_t total_nmers = std::min((b + 1) << bin_shift, nmers) - (a << bin_shift);

  llhf->set_counts(hist.data(), total_nmers - total_merhit);
  auto f = [&](const double& D) { return (*llhf)(D); };
  return boost::math::tools::brent_find_minima(f, EPS, 0.5, 16).first;
}

template<typename T>
void DIM<T>::map_contiguous_segments(uint64_t bin_shift)
{
  segments_v.clear();

  // Collect all interval-boundary breakpoints across all thresholds.
  // TODO: Intervals are already sorted per threshold; a k-way merge would give O(n) here.
  vec<uint64_t> pts;
  for (size_t ti = 0; ti < WIDTH; ++ti) {
    for (const auto& iv : eintervals_v[ti]) {
      pts.push_back(iv.first);
      pts.push_back(iv.second + 1);
    }
  }
  if (pts.empty()) return;

  // TODO: This entire part seems slow and inefficient.
  // TODO: Sorting seems unnecessary, as the intervals are already sorted just merging is better?
  std::sort(pts.begin(), pts.end());
  pts.erase(std::unique(pts.begin(), pts.end()), pts.end());

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
    const uint64_t a = ab.start << bin_shift;
    const uint64_t b = std::min(((ab.end + 1) << bin_shift) - 1, enmers - 1) + k - 1;
    // TODO: Revisit how we report (rc/fw and other nuances)?
    sout << WRITE_CINTERVAL(qid_batch[bix], n, a, b, strand, rid, ab.d_llh) << '\t' << static_cast<uint32_t>(ab.mask)
         << '\n';
    // TODO: Either remove the mask or report 8-bit vector for eight thresholds.
  }
}

template class QIE<double>;
template class QIE<cm512_t>;

template class DIM<double>;
template class DIM<cm512_t>;

template class LLH<double>;
template class LLH<cm512_t>;
