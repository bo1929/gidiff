#include "map.hpp"
#include <boost/math/tools/minima.hpp>
#include <boost/math/distributions/gamma.hpp>

static constexpr double EPS = 1e-10;

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
    t_q++;
    if (!enum_only) hdisthist_v[(i + 1) * (hdist_th + 1) + hdist_min]++;
    add_to(sdc_v[i], llhf->get_sdc(hdist_min));
    add_to(fdc_v[i], llhf->get_fdc(hdist_min));
  } else {
    u_q++;
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
void DIM<T>::extract_histogram(uint64_t a, uint64_t b, uint64_t bin_shift, vec<uint64_t>& v, uint64_t& u, uint64_t& t) const
{
  const uint32_t W = hdist_th + 1;
  v.assign(MAXHD + 1, 0);
  const simde__mmask8 mask = static_cast<simde__mmask8>((1u << W) - 1);
  const simde__m512i vb = simde_mm512_maskz_loadu_epi64(mask, &hdisthist_v[b * W]);
  const simde__m512i va = simde_mm512_maskz_loadu_epi64(mask, &hdisthist_v[a * W]);
  const simde__m512i vd = simde_mm512_sub_epi64(vb, va);
  simde_mm512_storeu_si512(v.data(), vd);
  const simde__m256i lo = simde_mm512_castsi512_si256(vd);
  const simde__m256i hi = simde_mm512_extracti64x4_epi64(vd, 1);
  const simde__m256i s4 = simde_mm256_add_epi64(lo, hi);
  const simde__m128i s4_lo = simde_mm256_castsi256_si128(s4);
  const simde__m128i s4_hi = simde_mm256_extracti128_si256(s4, 1);
  const simde__m128i s2 = simde_mm_add_epi64(s4_lo, s4_hi);
  t = simde_mm_extract_epi64(s2, 0) + simde_mm_extract_epi64(s2, 1);
  const uint64_t total_nmers = std::min(b << bin_shift, nmers) - (a << bin_shift);
  u = total_nmers - t;
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
void DIM<T>::extrema_scan()
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
void DIM<T>::expand_intervals(const double chisq_th, const size_t idx)
{
  if (rintervals_v[idx].empty()) {
    return;
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
    }

    ap = a;
    bp = b;
  }

  fdiff = at(fdps_v[bp], idx) - at(fdps_v[ap], idx);
  sdiff = at(sdps_v[ap], idx) - at(sdps_v[bp], idx);
  // chisq_val = (sdiff > 0.0) ? (fdiff * fdiff) / sdiff : std::numeric_limits<double>::infinity();
  chisq_val = (fdiff * fdiff) / (sdiff + EPS);
  eintervals_v[idx].emplace_back(ap, bp);
  rintervals_v[idx].clear();
  rintervals_v[idx].shrink_to_fit();
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
  , tau(params.tau)
  , btau((params.tau + bin_size - 1) >> params.bin_shift)
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
  if (!params.enum_only) {
    v_acc.assign(MAXHD + 1, 0);
    gstride = std::max(uint64_t(1), static_cast<uint64_t>(std::ceil(btau * SUBSAMPLE_FACTOR)));
  }
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

    // Convert tau (in base-pairs / k-mer units) to bin units for tau
    const uint64_t btau = (tau + bin_size - 1) >> bin_shift;

    // Extract intervals for all thresholds
    dim_fw.inclusive_scan();
    dim_fw.extrema_scan();
    // dim_fw.release_accumulators();
    dim_rc.inclusive_scan();
    dim_rc.extrema_scan();
    // dim_rc.release_accumulators();

    if constexpr (std::is_same_v<T, double>) {
      /* dim_fw.extract_intervals_sx(std::min(btau, nbins) - 1); */
      /* dim_rc.extract_intervals_sx(std::min(btau, nbins) - 1); */
      dim_fw.extract_intervals_mx(std::min(btau, nbins) - 1);
      dim_rc.extract_intervals_mx(std::min(btau, nbins) - 1);
      dim_fw.expand_intervals(chisq);
      dim_rc.expand_intervals(chisq);
    } else {
      for (size_t i = 0; i < WIDTH; ++i) {
        /* dim_fw.extract_intervals_sx(std::min(btau, nbins) - 1, i); */
        /* dim_rc.extract_intervals_sx(std::min(btau, nbins) - 1, i); */
        dim_fw.extract_intervals_mx(std::min(btau, nbins) - 1, i);
        dim_rc.extract_intervals_mx(std::min(btau, nbins) - 1, i);
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

      // Build positive/negative threshold masks from sign
      uint8_t pos_bv = 0, neg_bv = 0;
      const T sign = llhf->get_sign();
      if constexpr (std::is_same_v<T, double>) {
        if (sign > 0)
          pos_bv = 1;
        else
          neg_bv = 1;
      } else {
        for (size_t ti = 0; ti < WIDTH; ++ti) {
          if (sign[ti] > 0)
            pos_bv |= static_cast<uint8_t>(1u << ti);
          else
            neg_bv |= static_cast<uint8_t>(1u << ti);
        }
      }

      dim_fw.map_contiguous_segments(bin_shift, pos_bv, '<');
      dim_fw.map_contiguous_segments(bin_shift, neg_bv, '>');
      dim_rc.map_contiguous_segments(bin_shift, pos_bv, '<');
      dim_rc.map_contiguous_segments(bin_shift, neg_bv, '>');

      // Extract per-query histograms, compute per-query MLE, accumulate into genome-wide histogram
      vec<uint64_t> v_q_fw, v_q_rc;
      uint64_t u_q_fw = 0, u_q_rc = 0, t_q_fw = 0, t_q_rc = 0;
      dim_fw.extract_histogram(0, nbins, bin_shift, v_q_fw, u_q_fw, t_q_fw);
      dim_rc.extract_histogram(0, nbins, bin_shift, v_q_rc, u_q_rc, t_q_rc);
      const double d_q_fw = compute_mle_dist(v_q_fw, u_q_fw);
      const double d_q_rc = compute_mle_dist(v_q_rc, u_q_rc);

      collect_segments(dim_fw, false, d_q_fw);
      collect_segments(dim_rc, true, d_q_rc);

      store_qrec(dim_fw);
      store_qrec(dim_rc);

      // Accumulate the strand with lower per-query distance into genome-wide histogram
      const bool pick_fw = !(d_q_fw > d_q_rc); // prefer fw on tie or NaN
      const auto& v_q = pick_fw ? v_q_fw : v_q_rc;
      const uint64_t u_q = pick_fw ? u_q_fw : u_q_rc;
      simde__m512i vc = simde_mm512_loadu_si512(v_acc.data());
      vc = simde_mm512_add_epi64(vc, simde_mm512_loadu_si512(v_q.data()));
      simde_mm512_storeu_si512(v_acc.data(), vc);
      u_acc += u_q;
    }
  }

  if (!params.enum_only) {
    d_acc = compute_mle_dist(v_acc, u_acc);
    fit_gamma_significance();
    emit_segments(sout, rid);
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
    const uint64_t bin_j = j >> bin_shift; // The bin index for this k-mer position
#ifdef CANONICAL
    if (rcenc64_bp < orenc64_bp) {
      orrix = lshf->compute_hash(orenc64_bp);
      const uint32_t off_fw = sketch->partial_offset(orrix);
      sketch->prefetch_offset_inc(off_fw);
      const enc_t enc_lr_fw = lshf->drop_ppos_lr(orenc64_lr);
      sketch->prefetch_offset_enc(off_fw);
      uint32_t hdist_fw;
      if (sketch->scan_bucket(off_fw, enc_lr_fw, hdist_fw)) {
        dim_fw.aggregate_mer(hdist_fw, bin_j);
      }
    } else {
      rcrix = lshf->compute_hash(rcenc64_bp);
      const uint32_t off_rc = sketch->partial_offset(rcrix);
      sketch->prefetch_offset_inc(off_rc);                                  // Phase 1
      const enc_t enc_lr_rc = lshf->drop_ppos_lr(bp64_to_lr64(rcenc64_bp)); // Phase 2
      sketch->prefetch_offset_enc(off_rc);                                  // Phase 3
      uint32_t hdist_rc;
      if (sketch->scan_bucket(off_rc, enc_lr_rc, hdist_rc)) {
        dim_rc.aggregate_mer(hdist_rc, bin_j);
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
      dim_fw.aggregate_mer(hdist_fw, bin_j);
    }
    uint32_t hdist_rc;
    if (sketch->scan_bucket(off_rc, enc_lr_rc, hdist_rc)) {
      dim_rc.aggregate_mer(hdist_rc, bin_j);
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
    const uint64_t a = (ab.first << bin_shift) + 1;
    const uint64_t b = std::min(((ab.second + 1) << bin_shift), enmers) + k - 1;
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
  vec<uint64_t> v;
  uint64_t u, t;
  extract_histogram(a, b + 1, bin_shift, v, u, t);
  llhf->set_counts(v.data(), u);
  auto f = [&](const double& D) { return (*llhf)(D); };
  const double ub = (t == 0) ? 0.75 + EPS : 0.5;
  return boost::math::tools::brent_find_minima(f, EPS, ub, 24).first;
}

template<typename T>
void DIM<T>::map_contiguous_segments(uint64_t bin_shift, uint8_t th_bv, char sign)
{
  if (th_bv == 0) return;

  // k-way merge of already-sorted per-threshold breakpoints into a unique sorted list.
  // Each interval [first, second] contributes two breakpoints: first and second+1.
  // Per-threshold breakpoints are already sorted, so a simple merge suffices.
  vec<uint64_t> pts;
  arr<size_t, WIDTH> ix = {};
  for (;;) {
    uint64_t smin = std::numeric_limits<uint64_t>::max();
    for (size_t ti = 0; ti < WIDTH; ++ti) {
      if (!(th_bv & (1u << ti))) continue;
      const auto& ev = eintervals_v[ti];
      if (ix[ti] < ev.size() * 2) {
        const uint64_t v = (ix[ti] & 1) ? ev[ix[ti] >> 1].second + 1 : ev[ix[ti] >> 1].first;
        if (v < smin) smin = v;
      }
    }
    if (smin == std::numeric_limits<uint64_t>::max()) break;
    if (pts.empty() || pts.back() != smin) pts.push_back(smin);
    for (size_t ti = 0; ti < WIDTH; ++ti) {
      if (!(th_bv & (1u << ti))) continue;
      const auto& ev = eintervals_v[ti];
      while (ix[ti] < ev.size() * 2) {
        const uint64_t v = (ix[ti] & 1) ? ev[ix[ti] >> 1].second + 1 : ev[ix[ti] >> 1].first;
        if (v != smin) break;
        ++ix[ti];
      }
    }
  }
  if (pts.size() < 2) return;

  // Each consecutive pair of breakpoints defines a contiguous segment [a, b] where
  // the set of thresholds satisfied is constant.
  arr<size_t, WIDTH> ti_ix = {};
  for (size_t pi = 0; pi + 1 < pts.size(); ++pi) {
    const uint64_t a = pts[pi];
    const uint64_t b = pts[pi + 1] - 1;
    uint8_t mask = 0;
    for (size_t ti = 0; ti < WIDTH; ++ti) {
      if (!(th_bv & (1u << ti))) continue;
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
    const double d_s = estimate_interval_distance(a, b, bin_shift);
    segments_v.push_back({a, b, d_s, mask, sign});
  }
}

template<typename T>
double QIE<T>::compute_mle_dist(const vec<uint64_t>& v, uint64_t u)
{
  uint64_t t = 0;
  for (auto c : v)
    t += c;
  if (t == 0) return std::numeric_limits<double>::quiet_NaN();
  llhf->set_counts(v.data(), u);
  auto f = [&](const double& D) { return (*llhf)(D); };
  return boost::math::tools::brent_find_minima(f, EPS, 0.5, 24).first;
}

template<typename T>
void QIE<T>::collect_segments(const DIM<T>& dim, bool rc, double d_q)
{
  const char strand = rc ? '-' : '+';
  const uint64_t n = enmers + k - 1;
  const auto& segments_v = dim.get_segments();

  for (const auto& ab : segments_v) {
    // if (std::isnan(ab.d_s)) continue;
    const uint64_t a = (ab.start << bin_shift) + 1;
    const uint64_t b = std::min(((ab.end + 1) << bin_shift), enmers) + k - 1;
    const uint64_t nbins_s = ab.end - ab.start + 1;
    output_records.push_back({bix, n, a, b, nbins_s, strand, ab.d_s, d_q, 0.0, 0.0, ab.mask, ab.sign});
  }
}

template<typename T>
void QIE<T>::store_qrec(const DIM<T>& dim)
{
  const uint32_t W = params.hdist_th + 1;
  const uint64_t G = gstride;
  const uint64_t nbins_q = dim.get_nbins();
  const uint64_t nbins_qsub = nbins_q / G + 1; // rows at 0, G, 2G, ..., floor(nbins/G)*G

  qrec_t qr;
  qr.nbins = nbins_q;
  qr.nmers = dim.get_nmers();
  qr.hdisthist_v.resize(nbins_qsub * W);

  const auto& src = dim.get_hdisthist();
  for (uint64_t i = 0; i < nbins_qsub; ++i) {
    const uint64_t rix = i * G;
    std::copy_n(src.begin() + rix * W, W, qr.hdisthist_v.begin() + i * W);
  }

  qrecs.push_back(std::move(qr));
}

template<typename T>
void QIE<T>::build_length_grid()
{
  const uint64_t G = gstride;

  // Find max usable bins across all stored contigs
  uint64_t max_bins = 0;
  for (const auto& qr : qrecs) {
    const uint64_t usable = (qr.nbins / G) * G;
    if (usable > max_bins) max_bins = usable;
  }

  if (max_bins < G) return;

  // Build geometric grid of lengths, all multiples of G
  length_grid.clear();
  uint64_t L = G;
  while (L <= max_bins) {
    length_grid.push_back(L);
    uint64_t next = static_cast<uint64_t>(std::ceil(L * GRID_GROWTH / G)) * G;
    if (next <= L) next = L + G;
    L = next;
  }
}

template<typename T>
void QIE<T>::fit_gamma_significance()
{
  using boost::math::gamma_distribution;

  build_length_grid();

  // Fit Gamma to a sorted sample vector using quantile matching + 2D Nelder-Mead
  auto fit_gamma = [](const vec<double>& sorted_samples) -> std::pair<double, double> {
    const size_t n = sorted_samples.size();
    if (n < 3) return {1.0, 1.0};

    // Compute empirical quantiles
    std::array<double, GAMMA_FIT_PROBS.size()> emp_q;
    for (size_t i = 0; i < GAMMA_FIT_PROBS.size(); ++i) {
      const double idx = GAMMA_FIT_PROBS[i] * (n - 1);
      const size_t lo = static_cast<size_t>(idx);
      const size_t hi = std::min(lo + 1, n - 1);
      const double frac = idx - lo;
      emp_q[i] = sorted_samples[lo] * (1.0 - frac) + sorted_samples[hi] * frac;
    }

    if (emp_q.back() < EPS) return {1.0, EPS};

    // Mean distance for initialization
    double mean_d = 0.0;
    for (double x : sorted_samples)
      mean_d += x;
    mean_d /= n;
    if (mean_d < EPS) mean_d = EPS;

    // SSE objective over both shape (alpha) and scale (beta)
    auto sse = [&](double a, double b) -> double {
      if (a < EPS || b < EPS) return 1e30;
      gamma_distribution<double> g(a, b);
      double s = 0.0;
      for (size_t i = 0; i < GAMMA_FIT_PROBS.size(); ++i) {
        const double diff = quantile(g, GAMMA_FIT_PROBS[i]) - emp_q[i];
        s += diff * diff;
      }
      return s;
    };

    // Nelder-Mead on (alpha, beta), initialized at (1, mean_d)
    using pt = std::array<double, 2>;
    std::array<pt, 3> S = {pt{1.0, mean_d}, pt{2.0, mean_d}, pt{1.0, 2.0 * mean_d}};
    std::array<double, 3> F;
    for (int i = 0; i < 3; ++i)
      F[i] = sse(S[i][0], S[i][1]);

    for (int iter = 0; iter < 500; ++iter) {
      // Sort vertices by objective
      if (F[0] > F[1]) {
        std::swap(S[0], S[1]);
        std::swap(F[0], F[1]);
      }
      if (F[1] > F[2]) {
        std::swap(S[1], S[2]);
        std::swap(F[1], F[2]);
      }
      if (F[0] > F[1]) {
        std::swap(S[0], S[1]);
        std::swap(F[0], F[1]);
      }

      if (F[2] - F[0] < 1e-12 && iter > 10) break;

      // Centroid of best 2
      pt c = {0.5 * (S[0][0] + S[1][0]), 0.5 * (S[0][1] + S[1][1])};

      // Reflect
      pt r = {2.0 * c[0] - S[2][0], 2.0 * c[1] - S[2][1]};
      double fr = sse(r[0], r[1]);

      if (fr < F[0]) {
        // Expand
        pt e = {c[0] + 2.0 * (r[0] - c[0]), c[1] + 2.0 * (r[1] - c[1])};
        double fe = sse(e[0], e[1]);
        if (fe < fr) {
          S[2] = e;
          F[2] = fe;
        } else {
          S[2] = r;
          F[2] = fr;
        }
      } else if (fr < F[1]) {
        S[2] = r;
        F[2] = fr;
      } else {
        // Contract
        pt w = (fr < F[2]) ? r : S[2];
        double fw = std::min(fr, F[2]);
        pt cc = {0.5 * (c[0] + w[0]), 0.5 * (c[1] + w[1])};
        double fc = sse(cc[0], cc[1]);
        if (fc <= fw) {
          S[2] = cc;
          F[2] = fc;
        } else {
          // Shrink toward best
          for (int j = 1; j < 3; ++j) {
            S[j][0] = 0.5 * (S[0][0] + S[j][0]);
            S[j][1] = 0.5 * (S[0][1] + S[j][1]);
            F[j] = sse(S[j][0], S[j][1]);
          }
        }
      }
    }

    // Return best vertex
    int best = 0;
    if (F[1] < F[best]) best = 1;
    if (F[2] < F[best]) best = 2;
    return {std::max(S[best][0], EPS), std::max(S[best][1], EPS)};
  };

  // Sample null distances per grid length, fit Gamma immediately, accumulate for fallback
  const uint32_t W = params.hdist_th + 1;
  const uint64_t G = gstride;
  vec<double> all_samples;
  std::unordered_map<uint64_t, std::pair<double, double>> gamma_params;

  for (const uint64_t L_g : length_grid) {
    const uint64_t L_steps = L_g / G;
    vec<double> dists;

    for (const auto& qr : qrecs) {
      const uint64_t nbins_qsub = qr.nbins / G + 1;
      if (nbins_qsub <= L_steps) continue;

      const uint64_t max_start_idx = nbins_qsub - 1 - L_steps;
      const uint64_t n_samples = std::min(NULL_SAMPLES_PER_LENGTH, max_start_idx + 1);
      std::uniform_int_distribution<uint64_t> dist(0, max_start_idx);

      for (uint64_t si = 0; si < n_samples; ++si) {
        const uint64_t idx = dist(gen);

        // Compute histogram: row[idx + L_steps] - row[idx]
        vec<uint64_t> v(MAXHD + 1, 0);
        uint64_t t = 0;
        for (uint32_t d = 0; d < W; ++d) {
          v[d] = qr.hdisthist_v[(idx + L_steps) * W + d] - qr.hdisthist_v[idx * W + d];
          t += v[d];
        }
        if (t == 0) continue;

        // Compute miss count
        const uint64_t start_bin = idx * G;
        const uint64_t end_bin = start_bin + L_g;
        const uint64_t total_nmers = std::min(end_bin << bin_shift, qr.nmers) - (start_bin << bin_shift);
        const uint64_t u = total_nmers - t;

        // Compute distance via LLH + Brent
        llhf->set_counts(v.data(), u);
        auto f = [&](const double& D) { return (*llhf)(D); };
        dists.push_back(boost::math::tools::brent_find_minima(f, EPS, 0.5, 24).first);
      }
    }

    all_samples.insert(all_samples.end(), dists.begin(), dists.end());
    std::sort(dists.begin(), dists.end());
    if (dists.size() >= NULL_MIN_SAMPLES) {
      gamma_params[L_g] = fit_gamma(dists);
      // TODO: remove debug print
      double mean_d = 0.0;
      for (double x : dists)
        mean_d += x;
      mean_d /= dists.size();
      const size_t nd = dists.size();
      const double q1 = dists[nd / 4], med = dists[nd / 2], q3 = dists[3 * nd / 4];
      const auto& [a, b] = gamma_params[L_g];
      std::cerr << "[DEBUG gamma] L=" << L_g << " n=" << nd << " mean=" << mean_d << " Q1=" << q1 << " med=" << med
                << " Q3=" << q3 << " alpha=" << a << " beta=" << b << '\n';
    }
  }

  // Fit fallback from all pooled samples
  std::sort(all_samples.begin(), all_samples.end());
  std::pair<double, double> fallback_params = {1.0, 1.0};
  if (all_samples.size() >= NULL_MIN_SAMPLES) {
    fallback_params = fit_gamma(all_samples);
    // TODO: remove debug print
    /* double mean_all = 0.0;
    for (double x : all_samples) mean_all += x;
    mean_all /= all_samples.size();
    const size_t na = all_samples.size();
    const double q1a = all_samples[na / 4], meda = all_samples[na / 2], q3a = all_samples[3 * na / 4];
    std::cerr << "[DEBUG gamma] FALLBACK n=" << na
              << " mean=" << mean_all << " Q1=" << q1a << " med=" << meda << " Q3=" << q3a
              << " alpha=" << fallback_params.first
              << " beta=" << fallback_params.second << '\n'; */
  }

  // Score each output record (snap nbins_s up to nearest grid length)
  for (auto& r : output_records) {
    auto git = std::lower_bound(length_grid.begin(), length_grid.end(), r.nbins_s);
    const uint64_t grid_L = (git != length_grid.end()) ? *git : (length_grid.empty() ? 0 : length_grid.back());
    auto it = gamma_params.find(grid_L);
    const auto [alpha, beta] = (it != gamma_params.end()) ? it->second : fallback_params;
    gamma_distribution<double> g(alpha, beta);
    r.percentile = boost::math::cdf(g, r.d_s);
    const double med = quantile(g, 0.5);
    r.fold = (med > EPS) ? r.d_s / med : std::numeric_limits<double>::quiet_NaN();
  }
}

template<typename T>
void QIE<T>::emit_segments(std::ostream& sout, const str& rid) const
{
  const str strand_fw = "+";
  const str strand_rc = "-";
  for (const auto& r : output_records) {
    const str& strand = (r.strand == '+') ? strand_fw : strand_rc;
    sout << WRITE_SEGMENT(qid_batch[r.bix],
                          r.n,
                          r.a,
                          r.b,
                          strand,
                          rid,
                          r.d_s,
                          static_cast<uint32_t>(r.mask),
                          r.sign,
                          r.d_q,
                          d_acc,
                          r.percentile,
                          r.fold)
         << '\n';
  }
}

template class QIE<double>;
template class QIE<cm512_t>;

template class DIM<double>;
template class DIM<cm512_t>;

template class LLH<double>;
template class LLH<cm512_t>;
