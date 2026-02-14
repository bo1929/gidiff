#include "map.hpp"

DIM::DIM(llh_sptr_t llhf, uint64_t en_mers)
  : llhf(llhf)
  , en_mers(en_mers)
  , opposite(llhf->opposite)
  , hdist_th(llhf->hdist_th)
{
  hdisthist_v.resize(hdist_th + 1, 0);
  fdc_v.resize(en_mers);
  sdc_v.resize(en_mers);
  // fdc_v.reserve(en_mers);
  // sdc_v.reserve(en_mers);
}

void DIM::aggregate_mer(sketch_sptr_t sketch, uint32_t rix, enc_t enc_lr, uint64_t i)
{
  const uint32_t hdist_min = sketch->search_mer(rix, enc_lr);
  if (hdist_min <= hdist_th) {
    merhit_count++;
    hdisthist_v[hdist_min]++;
    // sdc_v.push_back(llhf->get_sdc(hdist_min));
    // fdc_v.push_back(llhf->get_fdc(hdist_min));
    sdc_v[i] = llhf->get_sdc(hdist_min);
    fdc_v[i] = llhf->get_fdc(hdist_min);
  } else {
    mermiss_count++;
    // sdc_v.push_back(llhf->get_sdc());
    // fdc_v.push_back(llhf->get_fdc());
    sdc_v[i] = llhf->get_sdc();
    fdc_v[i] = llhf->get_fdc();
  }
  // fdt += fdc_v.back();
  // sdt += sdc_v.back();
  fdt += fdc_v[i];
  sdt += sdc_v[i];
}

// void DIM::skip_mer()
// {
//   merna_count++;
//   sdc_v.push_back(0);
//   fdc_v.push_back(0);
// }

void DIM::inclusive_scan()
{
  assert(en_mers > 0);
  const uint64_t s = en_mers + 1;
  fdps_v.resize(s);
  sdps_v.resize(s);
  fdps_v[0] = 0;
  sdps_v[0] = 0;
  for (uint64_t i = 1; i < s; ++i) {
    fdps_v[i] = fdps_v[i - 1] + fdc_v[i - 1];
    sdps_v[i] = sdps_v[i - 1] + sdc_v[i - 1];
  }
  fdpmax_v.resize(s + 1);
  fdpmax_v.front() = -std::numeric_limits<double>::infinity();
  std::inclusive_scan(fdps_v.begin(), fdps_v.end(), fdpmax_v.begin() + 1, [](double a, double b) { return std::max(a, b); });
  fdsmin_v.resize(s);
  std::inclusive_scan(fdps_v.rbegin(), fdps_v.rend(), fdsmin_v.rbegin(), [](double a, double b) { return std::min(a, b); });
  fdsmin_v.push_back(std::numeric_limits<double>::infinity());
}

void DIM::extract_intervals(uint64_t tau)
{
  for (uint64_t a = 1, b = 1; a <= en_mers; ++a) {
    if (fdpmax_v[a - 1] >= fdps_v[a]) {
      continue;
    }
    if (b < (a + tau)) {
      b = a + tau - 1;
    }
    if (b > en_mers) {
      break;
    }
    if (fdsmin_v[b + 1] >= fdps_v[a]) {
      continue;
    }
    b++;
    while (b <= en_mers) {
      if ((fdpmax_v[a - 1] <= fdps_v[b]) && (fdps_v[b] < fdps_v[a]) && (fdps_v[a] <= fdsmin_v[b + 1])) {
        rintervals_v.emplace_back(a, b);
        break;
      }
      b++;
    }
  }
}

void DIM::report_intervals(std::ostream& output_stream, const std::string& identifier)
{ // TODO: Revisit.
  if (eintervals_v.empty()) {
    output_stream << identifier << ',' << en_mers << ',' << en_mers << ',' << en_mers << ",0,0\n";
  } else {
    for (uint64_t i = 0; i < eintervals_v.size(); ++i) {
      const uint64_t a = eintervals_v[i].first;
      const uint64_t b = eintervals_v[i].second;
      output_stream << identifier << ',' << a << ',' << b - 1 << ',' << en_mers << ',' << (fdps_v[b] - fdps_v[a]) << ','
                    << chisq_v[i] << '\n';
    }
  }
}

uint64_t DIM::expand_intervals(double chisq_th)
{
  if (rintervals_v.empty()) {
    return 0;
  }
  double fdiff, sdiff, chisq_val;
  uint64_t a, ap, b, bp;
  ap = rintervals_v[0].first;
  bp = rintervals_v[0].second;
  for (uint64_t i = 1; i < rintervals_v.size(); ++i) {
    a = rintervals_v[i].first;
    b = rintervals_v[i].second;
    fdiff = fdps_v[b] - fdps_v[ap];
    sdiff = sdps_v[ap] - sdps_v[b];
    chisq_val = (fdiff * fdiff) / sdiff;
    assert(chisq_val > 0);
    if ((chisq_val < chisq_th) && (a < bp)) {
      a = ap;
    } else {
      eintervals_v.emplace_back(ap, bp);
      chisq_v.push_back(chisq_val);
    }
    ap = a;
    bp = b;
  }
  fdiff = fdps_v[b] - fdps_v[ap];
  sdiff = sdps_v[ap] - sdps_v[b];
  chisq_val = (fdiff * fdiff) / sdiff;
  assert(chisq_val > 0);
  eintervals_v.emplace_back(ap, bp);
  chisq_v.push_back(chisq_val);
  return eintervals_v.size();
}

void DIM::optimize_loglikelihood()
{
  llhf->set_counts(hdisthist_v.data(), mermiss_count);
  /* std::pair<double, double> sol_r = boost::math::tools::brent_find_minima((*llhf), 1e-10, 0.5, 16); */
  /* d_llh = sol_r.first; */
  /* v_llh = sol_r.second; */
}

void QIE::map_sequences(std::ostream& output_stream)
{
  strstream batch_stream;
  // batch_stream.precision(4);
  const bool opposite = dist_th < 0;
  for (bix = 0; bix < batch_size; ++bix) {
    const char* cseq = seq_batch[bix].data();
    const uint64_t len = seq_batch[bix].size();
    onmers = 0;
    en_mers = len - k + 1;

    DIM dim_or(llhf, en_mers);
    DIM dim_rc(llhf, en_mers);
    search_mers(cseq, len, dim_or, dim_rc);

    const bool use_rc = opposite ? dim_or.get_fdt() < dim_rc.get_fdt() : dim_or.get_fdt() >= dim_rc.get_fdt();
    DIM& dim = use_rc ? dim_rc : dim_or;
    dim.inclusive_scan();
    dim.extract_intervals(std::min(min_length, en_mers) - 1);
    dim.expand_intervals(chisq);
    dim.report_intervals(batch_stream, identifer_batch[bix]);
  }
  output_stream << batch_stream.rdbuf();
}

QIE::QIE(sketch_sptr_t sketch, lshf_sptr_t lshf, qseq_sptr_t qs, params_t params)
  : sketch(sketch)
  , rho(sketch->get_rho())
  , lshf(lshf)
  , k(lshf->get_k())
  , h(lshf->get_h())
  , m(lshf->get_m())
  , batch_size(qs->cbatch_size)
  , hdist_th(params.hdist_th)
  , min_length(params.min_length)
  , dist_th(params.dist_th)
  , chisq(params.chisq)
{
  std::swap(qs->seq_batch, seq_batch);
  std::swap(qs->identifer_batch, identifer_batch);
  llhf = std::make_shared<LLH>(h, k, hdist_th, dist_th, rho);
  const uint64_t u64m = std::numeric_limits<uint64_t>::max();
  mask_lr = ((u64m >> (64 - k)) << 32) + ((u64m << 32) >> (64 - k));
  mask_bp = u64m >> ((32 - k) * 2);
}

void QIE::search_mers(const char* cseq, uint64_t len, DIM& dim_or, DIM& dim_rc)
{
  uint32_t i = 0, j = 0, l = 0;
  uint32_t orrix, rcrix;
  uint64_t orenc64_bp, orenc64_lr, rcenc64_bp;
  for (; i < len; ++i) {
    if (SEQ_NT4_TABLE[cseq[i]] >= 4) {
      l = 0;
      continue;
    }
    ++l;
    if (l < k) {
      continue;
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
#ifdef CANONICAL
    if (rcenc64_bp < orenc64_bp) {
      orrix = lshf->compute_hash(orenc64_bp);
      if (sketch->check_partial(orrix)) {
        dim_or.aggregate_mer(sketch, orrix, lshf->drop_ppos_lr(orenc64_lr), j);
      }
    } else {
      rcrix = lshf->compute_hash(rcenc64_bp);
      if (sketch->check_partial(rcrix)) {
        dim_rc.aggregate_mer(sketch, rcrix, lshf->drop_ppos_lr(bp64_to_lr64(rcenc64_bp)), j);
      }
    }
#else
    orrix = lshf->compute_hash(orenc64_bp);
    if (sketch->check_partial(orrix)) {
      dim_or.aggregate_mer(sketch, orrix, lshf->drop_ppos_lr(orenc64_lr), j);
    }
    rcrix = lshf->compute_hash(rcenc64_bp);
    if (sketch->check_partial(rcrix)) {
      dim_rc.aggregate_mer(sketch, rcrix, lshf->drop_ppos_lr(bp64_to_lr64(rcenc64_bp)), j);
    }
#endif /* CANONICAL */
  }
}
