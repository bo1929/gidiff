#include "map.hpp"

template<size_t N>
DIM<N>::DIM(const std::array<llh_sptr_t, N>& llhf_arr, uint64_t en_mers)
  : llhf_arr(llhf_arr)
  , en_mers(en_mers)
{
  for (size_t i = 0; i < N; ++i) {
    opposite[i] = llhf_arr[i]->opposite;
    hdist_th[i] = llhf_arr[i]->hdist_th;
    merhit_count[i] = 0;
    mermiss_count[i] = 0;
    hdisthist_v[i].resize(hdist_th[i] + 1, 0);
    fdt[i] = 0;
    sdt[i] = 0;
    d_llh[i] = std::numeric_limits<double>::quiet_NaN();
    v_llh[i] = std::numeric_limits<double>::quiet_NaN();
  }
  
  fdc_v.resize(en_mers);
  sdc_v.resize(en_mers);
}

template<size_t N>
void DIM<N>::aggregate_mer(sketch_sptr_t sketch, uint32_t rix, enc_t enc_lr, uint64_t i)
{
  const uint32_t hdist_min = sketch->search_mer(rix, enc_lr);
  
  if constexpr (N == 1) {
    if (hdist_min <= hdist_th[0]) {
      ++merhit_count[0];
      ++hdisthist_v[0][hdist_min];
      fdc_v[i][0] = llhf_arr[0]->get_fdc(hdist_min);
      sdc_v[i][0] = llhf_arr[0]->get_sdc(hdist_min);
    } else {
      ++mermiss_count[0];
      fdc_v[i][0] = llhf_arr[0]->get_fdc();
      sdc_v[i][0] = llhf_arr[0]->get_sdc();
    }
    fdt[0] += fdc_v[i][0];
    sdt[0] += sdc_v[i][0];
  } else {
    alignas(64) double fdc_tmp[8];
    alignas(64) double sdc_tmp[8];
    
    for (size_t j = 0; j < 8; ++j) {
      if (hdist_min <= hdist_th[j]) {
        ++merhit_count[j];
        ++hdisthist_v[j][hdist_min];
        fdc_tmp[j] = llhf_arr[j]->get_fdc(hdist_min);
        sdc_tmp[j] = llhf_arr[j]->get_sdc(hdist_min);
      } else {
        ++mermiss_count[j];
        fdc_tmp[j] = llhf_arr[j]->get_fdc();
        sdc_tmp[j] = llhf_arr[j]->get_sdc();
      }
    }
    
    simde__m512d fdc_vec = simde_mm512_load_pd(fdc_tmp);
    simde__m512d sdc_vec = simde_mm512_load_pd(sdc_tmp);
    simde_mm512_store_pd(fdc_v[i].data(), fdc_vec);
    simde_mm512_store_pd(sdc_v[i].data(), sdc_vec);
    
    simde__m512d fdt_vec = simde_mm512_load_pd(fdt.data());
    simde__m512d sdt_vec = simde_mm512_load_pd(sdt.data());
    fdt_vec = simde_mm512_add_pd(fdt_vec, fdc_vec);
    sdt_vec = simde_mm512_add_pd(sdt_vec, sdc_vec);
    simde_mm512_store_pd(fdt.data(), fdt_vec);
    simde_mm512_store_pd(sdt.data(), sdt_vec);
  }
}

template<size_t N>
void DIM<N>::inclusive_scan()
{
  assert(en_mers > 0);
  const uint64_t s = en_mers + 1;
  fdps_v.resize(s);
  sdps_v.resize(s);
  fdpmax_v.resize(s + 1);
  fdsmin_v.resize(s + 1);
  
  if constexpr (N == 1) {
    fdps_v[0][0] = 0;
    sdps_v[0][0] = 0;
    
    for (uint64_t i = 1; i < s; ++i) {
      fdps_v[i][0] = fdps_v[i - 1][0] + fdc_v[i - 1][0];
      sdps_v[i][0] = sdps_v[i - 1][0] + sdc_v[i - 1][0];
    }
    
    fdpmax_v[0][0] = -std::numeric_limits<double>::infinity();
    for (uint64_t i = 0; i < s; ++i) {
      fdpmax_v[i + 1][0] = std::max(fdpmax_v[i][0], fdps_v[i][0]);
    }
    
    fdsmin_v[s][0] = std::numeric_limits<double>::infinity();
    for (int64_t i = s - 1; i >= 0; --i) {
      fdsmin_v[i][0] = std::min(fdps_v[i][0], fdsmin_v[i + 1][0]);
    }
  } else {
    simde__m512d fdps_acc = simde_mm512_setzero_pd();
    simde__m512d sdps_acc = simde_mm512_setzero_pd();
    simde_mm512_store_pd(fdps_v[0].data(), fdps_acc);
    simde_mm512_store_pd(sdps_v[0].data(), sdps_acc);
    
    for (uint64_t i = 1; i < s; ++i) {
      simde__m512d fdc = simde_mm512_load_pd(fdc_v[i - 1].data());
      simde__m512d sdc = simde_mm512_load_pd(sdc_v[i - 1].data());
      fdps_acc = simde_mm512_add_pd(fdps_acc, fdc);
      sdps_acc = simde_mm512_add_pd(sdps_acc, sdc);
      simde_mm512_store_pd(fdps_v[i].data(), fdps_acc);
      simde_mm512_store_pd(sdps_v[i].data(), sdps_acc);
    }
    
    alignas(64) double neg_inf[8];
    std::fill_n(neg_inf, 8, -std::numeric_limits<double>::infinity());
    simde__m512d fdpmax_acc = simde_mm512_load_pd(neg_inf);
    simde_mm512_store_pd(fdpmax_v[0].data(), fdpmax_acc);
    
    for (uint64_t i = 0; i < s; ++i) {
      simde__m512d fdps = simde_mm512_load_pd(fdps_v[i].data());
      fdpmax_acc = simde_mm512_max_pd(fdpmax_acc, fdps);
      simde_mm512_store_pd(fdpmax_v[i + 1].data(), fdpmax_acc);
    }
    
    alignas(64) double pos_inf[8];
    std::fill_n(pos_inf, 8, std::numeric_limits<double>::infinity());
    simde__m512d fdsmin_acc = simde_mm512_load_pd(pos_inf);
    simde_mm512_store_pd(fdsmin_v[s].data(), fdsmin_acc);
    
    for (int64_t i = s - 1; i >= 0; --i) {
      simde__m512d fdps = simde_mm512_load_pd(fdps_v[i].data());
      fdsmin_acc = simde_mm512_min_pd(fdsmin_acc, fdps);
      simde_mm512_store_pd(fdsmin_v[i].data(), fdsmin_acc);
    }
  }
}

template<size_t N>
void DIM<N>::extract_intervals(uint64_t tau, size_t idx)
{
  for (uint64_t a = 1, b = 1; a <= en_mers; ++a) {
    if (fdpmax_v[a - 1][idx] >= fdps_v[a][idx]) {
      continue;
    }
    if (b < (a + tau)) {
      b = a + tau - 1;
    }
    if (b > en_mers) {
      break;
    }
    if (fdsmin_v[b + 1][idx] >= fdps_v[a][idx]) {
      continue;
    }
    ++b;
    while (b <= en_mers) {
      if ((fdpmax_v[a - 1][idx] <= fdps_v[b][idx]) && 
          (fdps_v[b][idx] < fdps_v[a][idx]) && 
          (fdps_v[a][idx] <= fdsmin_v[b + 1][idx])) {
        rintervals_v[idx].emplace_back(a, b);
        break;
      }
      ++b;
    }
  }
}

template<size_t N>
void DIM<N>::report_intervals(std::ostream& output_stream, const std::string& identifier, size_t idx)
{
  if (eintervals_v[idx].empty()) {
    output_stream << identifier << ',' << en_mers << ',' << en_mers << ',' << en_mers << ",0,0\n";
  } else {
    for (uint64_t i = 0; i < eintervals_v[idx].size(); ++i) {
      const uint64_t a = eintervals_v[idx][i].first;
      const uint64_t b = eintervals_v[idx][i].second;
      output_stream << identifier << ',' << a << ',' << b - 1 << ',' << en_mers << ',' 
                    << (fdps_v[b][idx] - fdps_v[a][idx]) << ',' << chisq_v[idx][i] << '\n';
    }
  }
}

template<size_t N>
uint64_t DIM<N>::expand_intervals(double chisq_th, size_t idx)
{
  if (rintervals_v[idx].empty()) {
    return 0;
  }
  
  uint64_t ap = rintervals_v[idx][0].first;
  uint64_t bp = rintervals_v[idx][0].second;
  
  for (uint64_t i = 1; i < rintervals_v[idx].size(); ++i) {
    uint64_t a = rintervals_v[idx][i].first;
    const uint64_t b = rintervals_v[idx][i].second;
    
    const double fdiff = fdps_v[b][idx] - fdps_v[ap][idx];
    const double sdiff = sdps_v[ap][idx] - sdps_v[b][idx];
    const double chisq_val = (fdiff * fdiff) / sdiff;
    assert(chisq_val > 0);
    
    if ((chisq_val < chisq_th) && (a < bp)) {
      a = ap;
    } else {
      eintervals_v[idx].emplace_back(ap, bp);
      chisq_v[idx].push_back(chisq_val);
    }
    ap = a;
    bp = b;
  }
  
  const double fdiff = fdps_v[bp][idx] - fdps_v[ap][idx];
  const double sdiff = sdps_v[ap][idx] - sdps_v[bp][idx];
  const double chisq_val = (fdiff * fdiff) / sdiff;
  assert(chisq_val > 0);
  
  eintervals_v[idx].emplace_back(ap, bp);
  chisq_v[idx].push_back(chisq_val);
  return eintervals_v[idx].size();
}

template<size_t N>
void DIM<N>::optimize_loglikelihood()
{
  for (size_t i = 0; i < N; ++i) {
    llhf_arr[i]->set_counts(hdisthist_v[i].data(), mermiss_count[i]);
  }
}

QIE::QIE(sketch_sptr_t sketch, lshf_sptr_t lshf, qseq_sptr_t qs, const std::vector<params_t>& params_vec)
  : sketch(sketch)
  , lshf(lshf)
  , rho(sketch->get_rho())
  , k(lshf->get_k())
  , h(lshf->get_h())
  , m(lshf->get_m())
  , batch_size(qs->cbatch_size)
  , mask_bp(std::numeric_limits<uint64_t>::max() >> ((32 - k) * 2))
  , mask_lr(((std::numeric_limits<uint64_t>::max() >> (64 - k)) << 32) + 
            ((std::numeric_limits<uint64_t>::max() << 32) >> (64 - k)))
  , params_vec(params_vec)
  , num_params(params_vec.size())
{
  if (num_params != 1 && num_params != 8) {
    std::cerr << "Error: Must provide exactly 1 or 8 dist_th values, got " << num_params << std::endl;
    std::exit(1);
  }
  
  std::swap(qs->seq_batch, seq_batch);
  std::swap(qs->identifer_batch, identifer_batch);
}

void QIE::map_sequences(std::ostream& output_stream)
{
  if (num_params == 1) {
    map_sequences_impl<1>(output_stream);
  } else {
    map_sequences_impl<8>(output_stream);
  }
}

template<size_t N>
void QIE::map_sequences_impl(std::ostream& output_stream)
{
  strstream batch_stream;
  
  std::array<llh_sptr_t, N> llhf_arr;
  std::array<bool, N> opposite;
  
  for (size_t i = 0; i < N; ++i) {
    const double dist_th = params_vec[i].dist_th;
    llhf_arr[i] = std::make_shared<LLH>(h, k, params_vec[i].hdist_th, dist_th, rho);
    opposite[i] = dist_th < 0;
  }
  
  for (bix = 0; bix < batch_size; ++bix) {
    const char* cseq = seq_batch[bix].data();
    const uint64_t len = seq_batch[bix].size();
    onmers = 0;
    en_mers = len - k + 1;
    
    DIM<N> dim_or(llhf_arr, en_mers);
    DIM<N> dim_rc(llhf_arr, en_mers);
    
    search_mers<N>(cseq, len, dim_or, dim_rc);
    
    dim_or.inclusive_scan();
    dim_rc.inclusive_scan();
    
    for (size_t i = 0; i < N; ++i) {
      auto fdt_or = dim_or.get_fdt();
      auto fdt_rc = dim_rc.get_fdt();
      
      const bool use_rc = opposite[i] ? fdt_or[i] < fdt_rc[i] : fdt_or[i] >= fdt_rc[i];
      DIM<N>& dim = use_rc ? dim_rc : dim_or;
      
      dim.extract_intervals(std::min(params_vec[i].min_length, en_mers) - 1, i);
      dim.expand_intervals(params_vec[i].chisq, i);
      dim.report_intervals(batch_stream, identifer_batch[bix], i);
    }
  }
  
  output_stream << batch_stream.rdbuf();
}

template<size_t N>
void QIE::search_mers(const char* cseq, uint64_t len, DIM<N>& dim_or, DIM<N>& dim_rc)
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
    ++onmers;
    
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
#endif
  }
}

template class DIM<1>;
template class DIM<8>;
