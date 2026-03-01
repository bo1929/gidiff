#ifndef _SKETCH_HPP
#define _SKETCH_HPP

#include "types.hpp"
#include "msg.hpp"
#include "hm.hpp"
#include "lshf.hpp"

typedef std::vector<enc_t>::const_iterator vec_enc_it;

class Sketch
{
public:
  Sketch(std::filesystem::path sketch_path);
  void load_from_offset(std::ifstream& stream, uint64_t offset);
  void make_rho_partial();
  bool check_partial(uint32_t rix);
  uint32_t search_mer(uint32_t rix, enc_t enc_lr);
  std::pair<vec_enc_it, vec_enc_it> bucket_indices(uint32_t rix);
  sfhm_sptr_t get_sfhm_sptr();
  lshf_sptr_t get_lshf();
  double get_rho();
  str get_refid() { return refid; }
  uint64_t get_timestamp() { return timestamp; }
  void set_refid(const str& ident) { refid = ident; }
  void set_timestamp(uint64_t ts) { timestamp = ts; }
  uint8_t get_k() { return k; }
  uint8_t get_w() { return w; }
  uint8_t get_h() { return h; }
  uint32_t get_m() { return m; }
  uint32_t get_r() { return r; }
  bool get_frac() { return frac; }
  uint32_t get_nrows() { return nrows; }
  void write_header(std::ofstream& stream);
  void write_config(std::ofstream& stream);

private:
  uint8_t k;
  uint8_t w;
  uint8_t h;
  bool frac;
  uint32_t r;
  uint32_t m;
  double rho;
  uint32_t nrows;
  lshf_sptr_t lshf = nullptr;
  sfhm_sptr_t sfhm = nullptr;
  std::filesystem::path sketch_path;
  str refid;
  uint64_t timestamp;
};

#endif
