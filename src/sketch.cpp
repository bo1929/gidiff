#include "sketch.hpp"

Sketch::Sketch(std::filesystem::path sketch_path)
  : sketch_path(sketch_path)
{
}

void Sketch::write_header(std::ofstream& stream)
{
  uint64_t refid_len = refid.length();
  stream.write(reinterpret_cast<char*>(&refid_len), sizeof(uint64_t));
  stream.write(refid.c_str(), refid_len);
  stream.write(reinterpret_cast<char*>(&timestamp), sizeof(uint64_t));
}

void Sketch::write_config(std::ofstream& stream)
{
  stream.write(reinterpret_cast<char*>(&k), sizeof(uint8_t));
  stream.write(reinterpret_cast<char*>(&w), sizeof(uint8_t));
  stream.write(reinterpret_cast<char*>(&h), sizeof(uint8_t));
  stream.write(reinterpret_cast<char*>(&m), sizeof(uint32_t));
  stream.write(reinterpret_cast<char*>(&r), sizeof(uint32_t));
  stream.write(reinterpret_cast<char*>(&frac), sizeof(bool));
  stream.write(reinterpret_cast<char*>(&nrows), sizeof(uint32_t));
  stream.write(reinterpret_cast<char*>(lshf->ppos_data()), h * sizeof(uint8_t));
  stream.write(reinterpret_cast<char*>(lshf->npos_data()), (k - h) * sizeof(uint8_t));
  stream.write(reinterpret_cast<char*>(&rho), sizeof(double));
}

void Sketch::load_from_offset(std::ifstream& stream, uint64_t offset)
{
  if (offset > 0) {
    stream.seekg(offset);
  }

  uint64_t refid_len;
  stream.read(reinterpret_cast<char*>(&refid_len), sizeof(uint64_t));
  refid.resize(refid_len);
  stream.read(&refid[0], refid_len);
  stream.read(reinterpret_cast<char*>(&timestamp), sizeof(uint64_t));

  stream.read(reinterpret_cast<char*>(&k), sizeof(uint8_t));
  stream.read(reinterpret_cast<char*>(&w), sizeof(uint8_t));
  stream.read(reinterpret_cast<char*>(&h), sizeof(uint8_t));
  stream.read(reinterpret_cast<char*>(&m), sizeof(uint32_t));
  stream.read(reinterpret_cast<char*>(&r), sizeof(uint32_t));
  stream.read(reinterpret_cast<char*>(&frac), sizeof(bool));
  stream.read(reinterpret_cast<char*>(&nrows), sizeof(uint32_t));

  vec<uint8_t> ppos_v(h), npos_v(k - h);
  stream.read(reinterpret_cast<char*>(ppos_v.data()), h * sizeof(uint8_t));
  stream.read(reinterpret_cast<char*>(npos_v.data()), (k - h) * sizeof(uint8_t));

  lshf = std::make_shared<LSHF>(m, ppos_v, npos_v);

  stream.read(reinterpret_cast<char*>(&rho), sizeof(double));

  sfhm = std::make_shared<SFHM>();
  sfhm->load(stream);

  // check_fstream(stream, "Failed to read the sketch file!", sketch_path);
}

void Sketch::make_rho_partial()
{
  if (frac) {
    rho *= (static_cast<double>(r) + 1.0) / static_cast<double>(m);
  } else {
    rho *= 1.0 / static_cast<double>(m);
  }
}

sfhm_sptr_t Sketch::get_sfhm_sptr() { return sfhm; }

lshf_sptr_t Sketch::get_lshf() { return lshf; }

double Sketch::get_rho() { return rho; }

bool Sketch::check_partial(uint32_t rix)
{
  uint32_t rix_res = rix % m;
  return (frac && (rix_res <= r)) || (rix_res == r);
}

uint32_t Sketch::search_mer(uint32_t rix, enc_t enc_lr)
{
  uint32_t offset;
  if (frac) {
    offset = (rix / m) * (r + 1) + (rix % m);
  } else {
    offset = rix / m;
  }
  vec_enc_it ix1 = sfhm->bucket_start(offset);
  vec_enc_it ix2 = sfhm->bucket_next(offset);
  uint32_t hdist_curr;
  uint32_t hdist_min = std::numeric_limits<uint32_t>::max();
  for (; ix1 < ix2; ++ix1) {
    hdist_curr = popcount_lr32((*ix1) ^ enc_lr);
    if (hdist_curr < hdist_min) {
      hdist_min = hdist_curr;
    }
  }
  return hdist_min;
}

std::pair<vec_enc_it, vec_enc_it> Sketch::bucket_indices(uint32_t rix)
{
  uint32_t offset;
  if (frac) {
    offset = (rix / m) * (r + 1) + (rix % m);
  } else {
    offset = rix / m;
  }
  return std::make_pair(sfhm->bucket_start(offset), sfhm->bucket_next(offset));
}
