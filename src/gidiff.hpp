#ifndef _GIDIFF_H
#define _GIDIFF_H

#include <chrono>
#include <ctime>
#include "common.hpp"
#include "types.hpp"
#include "lshf.hpp"
#include "rqseq.hpp"
#include "map.hpp"
#include "sketch.hpp"
#include "hm.hpp"
#include "CLI11.hpp"

#define VERSION "v0.0.0"
#define PRINT_VERSION std::cerr << "??? version: " << VERSION << std::endl;
#define STRSTREAM_PRECISION 4

extern uint32_t num_threads;
extern std::string invocation;

const auto url_validator = CLI::Validator(
  [](std::string& input) {
    if (match_url(input)) {
      return std::string("");
    } else {
      return "Given URL is not valid: " + input;
    }
  },
  "URL",
  "URL validator");

class BaseLSH
{
public:
  void set_lshf();
  void set_nrows();
  bool validate_configuration();
  void save_configuration(std::ofstream& cfg_stream);
  void set_sketch_defaults()
  {
    k = 27;
    w = k + 5;
    h = 12;
    m = 2;
    r = 1;
    frac = true;
    nrows = pow(2, 2 * h - 1);
    // sdust_t = 0;
    // sdust_w = 0;
  }

protected:
  uint8_t w;
  uint8_t k;
  uint8_t h;
  bool frac;
  uint32_t m;
  uint32_t r;
  uint32_t nrows;
  uint32_t sdust_t = 0;
  uint32_t sdust_w = 0;
  lshf_sptr_t lshf = nullptr;
};

class SketchTarget : public BaseLSH
{
public:
  SketchTarget(CLI::App& sc);
  void create_sketch();
  void save_sketch();

private:
  double rho;
  std::string input;
  std::filesystem::path sketch_path;
  sfhm_sptr_t sketch_sfhm = nullptr;
};

class MapSketch
{
public:
  MapSketch(CLI::App& sc);
  void load_sketch();
  void map_sequences();
  void header_dreport(strstream& dreport_stream);
  uint32_t get_total_qseq() { return total_qseq; }

private:
  sketch_sptr_t sketch = nullptr;
  std::filesystem::path sketch_path;
  std::string query;
  std::filesystem::path output_path;
  std::ofstream output_file;
  std::ostream* output_stream = &std::cout;
  uint32_t hdist_th = 4;
  uint64_t total_qseq = 0;
  uint64_t min_length = 0;
  double chisq = 3.841; // 95%
  std::vector<double> dist_th;
};

#endif
