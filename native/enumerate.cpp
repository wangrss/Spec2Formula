#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr double kHydrogenMass = 1.00782503223;
constexpr double kEpsilon = 1.0e-10;
constexpr int kElementCount = 16;
constexpr int kReachabilityWords = 8;
constexpr int kHydrogenIndex = 1;
constexpr int kHydrogenMaximum = 77;

const std::array<const char*, kElementCount> kOutputElements = {
    "C", "H", "B", "N", "O", "F", "Si", "P", "S", "Cl", "As", "Se", "Br", "Sn", "Sb", "I"};
const std::array<int, kElementCount> kNominalMass = {
    12, 1, 11, 14, 16, 19, 28, 31, 32, 35, 75, 80, 79, 120, 121, 127};

struct Element {
  double mass;
  int valence;
  int maximum;
  bool carbon;
  int output_index;
};

const std::vector<Element> kHeavyElements = {
    {126.904468, 1, 5, false, 15},
    {120.9038120, 3, 1, false, 14},
    {119.90220163, 4, 1, false, 13},
    {79.9165218, 2, 2, false, 11},
    {78.9183376, 1, 6, false, 12},
    {74.92159457, 3, 1, false, 10},
    {34.968852682, 1, 8, false, 9},
    {31.9720711744, 2, 8, false, 8},
    {30.97376199842, 3, 5, false, 7},
    {27.97692653465, 4, 5, false, 6},
    {18.99840316273, 1, 17, false, 5},
    {15.99491461957, 2, 19, false, 4},
    {14.00307400443, 3, 13, false, 3},
    {12.0, 4, 43, true, 0},
    {11.00930536, 3, 1, false, 2},
};

struct Center {
  std::int64_t center_id = -1;
  std::string center_key;
  double neutral_mass = 0.0;
};

struct Candidate {
  std::array<std::uint8_t, kElementCount> counts{};
  double exact_mass = 0.0;
  float dbe = 0.0f;
};

std::vector<std::string> SplitCsv(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) fields.push_back(field);
  return fields;
}

std::vector<Center> ReadCenters(const std::filesystem::path& path) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open input: " + path.string());
  std::string line;
  if (!std::getline(input, line)) throw std::runtime_error("empty centers CSV");
  if (!line.empty() && line.back() == '\r') line.pop_back();
  if (line != "center_id,center_key,neutral_mass") throw std::runtime_error("unexpected centers header");
  std::vector<Center> centers;
  while (std::getline(input, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    auto fields = SplitCsv(line);
    if (fields.size() != 3) throw std::runtime_error("unexpected centers row");
    centers.push_back({std::stoll(fields[0]), fields[1], std::stod(fields[2])});
  }
  return centers;
}

class Enumerator {
 public:
  explicit Enumerator(double center)
      : center_(center), half_width_(std::max(0.001, center * 3.0e-6)),
        low_(center - half_width_), high_(center + half_width_) {
    suffix_max_mass_.assign(kHeavyElements.size() + 1, 0.0);
    for (int i = static_cast<int>(kHeavyElements.size()) - 1; i >= 0; --i) {
      const auto& element = kHeavyElements[static_cast<std::size_t>(i)];
      const int physical = static_cast<int>(std::floor(high_ / element.mass + 1.0e-12));
      suffix_max_mass_[static_cast<std::size_t>(i)] =
          suffix_max_mass_[static_cast<std::size_t>(i + 1)] +
          std::min(element.maximum, physical) * element.mass;
    }
  }

  std::vector<Candidate> Run() {
    Recurse(0, 0.0, 0, 0, 0);
    std::sort(candidates_.begin(), candidates_.end(), [](const Candidate& left, const Candidate& right) {
      if (left.exact_mass != right.exact_mass) return left.exact_mass < right.exact_mass;
      return left.counts < right.counts;
    });
    return std::move(candidates_);
  }

 private:
  void Recurse(std::size_t index, double partial_mass, int atom_count,
               int valence_sum, int valence_minus_two_sum) {
    if (index == kHeavyElements.size()) {
      const int hydrogen_low = std::max(
          0, static_cast<int>(std::ceil((low_ - partial_mass) / kHydrogenMass - kEpsilon)));
      const int hydrogen_high = std::min(
          kHydrogenMaximum,
          static_cast<int>(std::floor((high_ - partial_mass) / kHydrogenMass + kEpsilon)));
      for (int hydrogen = hydrogen_low; hydrogen <= hydrogen_high; ++hydrogen) {
        const int final_atoms = atom_count + hydrogen;
        const int final_valence = valence_sum + hydrogen;
        const int final_minus_two = valence_minus_two_sum - hydrogen;
        const double dbe = 1.0 + 0.5 * static_cast<double>(final_minus_two);
        if (dbe < -kEpsilon || (final_valence & 1) != 0) continue;
        if (final_atoms > 1 && final_valence < 2 * (final_atoms - 1)) continue;
        if (final_atoms > 1 && final_valence < 8) continue;
        const double mass = partial_mass + hydrogen * kHydrogenMass;
        if (mass < low_ - kEpsilon || mass > high_ + kEpsilon) continue;
        current_counts_[kHydrogenIndex] = static_cast<std::uint8_t>(hydrogen);
        candidates_.push_back({current_counts_, mass, static_cast<float>(dbe)});
      }
      current_counts_[kHydrogenIndex] = 0;
      return;
    }
    const auto& element = kHeavyElements[index];
    const int minimum = element.carbon ? 1 : 0;
    const int physical = static_cast<int>(std::floor((high_ - partial_mass) / element.mass + 1.0e-12));
    const int maximum = std::min(element.maximum, physical);
    const double remaining_max = suffix_max_mass_[index + 1] + kHydrogenMaximum * kHydrogenMass;
    for (int count = minimum; count <= maximum; ++count) {
      const double next_mass = partial_mass + count * element.mass;
      if (next_mass > high_ + kEpsilon) break;
      if (next_mass + remaining_max < low_ - kEpsilon) continue;
      current_counts_[static_cast<std::size_t>(element.output_index)] = static_cast<std::uint8_t>(count);
      Recurse(index + 1, next_mass, atom_count + count,
              valence_sum + count * element.valence,
              valence_minus_two_sum + count * (element.valence - 2));
    }
    current_counts_[static_cast<std::size_t>(element.output_index)] = 0;
  }

  double center_;
  double half_width_;
  double low_;
  double high_;
  std::vector<double> suffix_max_mass_;
  std::array<std::uint8_t, kElementCount> current_counts_{};
  std::vector<Candidate> candidates_;
};

} // namespace
int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::runtime_error("usage: enumerate NEUTRAL_MASS");
    const double mass = std::stod(argv[1]);
    if (!std::isfinite(mass) || mass <= 0.001 || mass >= 500.0)
      throw std::runtime_error("neutral mass out of range");
    const auto candidates = Enumerator(mass).Run();
    std::cout << std::setprecision(17);
    for (const auto& candidate : candidates) {
      std::cout << candidate.exact_mass << "\t" << candidate.dbe;
      for (auto count : candidate.counts) std::cout << "\t" << static_cast<int>(count);
      std::cout << "\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << "\n";
    return 1;
  }
}
