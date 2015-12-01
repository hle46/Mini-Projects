#include <iostream>
#include <vector>
#include <random>

using std::string;

template <typename T> class Knuth {
public:
  static void shuffle(std::vector<T> &a) {
    // Seed with a real random value, if available
    std::random_device rd;

    // Used to choose a random number between 0 and 1
    std::default_random_engine gen(rd());
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    
    int N = a.size();
    for (int i = 0; i < N; ++i) {
      // Choose index uniformly in [i, N - 1]
      int r = i + (int) (uniform_dist(gen) * (N - i));
      swap(a[i], a[r]);
    }
  }
private:
  Knuth() = default;
};

int main() {
  string str;
  std::vector<string> strs;
  while (std::cin >> str) {
    strs.emplace_back(str);
  }
  Knuth<string>::shuffle(strs);
  for (auto &str: strs) {
    std::cout << str << "\n";
  }
  return 0;
}
