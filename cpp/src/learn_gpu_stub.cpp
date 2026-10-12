// Builds without CUDA: the GPU learner is unavailable (the reference learner still works).
#include "laminar/batch_learner.h"
#include <stdexcept>
namespace laminar {
std::unique_ptr<Learner> make_gpu_learner(const LearnConfig&, uint8_t*) { throw std::runtime_error("built without CUDA: no GPU learner"); }
}
