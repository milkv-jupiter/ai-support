#ifndef SUPPORT_SRC_CORE_ENGINE_H_
#define SUPPORT_SRC_CORE_ENGINE_H_

#include <string>
#include <vector>

#include "opencv2/opencv.hpp"
#include "src/core/ort_wrapper.h"
#include "src/utils/json.hpp"
using json = nlohmann::json;
class Engine {
 public:
  Engine() { OrtWrapper ortwrapper_; }
  ~Engine() {}
  int Init(std::string instanceName, std::string modelFilepath);
  int Init(json config);
  size_t GetInputCount() { return ortwrapper_.GetInputCount(); }
  size_t GetOutputCount() { return ortwrapper_.GetOutputCount(); }
  std::vector<std::vector<int64_t>> GetInputDims() {
    return ortwrapper_.GetInputDims();
  }
  std::vector<std::vector<int64_t>> GetOutputDims() {
    return ortwrapper_.GetOutputDims();
  }
  std::vector<Ort::Value> Interpreter(
      std::vector<std::vector<float>> &input_values_handler);

 protected:
 private:
  OrtWrapper ortwrapper_;
};

#endif  // SUPPORT_SRC_CORE_ENGINE_H_
