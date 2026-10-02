#include <cassert>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "model_io.h"

int main() {
  const auto bindings = SecureMR::NormalizeModelBindings(
      {{"left", "shared"}, {"right", "shared"}});
  assert(bindings.size() == 2);
  assert(bindings[0].operatorName == "left");
  assert(bindings[0].modelNodeName == "left");
  assert(bindings[0].tensorName == "shared");
  assert(bindings[1].operatorName == "right");
  assert(bindings[1].modelNodeName == "right");
  assert(bindings[1].tensorName == "shared");
  assert(bindings[0].operatorName != bindings[1].operatorName);

  char destination[512]{};
  SecureMR::CopyOperatorIoName(destination, std::string(511, 'a'), "model node name");
  assert(std::string(destination).size() == 511);

  bool rejectedCapacity = false;
  try {
    SecureMR::CopyOperatorIoName(destination, std::string(512, 'b'), "model node name");
  } catch (const std::runtime_error&) {
    rejectedCapacity = true;
  }
  assert(rejectedCapacity);

  bool rejectedNul = false;
  try {
    SecureMR::CopyOperatorIoName(destination, std::string("bad\0name", 8), "model node name");
  } catch (const std::runtime_error&) {
    rejectedNul = true;
  }
  assert(rejectedNul);

  assert(SecureMR::IsValidModelName("model_42"));
  assert(!SecureMR::IsValidModelName("model-name"));
  assert(!SecureMR::IsValidModelName(""));
  assert(SecureMR::NormalizeModelName("") == "main");
  assert(SecureMR::NormalizeModelName("model_42") == "model_42");
  bool rejectedModelName = false;
  try {
    SecureMR::NormalizeModelName("model-name");
  } catch (const std::runtime_error&) {
    rejectedModelName = true;
  }
  assert(rejectedModelName);
}
