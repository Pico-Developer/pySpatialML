#include <cassert>
#include <fstream>
#include <set>
#include <string>
#include <unordered_map>

#define SECUREMR_SERIALIZATION_PARSE_ONLY
#include "securemr_utils/serialization.cpp"

int main(int argc, char** argv) {
  assert(argc == 2);
  std::ifstream input(argv[1]);
  assert(input.good());

  SecureMR::Json spec;
  input >> spec;
  assert(spec.is_object());
  for (const char* key : {"tensors", "operators", "inputs", "outputs"}) {
    assert(spec.contains(key));
  }

  std::unordered_map<std::string, std::shared_ptr<SecureMR::PipelineTensor>> tensors;
  for (auto it = spec.at("tensors").begin(); it != spec.at("tensors").end(); ++it) {
    SecureMR::ValidateTensorSpec(it.key(), it.value());
    SecureMR::TensorAttribute attribute{};
    assert(it.value().value("is_gltf", false) || SecureMR::JsonToTensorAttribute(it.value(), attribute));
    tensors.emplace(it.key(), nullptr);
  }

  SecureMR::ValidateTopLevelTensorLists(spec, tensors);
  std::set<std::string> types;
  for (const auto& operatorSpec : spec.at("operators")) {
    assert(operatorSpec.is_object());
    assert(operatorSpec.contains("type"));
    assert(operatorSpec.contains("inputs"));
    assert(operatorSpec.contains("outputs"));
    const std::string type = SecureMR::FormatOperatorType(operatorSpec.at("type").get<std::string>());
    const auto inputs = SecureMR::ParseOperatorTensorSlots(operatorSpec.at("inputs"), "inputs");
    const auto outputs = SecureMR::ParseOperatorTensorSlots(operatorSpec.at("outputs"), "outputs");
    SecureMR::ValidateOperatorTensorReferences(inputs, "inputs", tensors);
    SecureMR::ValidateOperatorTensorReferences(outputs, "outputs", tensors);
    SecureMR::ValidateOperatorSlots(type, inputs, outputs);
    if (type == "run_algorithm") {
      SecureMR::ValidateModelMetadata(operatorSpec.at("model"), inputs, outputs);
    }
    types.insert(type);
  }

  assert(types.size() == spec.at("operators").size());

  SecureMR::Json imageModeSpec = {
      {"tensors",
       {{"right", {{"dimensions", {2, 2}}, {"channels", 3}, {"data_type", 1},
                    {"is_placeholder", false}, {"usage", 6}}},
        {"left", {{"dimensions", {2, 2}}, {"channels", 3}, {"data_type", 1},
                   {"is_placeholder", false}, {"usage", 6}}}}},
      {"operators",
       {{{"type", "XR_SECURE_MR_OPERATOR_TYPE_RECTIFIED_VST_ACCESS_PICO"},
         {"inputs", SecureMR::Json::array()},
         {"outputs", {{{"tensor", "right"}}, {{"tensor", "left"}}}}},
        {{"type", "XR_SECURE_MR_OPERATOR_TYPE_ASSIGNMENT_PICO"},
         {"inputs", {{{"tensor", "left"}}}},
         {"outputs", {{{"tensor", "right"}}}}}}},
      {"inputs", SecureMR::Json::array()},
      {"outputs", SecureMR::Json::array()},
  };
  SecureMR::RemoveOperatorsAndPromoteOutputsToInputs(imageModeSpec, {"camera_access"});
  assert(imageModeSpec.at("operators").size() == 1);
  assert(imageModeSpec.at("inputs") == SecureMR::Json({"right", "left"}));
  assert(imageModeSpec.at("tensors").at("right").at("is_placeholder") == true);
  assert(imageModeSpec.at("tensors").at("left").at("is_placeholder") == true);
  return 0;
}
