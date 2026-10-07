#include "securemr_base.h"

#include <android/log.h>
#include <sys/system_properties.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <filesystem>
#include <memory>
#include <mutex>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "image_utils/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "image_utils/stb_image_resize2.h"

#include "common.h"
#include "logger.h"
#include "openxr_program.h"
#include "securemr_utils/pipeline.h"
#include "securemr_utils/readback_async.h"
#include "securemr_utils/session.h"
#include "securemr_utils/tensor.h"
#include "securemr_utils/utils.h"

namespace {

constexpr int kDefaultCameraWidth = 580;
constexpr int kDefaultCameraHeight = 326;
constexpr auto kOneShotReadbackDelay = std::chrono::milliseconds(250);
constexpr const char* kDefaultPackageRoot =
    "/sdcard/Android/data/com.bytedance.pico.pyspatialml.xr_runner/files/package";
constexpr const char* kDefaultOutputDir =
    "/sdcard/Android/data/com.bytedance.pico.pyspatialml.xr_runner/files/outputs";
constexpr const char* kEmptyPropertyValue = "__pyspatialml_empty__";

std::string GetProp(const char* key, const std::string& fallback = "") {
  char value[PROP_VALUE_MAX] = {};
  const int length = __system_property_get(key, value);
  if (length <= 0) {
    return fallback;
  }
  std::string result(value, static_cast<size_t>(length));
  return result == kEmptyPropertyValue ? std::string{} : result;
}

std::string JsonEscape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char character : value) {
    if (character == '\\' || character == '"') {
      escaped.push_back('\\');
    }
    escaped.push_back(character);
  }
  return escaped;
}

bool GetBoolProp(const char* key, bool fallback = false) {
  std::string value = GetProp(key);
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return std::tolower(c); });
  if (value.empty()) {
    return fallback;
  }
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

int GetIntProp(const char* key, int fallback) {
  const std::string value = GetProp(key);
  if (value.empty()) {
    return fallback;
  }
  try {
    return std::stoi(value);
  } catch (...) {
    return fallback;
  }
}

std::vector<std::string> SplitCsv(const std::string& value) {
  std::vector<std::string> out;
  std::stringstream stream(value);
  std::string item;
  while (std::getline(stream, item, ',')) {
    item.erase(item.begin(), std::find_if(item.begin(), item.end(), [](unsigned char ch) {
      return !std::isspace(ch);
    }));
    item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch) {
      return !std::isspace(ch);
    }).base(), item.end());
    if (!item.empty()) {
      out.push_back(item);
    }
  }
  return out;
}

long long SteadyNowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

size_t DataTypeSize(XrSecureMrTensorDataTypePICO dataType) {
  switch (dataType) {
    case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO:
    case XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO:
    case XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_UINT8_PICO:
      return 1;
    case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO:
    case XR_SECURE_MR_TENSOR_DATA_TYPE_INT16_PICO:
      return 2;
    case XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO:
    case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO:
    case XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_FLOAT32_PICO:
      return 4;
    case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT64_PICO:
      return 8;
    default:
      return 0;
  }
}

size_t TensorByteSize(const SecureMR::TensorAttribute& attr) {
  size_t count = static_cast<size_t>(std::max<int8_t>(attr.channels, 1));
  for (int dim : attr.dimensions) {
    count *= static_cast<size_t>(std::max(dim, 1));
  }
  return count * DataTypeSize(attr.dataType);
}

bool ReadFile(const std::filesystem::path& path, std::vector<uint8_t>& out) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return false;
  }
  out.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
  return true;
}

bool WriteFile(const std::filesystem::path& path, const std::vector<uint8_t>& data) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    return false;
  }
  output.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
  return output.good();
}

std::vector<uint8_t> ConvertImageToTensor(const std::filesystem::path& path, const SecureMR::TensorAttribute& attr) {
  if (attr.dataType != XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO || attr.dimensions.size() < 2 || attr.channels <= 0) {
    return {};
  }
  const int targetHeight = attr.dimensions[0];
  const int targetWidth = attr.dimensions[1];
  const int channels = attr.channels;
  if (channels != 1 && channels != 3 && channels != 4) {
    return {};
  }

  int width = 0;
  int height = 0;
  int sourceChannels = 0;
  unsigned char* image = stbi_load(path.string().c_str(), &width, &height, &sourceChannels, channels);
  if (image == nullptr) {
    return {};
  }

  std::vector<uint8_t> resized(static_cast<size_t>(targetWidth) * targetHeight * channels);
  unsigned char* result = stbir_resize_uint8_srgb(image, width, height, width * channels, resized.data(), targetWidth,
                                                  targetHeight, targetWidth * channels,
                                                  static_cast<stbir_pixel_layout>(channels));
  stbi_image_free(image);
  if (result == nullptr) {
    return {};
  }
  return resized;
}

std::filesystem::path ResolveInputPath(const std::string& baseInputPath, const std::string& tensorName) {
  if (baseInputPath.empty()) {
    return {};
  }
  std::filesystem::path path(baseInputPath);
  if (!std::filesystem::is_directory(path)) {
    return path;
  }
  for (const char* ext : {".bin", ".raw", ".jpg", ".jpeg", ".png"}) {
    std::filesystem::path candidate = path / (tensorName + ext);
    if (std::filesystem::exists(candidate)) {
      return candidate;
    }
  }
  return {};
}

std::filesystem::path ResolveDefaultInputPath(const std::string& baseInputPath) {
  if (baseInputPath.empty()) {
    return {};
  }
  std::filesystem::path path(baseInputPath);
  if (!std::filesystem::is_directory(path)) {
    return path;
  }
  for (const char* ext : {".bin", ".raw", ".jpg", ".jpeg", ".png"}) {
    std::filesystem::path candidate = path / (std::string("__default") + ext);
    if (std::filesystem::exists(candidate)) {
      return candidate;
    }
  }
  return {};
}

bool IsImageTensor(const SecureMR::TensorAttribute& attr) {
  return attr.dataType == XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO && attr.dimensions.size() >= 2 &&
         (attr.channels == 1 || attr.channels == 3 || attr.channels == 4);
}

bool IsDefaultableTensorName(const std::string& tensorName, const SecureMR::TensorAttribute& attr) {
  return (tensorName.find("timestamp") != std::string::npos &&
          attr.dataType == XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO) ||
         (tensorName.find("camera_matrix") != std::string::npos &&
          attr.dataType == XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO);
}

bool IsImageModeExternalTensor(const std::string& tensorName, const SecureMR::TensorAttribute& attr) {
  return tensorName.rfind("vst_", 0) == 0 && (IsImageTensor(attr) || IsDefaultableTensorName(tensorName, attr));
}

std::vector<uint8_t> DefaultTensorData(const std::string& tensorName, const SecureMR::TensorAttribute& attr) {
  const size_t expectedBytes = TensorByteSize(attr);
  std::vector<uint8_t> data(expectedBytes, 0);
  if (expectedBytes == 0) {
    return data;
  }

  if (attr.dataType == XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO && attr.channels == 1 &&
      attr.dimensions.size() == 2 && attr.dimensions[0] == 3 && attr.dimensions[1] == 3) {
    auto* values = reinterpret_cast<float*>(data.data());
    values[0] = 1.0f;
    values[4] = 1.0f;
    values[8] = 1.0f;
  } else if (tensorName.find("timestamp") != std::string::npos &&
             attr.dataType == XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO) {
    auto* values = reinterpret_cast<int32_t*>(data.data());
    const size_t count = expectedBytes / sizeof(int32_t);
    for (size_t idx = 0; idx < count; ++idx) {
      values[idx] = 0;
    }
  }
  return data;
}

bool IsImagePath(const std::filesystem::path& path) {
  std::string ext = path.extension().string();
  std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
  return ext == ".jpg" || ext == ".jpeg" || ext == ".png";
}

std::string JoinShape(const std::vector<int>& values) {
  std::string out;
  for (size_t idx = 0; idx < values.size(); ++idx) {
    if (idx != 0) {
      out += "x";
    }
    out += std::to_string(values[idx]);
  }
  return out;
}

std::string DataTypeName(XrSecureMrTensorDataTypePICO dataType) {
  switch (dataType) {
    case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT8_PICO:
    case XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_UINT8_PICO:
      return "uint8";
    case XR_SECURE_MR_TENSOR_DATA_TYPE_INT8_PICO:
      return "int8";
    case XR_SECURE_MR_TENSOR_DATA_TYPE_UINT16_PICO:
      return "uint16";
    case XR_SECURE_MR_TENSOR_DATA_TYPE_INT16_PICO:
      return "int16";
    case XR_SECURE_MR_TENSOR_DATA_TYPE_INT32_PICO:
      return "int32";
    case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT32_PICO:
    case XR_SECURE_MR_TENSOR_DATA_TYPE_DYNAMIC_TEXTURE_FLOAT32_PICO:
      return "float32";
    case XR_SECURE_MR_TENSOR_DATA_TYPE_FLOAT64_PICO:
      return "float64";
    default:
      return "unknown";
  }
}

}  // namespace

namespace SecureMR {

struct OutputMetadata {
  std::string filename;
  std::string tensorName;
  std::string pipelineId;
  std::vector<int> dimensions;
  int channels = 0;
  int dataType = 0;
  std::string dtype;
  size_t bytes = 0;
  bool isOutput = true;
};

class XrPipelineRunnerProgram final : public ISecureMR {
 public:
  XrPipelineRunnerProgram(const XrInstance& instance, const XrSession& session)
      : xrInstance_(instance), xrSession_(session) {}

  ~XrPipelineRunnerProgram() override {
    keepRunning_ = false;
    initialized_.notify_all();
    if (readback_) {
      readback_->Stop();
    }
    if (runnerThread_.joinable()) {
      runnerThread_.join();
    }
    if (initializerThread_.joinable()) {
      initializerThread_.join();
    }
  }

  void CreateFramework() override {
    const int width = GetIntProp("debug.pyspatialml.xr_runner.camera_width", kDefaultCameraWidth);
    const int height = GetIntProp("debug.pyspatialml.xr_runner.camera_height", kDefaultCameraHeight);
    frameworkSession_ = std::make_shared<FrameworkSession>(xrInstance_, xrSession_, width, height);
  }

  void CreatePipelines() override {
    initializerThread_ = std::thread([this]() {
      const std::filesystem::path packageRoot =
          GetProp("debug.pyspatialml.xr_runner.package", kDefaultPackageRoot);
      outputDir_ = GetProp("debug.pyspatialml.xr_runner.output", kDefaultOutputDir);
      inputPath_ = GetProp("debug.pyspatialml.xr_runner.input", "");
      useVst_ = GetBoolProp("debug.pyspatialml.xr_runner.use_vst", inputPath_.empty());
      loop_ = GetBoolProp("debug.pyspatialml.xr_runner.loop", false);
      dumpAll_ = GetBoolProp("debug.pyspatialml.xr_runner.dump_all", false);
      intervalMs_ = GetIntProp("debug.pyspatialml.xr_runner.interval_ms", 50);

      std::string loadError;
      ModelPackageLoadOptions options{.stripRectifiedVstAccess = !useVst_};
      const bool loaded = SecureMrUtils::LoadModelPackagePipelinesFromFiles(packageRoot, frameworkSession_, {},
                                                                            bundle_, loadError, options);
      if (!loaded) {
        Log::Write(Log::Level::Error,
                   Fmt("pySpatialML XR runner: package load failed from %s: %s", packageRoot.string().c_str(),
                       loadError.c_str()));
        WriteStatus(-1, "error", loadError);
        FinishInit(false);
        return;
      }

      pipelineOrder_ = SplitCsv(GetProp("debug.pyspatialml.xr_runner.pipelines", ""));
      if (pipelineOrder_.empty() && bundle_.manifest.contains("pipelines") && bundle_.manifest["pipelines"].is_array()) {
        for (const auto& entry : bundle_.manifest["pipelines"]) {
          if (entry.is_object() && entry.contains("id") && entry["id"].is_string()) {
            pipelineOrder_.push_back(entry["id"].get<std::string>());
          }
        }
      }
      if (pipelineOrder_.empty()) {
        for (const auto& [id, _] : bundle_.pipelines) {
          pipelineOrder_.push_back(id);
        }
      }
      if (!BindInputs()) {
        WriteStatus(-1, "error", "failed to bind image/raw inputs");
        FinishInit(false);
        return;
      }
      if (loop_) {
        StartReadback();
      } else {
        PrepareReadbackTargets();
      }
      FinishInit(true);
    });
  }

  void RunPipelines() override {
    runnerThread_ = std::thread([this]() {
      {
        std::unique_lock<std::mutex> lock(initMutex_);
        initialized_.wait(lock, [this]() { return initializedReady_; });
      }
      if (!keepRunning_) {
        return;
      }

      do {
        XrSecureMrPipelineRunPICO previousRun = XR_NULL_HANDLE;
        runStartMs_.store(SteadyNowMs(), std::memory_order_release);
        for (const auto& pipelineId : pipelineOrder_) {
          auto it = bundle_.pipelines.find(pipelineId);
          if (it == bundle_.pipelines.end() || it->second.pipeline == nullptr) {
            Log::Write(Log::Level::Error, Fmt("pySpatialML XR runner: unknown pipeline '%s'", pipelineId.c_str()));
            keepRunning_ = false;
            break;
          }
          try {
            previousRun = it->second.pipeline->submit(it->second.submitBindings, previousRun, nullptr);
          } catch (const std::exception& error) {
            Log::Write(Log::Level::Error,
                       Fmt("pySpatialML XR runner: pipeline '%s' submission failed: %s", pipelineId.c_str(),
                           error.what()));
            WriteStatus(-1, "error", error.what());
            keepRunning_ = false;
            return;
          }
        }
        ++iteration_;
        WriteStatus(-1, loop_ ? "running" : "submitted");
        Log::Write(Log::Level::Info, Fmt("pySpatialML XR runner: iteration %d submitted", iteration_.load()));
        if (!loop_) {
          // Pipeline submission is asynchronous. Give the final submitted run
          // enough time to publish its global outputs before the first one-shot
          // readback request snapshots them.
          std::this_thread::sleep_for(kOneShotReadbackDelay);
          StartPreparedReadback();
        }
        if (loop_ && keepRunning_) {
          std::this_thread::sleep_for(std::chrono::milliseconds(std::max(intervalMs_, 1)));
        }
      } while (loop_ && keepRunning_);

      if (!loop_) {
        keepRunning_ = false;
      }
    });
  }

  [[nodiscard]] bool LoadingFinished() const override {
    return initializedReady_;
  }

  void Tick() override {
    // One-shot CLI runs leave the activity alive after writing outputs. Tearing
    // down the OpenMR session immediately after display pipelines can crash the
    // backend service on current device builds.
    if (!loop_ && allOutputsWritten_.load(std::memory_order_acquire) &&
        !readbackStopped_.exchange(true, std::memory_order_acq_rel) && readback_) {
      readback_->Stop();
    }
  }

 private:
  void FinishInit(bool success) {
    keepRunning_ = success;
    {
      std::lock_guard<std::mutex> lock(initMutex_);
      initializedReady_ = true;
    }
    initialized_.notify_all();
  }

  bool EnsureImageModeBindings() {
    if (useVst_) {
      return true;
    }

    for (auto& [pipelineId, package] : bundle_.pipelines) {
      for (const auto& [tensorName, pipelineTensor] : package.tensorMap) {
        if (pipelineTensor == nullptr || package.globalTensorMap.find(tensorName) != package.globalTensorMap.end()) {
          continue;
        }
        auto attrVariant = pipelineTensor->getAttribute();
        auto* attr = std::get_if<TensorAttribute>(&attrVariant);
        if (attr == nullptr) {
          continue;
        }
        if (!IsImageModeExternalTensor(tensorName, *attr)) {
          continue;
        }
        auto globalTensor = std::make_shared<GlobalTensor>(frameworkSession_, *attr);
        package.globalTensorMap[tensorName] = globalTensor;
        package.submitBindings[pipelineTensor] = globalTensor;
        Log::Write(Log::Level::Info,
                   Fmt("pySpatialML XR runner: created image-mode binding %s for pipeline %s shape=%s channels=%d",
                       tensorName.c_str(), pipelineId.c_str(), JoinShape(attr->dimensions).c_str(), attr->channels));
      }
    }
    return true;
  }

  bool BindInputs() {
    if (useVst_) {
      Log::Write(Log::Level::Info, "pySpatialML XR runner: using device VST inputs");
      return true;
    }
    if (inputPath_.empty()) {
      Log::Write(Log::Level::Error, "pySpatialML XR runner: image/raw input mode requires input path");
      return false;
    }

    if (!EnsureImageModeBindings()) {
      return false;
    }

    std::unordered_map<std::string, std::vector<uint8_t>> inputDataCache;
    for (auto& [pipelineId, package] : bundle_.pipelines) {
      std::vector<std::string> inputNames = package.inputs;
      if (inputNames.empty()) {
        const std::set<std::string> outputNames(package.outputs.begin(), package.outputs.end());
        for (const auto& [tensorName, pipelineTensor] : package.tensorMap) {
          if (pipelineTensor == nullptr || outputNames.find(tensorName) != outputNames.end()) {
            continue;
          }
          const auto attrVariant = pipelineTensor->getAttribute();
          const auto* attr = std::get_if<TensorAttribute>(&attrVariant);
          if (attr == nullptr || !IsImageModeExternalTensor(tensorName, *attr) ||
              package.globalTensorMap.find(tensorName) == package.globalTensorMap.end()) {
            continue;
          }
          inputNames.push_back(tensorName);
        }
      }

      for (const auto& inputName : inputNames) {
        const auto globalIt = package.globalTensorMap.find(inputName);
        if (globalIt == package.globalTensorMap.end()) {
          Log::Write(Log::Level::Warning,
                     Fmt("pySpatialML XR runner: input %s for pipeline %s has no global binding",
                         inputName.c_str(), pipelineId.c_str()));
          continue;
        }
        const auto& globalTensor = globalIt->second;
        if (globalTensor == nullptr) {
          continue;
        }
        auto attrVariant = globalTensor->getAttribute();
        auto* attr = std::get_if<TensorAttribute>(&attrVariant);
        if (attr == nullptr) {
          continue;
        }

        std::vector<uint8_t> data;
        std::filesystem::path path;
        if (IsDefaultableTensorName(inputName, *attr)) {
          data = DefaultTensorData(inputName, *attr);
          path = "<default>";
        } else {
          path = ResolveInputPath(inputPath_, inputName);
        }
        if (!data.empty()) {
          // The default tensor was populated above.
        } else if (!path.empty() && IsImagePath(path)) {
          if (!IsImageTensor(*attr)) {
            continue;
          }
          const std::string cacheKey = path.string() + "|" + JoinShape(attr->dimensions) + "|" +
                                       std::to_string(attr->channels) + "|" +
                                       std::to_string(static_cast<int>(attr->dataType));
          if (const auto cacheIt = inputDataCache.find(cacheKey); cacheIt != inputDataCache.end()) {
            data = cacheIt->second;
          } else {
            data = ConvertImageToTensor(path, *attr);
            inputDataCache.emplace(cacheKey, data);
          }
        } else if (!path.empty()) {
          ReadFile(path, data);
        } else if (std::filesystem::is_directory(std::filesystem::path(inputPath_)) && IsImageTensor(*attr)) {
          path = ResolveDefaultInputPath(inputPath_);
          if (path.empty()) {
            continue;
          }
          const std::string cacheKey = path.string() + "|" + JoinShape(attr->dimensions) + "|" +
                                       std::to_string(attr->channels) + "|" +
                                       std::to_string(static_cast<int>(attr->dataType));
          if (const auto cacheIt = inputDataCache.find(cacheKey); cacheIt != inputDataCache.end()) {
            data = cacheIt->second;
          } else {
            data = ConvertImageToTensor(path, *attr);
            inputDataCache.emplace(cacheKey, data);
          }
        } else if (!std::filesystem::is_directory(std::filesystem::path(inputPath_)) && IsImageTensor(*attr)) {
          path = inputPath_;
          const std::string cacheKey = path.string() + "|" + JoinShape(attr->dimensions) + "|" +
                                       std::to_string(attr->channels) + "|" +
                                       std::to_string(static_cast<int>(attr->dataType));
          if (const auto cacheIt = inputDataCache.find(cacheKey); cacheIt != inputDataCache.end()) {
            data = cacheIt->second;
          } else {
            data = ConvertImageToTensor(path, *attr);
            inputDataCache.emplace(cacheKey, data);
          }
        } else {
          continue;
        }
        if (data.empty()) {
          Log::Write(Log::Level::Error,
                     Fmt("pySpatialML XR runner: failed to load input %s from %s", inputName.c_str(),
                         path.string().c_str()));
          return false;
        }
        const size_t expectedBytes = TensorByteSize(*attr);
        if (expectedBytes != 0 && data.size() != expectedBytes) {
          Log::Write(Log::Level::Error,
                     Fmt("pySpatialML XR runner: input %s has %zu bytes from %s but tensor expects %zu bytes",
                         inputName.c_str(), data.size(), path.string().c_str(), expectedBytes));
          return false;
        }
        globalTensor->setData(reinterpret_cast<int8_t*>(data.data()), data.size());
        Log::Write(Log::Level::Info,
                   Fmt("pySpatialML XR runner: bound input %s for pipeline %s from %s shape=%s channels=%d",
                       inputName.c_str(), pipelineId.c_str(), path.string().c_str(), JoinShape(attr->dimensions).c_str(),
                       attr->channels));
      }
    }
    return true;
  }

  void PrepareReadbackTargets() {
    readbackTargets_.clear();
    for (const auto& pipelineId : pipelineOrder_) {
      const auto packageIt = bundle_.pipelines.find(pipelineId);
      if (packageIt == bundle_.pipelines.end()) {
        continue;
      }
      const auto& package = packageIt->second;
      std::set<std::string> targetNames;
      std::set<std::string> outputNames(package.outputs.begin(), package.outputs.end());
      for (const auto& outputName : package.outputs) {
        AddReadbackTarget(package, pipelineId, outputName, targetNames, true);
      }
      if (dumpAll_) {
        for (const auto& inputName : package.inputs) {
          AddReadbackTarget(package, pipelineId, inputName, targetNames, outputNames.find(inputName) != outputNames.end());
        }
        if (!package.detectionTensor.empty()) {
          AddReadbackTarget(package, pipelineId, package.detectionTensor, targetNames,
                            outputNames.find(package.detectionTensor) != outputNames.end());
        }
      }
    }
    expectedOutputCount_ = readbackTargets_.size();
  }

  void AddReadbackTarget(const ModelPackagePipeline& package, const std::string& pipelineId,
                         const std::string& tensorName, std::set<std::string>& targetNames, bool isOutput) {
    if (!targetNames.insert(tensorName).second) {
      return;
    }
    const auto tensorIt = package.globalTensorMap.find(tensorName);
    if (tensorIt == package.globalTensorMap.end() || tensorIt->second == nullptr) {
      return;
    }
    if (!std::holds_alternative<TensorAttribute>(tensorIt->second->getAttribute())) {
      return;
    }
    readbackTargets_.push_back(TensorReadback::Target{
        .tensor = tensorIt->second,
        .callback = [this, pipelineId, tensorName, isOutput](TensorReadbackResult&& result) {
              const std::filesystem::path path =
                  std::filesystem::path(outputDir_) /
                  (pipelineId + "_" + tensorName + "_" + std::to_string(iteration_.load()) + ".bin");
              if (!loop_) {
                std::lock_guard<std::mutex> lock(outputMutex_);
                if (writtenOutputNames_.find(path.filename().string()) != writtenOutputNames_.end()) {
                  return;
                }
              }
              if (!WriteFile(path, result.data)) {
                Log::Write(Log::Level::Error,
                           Fmt("pySpatialML XR runner: failed to write readback %s", path.string().c_str()));
                return;
              }
              Log::Write(Log::Level::Info,
                         Fmt("pySpatialML XR runner: wrote readback %s bytes=%zu shape=%s channels=%d dtype=%d",
                             path.string().c_str(), result.data.size(), JoinShape(result.dimensions).c_str(),
                             result.channels, static_cast<int>(result.dataType)));
              if (!loop_) {
                {
                  std::lock_guard<std::mutex> lock(outputMutex_);
                  writtenOutputNames_.insert(path.filename().string());
                  outputMetadata_.push_back(OutputMetadata{
                      .filename = path.filename().string(),
                      .tensorName = tensorName,
                      .pipelineId = pipelineId,
                      .dimensions = result.dimensions,
                      .channels = result.channels,
                      .dataType = static_cast<int>(result.dataType),
                      .dtype = DataTypeName(result.dataType),
                      .bytes = result.data.size(),
                      .isOutput = isOutput});
                }
                MarkOutputCompleteIfReady();
              }
            },
            .name = pipelineId + "." + tensorName});
  }

  void StartReadback() {
    PrepareReadbackTargets();
    StartPreparedReadback();
  }

  void StartPreparedReadback() {
    if (readbackStarted_.exchange(true, std::memory_order_acq_rel)) {
      return;
    }
    if (readbackTargets_.empty()) {
      Log::Write(Log::Level::Warning, "pySpatialML XR runner: no output tensors registered for readback");
      expectedOutputCount_ = 0;
      allOutputsWritten_.store(true, std::memory_order_release);
      WriteStatus(TotalElapsedMs(), "complete");
      return;
    }
    readback_ = std::make_unique<TensorReadback>(frameworkSession_, std::move(readbackTargets_));
    readback_->Start();
  }

  void MarkOutputCompleteIfReady() {
    size_t writtenCount = 0;
    {
      std::lock_guard<std::mutex> lock(outputMutex_);
      writtenCount = writtenOutputNames_.size();
    }
    const bool complete = expectedOutputCount_ > 0 && writtenCount >= expectedOutputCount_;
    if (!complete) {
      return;
    }
    if (allOutputsWritten_.exchange(true, std::memory_order_acq_rel)) {
      return;
    }
    const long long totalElapsedMs = TotalElapsedMs();
    WriteStatus(totalElapsedMs, "complete");
    Log::Write(Log::Level::Info,
               Fmt("pySpatialML XR runner: completed one-shot output readback count=%zu total=%lld ms", writtenCount,
                   static_cast<long long>(totalElapsedMs)));
  }

  long long TotalElapsedMs() const {
    const long long startMs = runStartMs_.load(std::memory_order_acquire);
    if (startMs <= 0) {
      return -1;
    }
    return std::max<long long>(SteadyNowMs() - startMs, 0);
  }

  void WriteStatus(long long totalElapsedMs, const char* state, const std::string& error = {}) const {
    std::filesystem::create_directories(outputDir_);
    std::ofstream status(std::filesystem::path(outputDir_) / "status.json", std::ios::trunc);
    if (!status) {
      return;
    }
    status << "{\n";
    status << "  \"iteration\": " << iteration_.load() << ",\n";
    status << "  \"state\": \"" << state << "\",\n";
    status << "  \"error\": ";
    if (error.empty()) {
      status << "null";
    } else {
      status << "\"" << JsonEscape(error) << "\"";
    }
    status << ",\n";
    status << "  \"total_elapsed_ms\": ";
    if (totalElapsedMs >= 0) {
      status << totalElapsedMs;
    } else {
      status << "null";
    }
    status << ",\n";
    status << "  \"loop\": " << (loop_ ? "true" : "false") << ",\n";
    status << "  \"use_vst\": " << (useVst_ ? "true" : "false") << ",\n";
    size_t writtenCount = 0;
    {
      std::lock_guard<std::mutex> lock(outputMutex_);
      writtenCount = writtenOutputNames_.size();
    }
    status << "  \"outputs_expected\": " << expectedOutputCount_ << ",\n";
    status << "  \"outputs_written\": " << writtenCount << ",\n";
    status << "  \"outputs_metadata\": [";
    {
      std::lock_guard<std::mutex> lock(outputMutex_);
      for (size_t idx = 0; idx < outputMetadata_.size(); ++idx) {
        const auto& item = outputMetadata_[idx];
        if (idx != 0) {
          status << ", ";
        }
        status << "{";
        status << "\"file\":\"" << item.filename << "\", ";
        status << "\"pipeline\":\"" << item.pipelineId << "\", ";
        status << "\"tensor\":\"" << item.tensorName << "\", ";
        status << "\"dtype\":\"" << item.dtype << "\", ";
        status << "\"data_type\":" << item.dataType << ", ";
        status << "\"channels\":" << item.channels << ", ";
        status << "\"bytes\":" << item.bytes << ", ";
        status << "\"is_output\":" << (item.isOutput ? "true" : "false") << ", ";
        status << "\"shape\":[";
        for (size_t dimIdx = 0; dimIdx < item.dimensions.size(); ++dimIdx) {
          if (dimIdx != 0) {
            status << ", ";
          }
          status << item.dimensions[dimIdx];
        }
        status << "]}";
      }
    }
    status << "],\n";
    status << "  \"runtime_modes\": [";
    if (bundle_.manifest.contains("runtime") && bundle_.manifest["runtime"].is_object() &&
        bundle_.manifest["runtime"].contains("supported_modes") &&
        bundle_.manifest["runtime"]["supported_modes"].is_array()) {
      const auto& modes = bundle_.manifest["runtime"]["supported_modes"];
      bool first = true;
      for (const auto& mode : modes) {
        if (!mode.is_string()) {
          continue;
        }
        if (!first) {
          status << ", ";
        }
        status << "\"" << mode.get<std::string>() << "\"";
        first = false;
      }
    }
    status << "],\n";
    status << "  \"pipelines\": [";
    for (size_t idx = 0; idx < pipelineOrder_.size(); ++idx) {
      if (idx != 0) {
        status << ", ";
      }
      status << "\"" << pipelineOrder_[idx] << "\"";
    }
    status << "]\n";
    status << "}\n";
  }

  XrInstance xrInstance_ = XR_NULL_HANDLE;
  XrSession xrSession_ = XR_NULL_HANDLE;
  std::shared_ptr<FrameworkSession> frameworkSession_;
  ModelPackagePipelineBundle bundle_;
  std::unique_ptr<TensorReadback> readback_;
  std::vector<TensorReadback::Target> readbackTargets_;
  std::vector<std::string> pipelineOrder_;
  std::string outputDir_;
  std::string inputPath_;
  bool useVst_ = true;
  bool loop_ = false;
  bool dumpAll_ = false;
  int intervalMs_ = 50;
  std::thread initializerThread_;
  std::thread runnerThread_;
  std::condition_variable initialized_;
  std::mutex initMutex_;
  mutable std::mutex outputMutex_;
  std::set<std::string> writtenOutputNames_;
  std::vector<OutputMetadata> outputMetadata_;
  size_t expectedOutputCount_ = 0;
  std::atomic<bool> keepRunning_{true};
  std::atomic<bool> allOutputsWritten_{false};
  std::atomic<bool> readbackStopped_{false};
  std::atomic<bool> readbackStarted_{false};
  std::atomic<long long> runStartMs_{0};
  std::atomic<int> iteration_{0};
  bool initializedReady_ = false;
};

std::shared_ptr<ISecureMR> CreateSecureMrProgram(const XrInstance& instance, const XrSession& session) {
  return std::make_shared<XrPipelineRunnerProgram>(instance, session);
}

}  // namespace SecureMR
