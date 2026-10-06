// Thin driver for the unchanged, pinned Px0 C++ ONNX exporter (GPL-3.0).
#include <fstream>
#include <iostream>
#include <string>
#include <zlib.h>
#include "neural/loader.h"
#include "neural/onnx/converter.h"

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  auto net = lczero::LoadWeightsFromFile(argv[1]);
  lczero::WeightsToOnnxConverterOptions options;
  options.opset = 17;
  auto converted = lczero::ConvertWeightsToOnnx(net, options);
  std::ofstream output(argv[2], std::ios::binary);
  output << converted.onnx_model().model();
  return output.good() ? 0 : 5;
}
