// Smoke tests for the TRT wrappers: construction is lazy, and a missing
// engine degrades to "unavailable" (false + stderr log) instead of crashing.
// No GPU and no real .plan files required.
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "tinynav_cpp/trt/models.hpp"

namespace tinynav::trt {
namespace {

constexpr const char* kBadDir = "/nonexistent/tinynav/models";

TEST(TRTBase, ConstructDoesNotLoad) {
  SuperPointTRT sp(kBadDir);
  EXPECT_FALSE(sp.loaded());
  EXPECT_TRUE(sp.last_error().empty());
  EXPECT_EQ(sp.engine_path(),
            std::string(kBadDir) + "/superpoint_fp16_dynamic_" + engine_arch() + ".plan");
}

TEST(TRTBase, LazyLoadFailureDegrades) {
  SuperPointTRT sp(kBadDir);
  EXPECT_FALSE(sp.available());   // first call attempts the load
  EXPECT_FALSE(sp.loaded());
  EXPECT_FALSE(sp.last_error().empty());
  EXPECT_FALSE(sp.available());   // failure is cached, no retry storm

  TrtOutputMap results;
  EXPECT_FALSE(sp.infer(cv::Mat(8, 8, CV_8UC1, cv::Scalar(0)), results));
}

TEST(Models, AllEnginesDegradeGracefully) {
  cv::Mat gray(16, 16, CV_8UC1, cv::Scalar(128));
  TrtOutputMap results;
  std::vector<float> emb;
  cv::Mat tokens, disp, depth;

  LightGlueTRT lg(kBadDir);
  EXPECT_FALSE(lg.infer(gray, gray, gray, gray, gray, gray, {16, 16}, {16, 16}, results));

  Dinov2TRT dino(kBadDir);
  EXPECT_FALSE(dino.infer(gray, emb));
  EXPECT_FALSE(dino.infer_global_and_patch_tokens(gray, emb, tokens));
  EXPECT_FALSE(dino.infer_patch_tokens(gray, tokens));

  SigLIPImageTRT sig_img(kBadDir);
  EXPECT_FALSE(sig_img.infer(gray, emb));

  FoundationStereoTRT fs(kBadDir);
  EXPECT_FALSE(fs.infer(gray, gray, 0.05, 320.0, disp, depth));

  RetinifyTRT ret(kBadDir);
  EXPECT_FALSE(ret.infer(gray, gray, 0.05, 320.0, disp, depth));
}

TEST(SigLIPText, NoTokenizerMeansUnavailable) {
  // Null tokenizer: the engine is never even attempted, encode() -> false.
  SigLIPTextTRT text(nullptr, kBadDir);
  std::vector<float> emb;
  EXPECT_FALSE(text.available());
  EXPECT_FALSE(text.encode("hello", emb));
  EXPECT_FALSE(text.last_error().empty());

  // Facade without tokenizer: image side also degrades (bad dir here).
  SigLIPTRT siglip(kBadDir);
  EXPECT_FALSE(siglip.encode_text("hello", emb));
  cv::Mat gray(16, 16, CV_8UC1, cv::Scalar(0));
  EXPECT_FALSE(siglip.encode_image(gray, emb));
}

TEST(SigLIPText, PrecomputedTextEncoder) {
  const std::string dir = ::testing::TempDir() + "/tinynav_trt_test_emb";
  std::filesystem::create_directories(dir);

  PrecomputedTextEncoder enc(dir);
  EXPECT_TRUE(enc.available());

  // Write a 3-dim embedding for "chair" (sanitized name is the text itself).
  {
    const float vals[3] = {1.0f, 2.0f, 3.0f};
    std::ofstream f(dir + "/chair.f32", std::ios::binary);
    f.write(reinterpret_cast<const char*>(vals), sizeof(vals));
  }
  std::vector<float> emb;
  EXPECT_TRUE(enc.encode("chair", emb));
  ASSERT_EQ(emb.size(), 3u);
  EXPECT_FLOAT_EQ(emb[0], 1.0f);
  EXPECT_FLOAT_EQ(emb[1], 2.0f);
  EXPECT_FLOAT_EQ(emb[2], 3.0f);

  // Missing key degrades to false; sanitize maps "a b" -> "a_b".
  EXPECT_FALSE(enc.encode("a b", emb));
  EXPECT_EQ(PrecomputedTextEncoder::sanitize("a b/c"), "a_b_c");

  // Injected into the facade, the text side works with no engine at all.
  SigLIPTRT siglip(kBadDir);
  siglip.set_text_encoder(std::make_shared<PrecomputedTextEncoder>(dir));
  EXPECT_TRUE(siglip.encode_text("chair", emb));

  PrecomputedTextEncoder missing(dir + "/does-not-exist");
  EXPECT_FALSE(missing.available());
}

TEST(DisparityToDepth, Basic) {
  cv::Mat disp(1, 4, CV_32F);
  disp.at<float>(0) = 2.0f;
  disp.at<float>(1) = 0.0f;                                    // zero -> 0
  disp.at<float>(2) = -1.0f;                                   // negative -> 0
  disp.at<float>(3) = std::numeric_limits<float>::infinity();  // inf -> 0
  cv::Mat depth = disparity_to_depth(disp, 0.05, 320.0);
  ASSERT_EQ(depth.type(), CV_32F);
  EXPECT_FLOAT_EQ(depth.at<float>(0), 8.0f);
  EXPECT_FLOAT_EQ(depth.at<float>(1), 0.0f);
  EXPECT_FLOAT_EQ(depth.at<float>(2), 0.0f);
  EXPECT_FLOAT_EQ(depth.at<float>(3), 0.0f);

  EXPECT_THROW(disparity_to_depth(disp, 0.0, 320.0), std::invalid_argument);
  EXPECT_THROW(disparity_to_depth(disp, 0.05, -1.0), std::invalid_argument);
}

}  // namespace
}  // namespace tinynav::trt
