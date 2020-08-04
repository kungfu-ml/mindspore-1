/**
 * Copyright 2020 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <chrono>
#include "common/common_test.h"
#include "gtest/gtest.h"
#include "securec.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/include/datasets.h"
#include "minddata/dataset/include/execute.h"
#include "minddata/dataset/util/path.h"

using namespace mindspore::dataset;
using namespace mindspore::dataset::api;
using namespace mindspore;

class MindDataTestEager : public mindspore::Common {
 public:
  MindDataTestEager() {}
};

TEST_F(MindDataTestEager, Test1) {
  std::string in_dir = "/sdcard/data/testPK/data/class1";
  Path base_dir = Path(in_dir);
  MS_LOG(WARNING) << base_dir.toString() << ".";
  if (!base_dir.IsDirectory() || !base_dir.Exists()) {
    MS_LOG(INFO) << "Input dir is not a directory or doesn't exist" << ".";
  }
  auto t_start = std::chrono::high_resolution_clock::now();
  // check if output_dir exists and create it if it does not exist

  // iterate over in dir and create json for all images
  auto dir_it = Path::DirIterator::OpenDirectory(&base_dir); 
  while (dir_it->hasNext()) {
    Path v = dir_it->next();
    MS_LOG(WARNING) << v.toString() << ".";
    std::shared_ptr<tensor::MSTensor> image = std::shared_ptr<tensor::MSTensor>(tensor::DETensor::CreateTensor(v.toString()));
    
    image = Execute(vision::Decode())(image);
    EXPECT_TRUE(image != nullptr);
    image = Execute(vision::Normalize({121.0, 115.0, 100.0}, {70.0, 68.0, 71.0}))(image);
    EXPECT_TRUE(image != nullptr);
    image = Execute(vision::Resize({224, 224}))(image);
    EXPECT_TRUE(image != nullptr);
    EXPECT_TRUE(image->DimensionSize(0) == 224);
    EXPECT_TRUE(image->DimensionSize(1) == 224);
  }
  auto t_end = std::chrono::high_resolution_clock::now();
  double elapsed_time_ms = std::chrono::duration<double, std::milli>(t_end-t_start).count();
  MS_LOG(INFO) << "duration: " << elapsed_time_ms << " ms\n";
}
