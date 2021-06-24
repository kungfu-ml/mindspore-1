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

#include "ops/layer_norm.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
void LayerNorm::Init(const int64_t begin_norm_axis, const int64_t begin_params_axis, const float epsilon) {
  this->set_begin_norm_axis(begin_norm_axis);
  this->set_begin_params_axis(begin_params_axis);
  this->set_epsilon(epsilon);
}
void LayerNorm::set_begin_norm_axis(const int64_t begin_norm_axis) {
  this->AddAttr(kBeginNormAxis, MakeValue(begin_norm_axis));
}
void LayerNorm::set_begin_params_axis(const int64_t begin_params_axis) {
  this->AddAttr(kBeginParamsAxis, MakeValue(begin_params_axis));
}
void LayerNorm::set_epsilon(const float epsilon) { this->AddAttr(kEpsilon, MakeValue(epsilon)); }

int64_t LayerNorm::get_begin_norm_axis() const {
  auto value_ptr = this->GetAttr(kBeginNormAxis);
  return GetValue<int64_t>(value_ptr);
}
int64_t LayerNorm::get_begin_params_axis() const {
  auto value_ptr = this->GetAttr(kBeginParamsAxis);
  return GetValue<int64_t>(value_ptr);
}
float LayerNorm::get_epsilon() const {
  auto value_ptr = this->GetAttr(kEpsilon);
  return GetValue<float>(value_ptr);
}
REGISTER_PRIMITIVE_C(kNameLayerNorm, LayerNorm);
}  // namespace ops
}  // namespace mindspore
