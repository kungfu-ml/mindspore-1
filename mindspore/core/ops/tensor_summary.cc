/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "ops/tensor_summary.h"
#include <memory>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/primitive_infer_map.h"

namespace mindspore {
namespace ops {
// scalar_summary
namespace {
abstract::ShapePtr TensorSummaryInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  // check
  auto v_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  CheckAndConvertUtils::CheckInteger("v rank", v_shape.size(), kGreaterEqual, 1, prim_name);
  return std::make_shared<abstract::Shape>(ShapeVector(1));
}
}  // namespace
void TensorSummary::set_side_effect_io() { this->AddAttr(kSideEffectIO, MakeValue(true)); }

bool TensorSummary::get_side_effect_io() const {
  auto value_ptr = GetAttr(kSideEffectIO);
  return GetValue<bool>(value_ptr);
}

void TensorSummary::Init() { this->set_side_effect_io(); }

AbstractBasePtr TensorSummaryInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  // check
  CheckAndConvertUtils::CheckSummaryParam(input_args[0], input_args[1], primitive->name());
  return std::make_shared<abstract::AbstractTensor>(kInt32, TensorSummaryInferShape(primitive, input_args));
}
REGISTER_PRIMITIVE_EVAL_IMPL(TensorSummary, prim::kPrimTensorSummary, TensorSummaryInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
