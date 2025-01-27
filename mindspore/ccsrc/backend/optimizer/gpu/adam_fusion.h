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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_ADAM_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_ADAM_FUSION_H_

#include <memory>
#include "backend/optimizer/common/optimizer.h"

namespace mindspore {
namespace opt {
class AdamFusion : public PatternProcessPass {
 public:
  explicit AdamFusion(bool multigraph = true) : PatternProcessPass("adam_fusion", multigraph) {
    beta1_ = std::make_shared<Var>();
    one_sub_beta1_ = std::make_shared<Var>();
    beta2_ = std::make_shared<Var>();
    one_sub_beta2_ = std::make_shared<Var>();
    eps_ = std::make_shared<Var>();
    lr_ = std::make_shared<Var>();
    param_ = std::make_shared<Var>();
    m_ = std::make_shared<Var>();
    v_ = std::make_shared<Var>();
    gradient_ = std::make_shared<Var>();
    u_ = std::make_shared<Var>();
    u2_ = std::make_shared<Var>();
  }
  ~AdamFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  VarPtr beta1_;
  VarPtr one_sub_beta1_;
  VarPtr beta2_;
  VarPtr one_sub_beta2_;
  VarPtr eps_;
  VarPtr lr_;
  VarPtr param_;
  VarPtr m_;
  VarPtr v_;
  VarPtr gradient_;
  VarPtr u_;
  VarPtr u2_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GPU_ADAM_FUSION_H_
