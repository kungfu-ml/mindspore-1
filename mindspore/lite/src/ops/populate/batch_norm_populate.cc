/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "src/ops/batch_norm.h"
#include "src/ops/primitive_c.h"
#include "src/ops/populate/populate_register.h"
#include "nnacl/batchnorm_parameter.h"
using mindspore::schema::PrimitiveType_BatchNorm;

namespace mindspore {
namespace lite {
OpParameter *PopulateBatchNorm(const mindspore::lite::PrimitiveC *primitive) {
  const auto param =
    reinterpret_cast<mindspore::lite::BatchNorm *>(const_cast<mindspore::lite::PrimitiveC *>(primitive));
  BatchNormParameter *batch_norm_param = reinterpret_cast<BatchNormParameter *>(malloc(sizeof(BatchNormParameter)));
  if (batch_norm_param == nullptr) {
    MS_LOG(ERROR) << "malloc BatchNormParameter failed.";
    return nullptr;
  }
  memset(batch_norm_param, 0, sizeof(BatchNormParameter));
  batch_norm_param->op_parameter_.type_ = primitive->Type();
  batch_norm_param->epsilon_ = param->GetEpsilon();
  batch_norm_param->fused_ = false;
  return reinterpret_cast<OpParameter *>(batch_norm_param);
}

REG_POPULATE(PrimitiveType_BatchNorm, PopulateBatchNorm)
}  // namespace lite
}  // namespace mindspore
