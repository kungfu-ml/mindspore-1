/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include "src/ops/populate/populate_register.h"
#include "nnacl/fp32/unique_fp32.h"

namespace mindspore {
namespace lite {
namespace {
OpParameter *PopulateUniqueParameter(const void *prim) {
  UniqueParameter *unique_param = reinterpret_cast<UniqueParameter *>(malloc(sizeof(UniqueParameter)));
  if (unique_param == nullptr) {
    MS_LOG(ERROR) << "malloc UniqueParameter failed.";
    return nullptr;
  }
  memset(unique_param, 0, sizeof(UniqueParameter));
  auto primitive = static_cast<const schema::Primitive *>(prim);
  unique_param->op_parameter_.type_ = primitive->value_type();
  return reinterpret_cast<OpParameter *>(unique_param);
}
}  // namespace

Registry g_uniqueParameterRegistry(schema::PrimitiveType_Unique, PopulateUniqueParameter, SCHEMA_CUR);

}  // namespace lite
}  // namespace mindspore
