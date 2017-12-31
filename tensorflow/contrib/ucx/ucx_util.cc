/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/contrib/ucx/ucx_util.h"

#include <vector>

#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/hash/hash.h"

namespace tensorflow {

// static
string UcxUtil::CreateTagString(const string& key, const int64 step_id,
                                bool MetaDataFlag) {
  return strings::StrCat(key, ";", step_id, ";", MetaDataFlag);
}

uint64_t UcxUtil::TagHash(const StringPiece& k) {
  return Hash64(k.data(), k.size());
}

uint64_t UcxUtil::CalcTag(const string& key, const int64 step_id,
                          bool MetaDataFlag) {
  string string_tag;
  uint64_t tag;

  string_tag = UcxUtil::CreateTagString(key, step_id, MetaDataFlag);
  tag = UcxUtil::TagHash(string_tag);
  return tag;
}

}  // namespace tensorflow
