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

#ifndef TENSORFLOW_USE_UCX

#include "tensorflow/contrib/ucx/ucx_mgr.h"
#include <vector>
//#include "tensorflow/contrib/ucx/grpc_ucx_client.h"
//#include "tensorflow/contrib/ucx/ucx_service.pb.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

UcxMgr::UcxMgr(const WorkerEnv* const worker_env,
                 GrpcChannelCache* const channel_cache)
    : worker_env_(worker_env) {

}

UcxMgr::~UcxMgr() {

}


}  // end namespace tensorflow

#endif //TENSORFLOW_USE_UCX
