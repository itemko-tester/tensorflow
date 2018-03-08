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

#include "tensorflow/contrib/ucx/grpc_ucx_client.h"

#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

Status GrpcUcxClient::GetRemoteWorkerAddress(
    CallOptions* call_options, const GetRemoteWorkerAddressRequest* request,
    GetRemoteWorkerAddressResponse* response) {
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  SetDeadline(&ctx, call_options->GetTimeout());
  return FromGrpcStatus(
      stub_->GetRemoteWorkerAddress(&ctx, *request, response));
}

Status GrpcUcxClient::GetRemoteWorkerAddress(
    const GetRemoteWorkerAddressRequest* request,
    GetRemoteWorkerAddressResponse* response) {
  CallOptions call_options;
  call_options.SetTimeout(-1);  // no time out
  return GetRemoteWorkerAddress(&call_options, request, response);
}

void GrpcUcxClient::SetDeadline(::grpc::ClientContext* ctx, int64 time_in_ms) {
  if (time_in_ms > 0) {
    ctx->set_deadline(gpr_time_from_millis(time_in_ms, GPR_TIMESPAN));
  }
}

}  // namespace tensorflow
