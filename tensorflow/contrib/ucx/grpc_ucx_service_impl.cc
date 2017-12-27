/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/contrib/ucx/grpc_ucx_service_impl.h"

#include "grpc++/impl/codegen/async_stream.h"
#include "grpc++/impl/codegen/async_unary_call.h"
#include "grpc++/impl/codegen/channel_interface.h"
#include "grpc++/impl/codegen/client_unary_call.h"
#include "grpc++/impl/codegen/method_handler_impl.h"
#include "grpc++/impl/codegen/rpc_service_method.h"
#include "grpc++/impl/codegen/service_type.h"
#include "grpc++/impl/codegen/sync_stream.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

namespace grpc {

static const char* grpcUcxService_method_names[] = {
    "/tensorflow.UcxService/GetRemoteWorkerAddress",
};

std::unique_ptr<UcxService::Stub> UcxService::NewStub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel,
    const ::grpc::StubOptions& options) {
  std::unique_ptr<UcxService::Stub> stub(new UcxService::Stub(channel));
  return stub;
}

UcxService::Stub::Stub(
    const std::shared_ptr< ::grpc::ChannelInterface>& channel)
    : channel_(channel),
      rpcmethod_GetRemoteWorkerAddress_(grpcUcxService_method_names[0],
                                        ::grpc::internal::RpcMethod::NORMAL_RPC,
                                        channel) {}

::grpc::Status UcxService::Stub::GetRemoteWorkerAddress(
    ::grpc::ClientContext* context,
    const GetRemoteWorkerAddressRequest& request,
    GetRemoteWorkerAddressResponse* response) {
  if (!context) LOG(INFO) << "Context is null!";
  return ::grpc::internal::BlockingUnaryCall(channel_.get(),
                                             rpcmethod_GetRemoteWorkerAddress_,
                                             context, request, response);
}

UcxService::AsyncService::AsyncService() {
  for (int i = 0; i < 1; ++i) {
    AddMethod(new ::grpc::internal::RpcServiceMethod(
        grpcUcxService_method_names[i], ::grpc::internal::RpcMethod::NORMAL_RPC,
        nullptr));
    ::grpc::Service::MarkMethodAsync(i);
  }
}

UcxService::AsyncService::~AsyncService() {}

}  // namespace grpc

}  // namespace tensorflow
