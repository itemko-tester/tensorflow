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

#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_UCX_GRPC_UCX_SERVICE_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_UCX_GRPC_UCX_SERVICE_H_

#ifndef TENSORFLOW_USE_UCX

//#include "tensorflow/contrib/ucx/grpc_ucx_service_impl.h"
#include "tensorflow/contrib/ucx/ucx_mgr.h"
#include "tensorflow/contrib/ucx/ucx_service.pb.h"
#include "tensorflow/core/distributed_runtime/rpc/async_service_interface.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_call.h"
#include "tensorflow/core/lib/core/refcount.h"

namespace grpc {
class ServerBuilder;
class ServerCompletionQueue;
class Alarm;
}  // namespace grpc

namespace tensorflow {

class GrpcUcxService : public AsyncServiceInterface {
 public:
  GrpcUcxService(const WorkerEnv* worker_env, ::grpc::ServerBuilder* builder);
  ~GrpcUcxService();
  void HandleRPCsLoop() override;
  void Shutdown() override;
  void SetUcxMgr(UcxMgr* ucx_mgr) { ucx_mgr_ = ucx_mgr; }

 private:
  template <class RequestMessage, class ResponseMessage>
  using WorkerCall = Call<GrpcUcxService, grpc::UcxService::AsyncService,
                          RequestMessage, ResponseMessage>;
  void GetRemoteWorkerAddressHandler(WorkerCall<
      GetRemoteWorkerAddressRequest, GetRemoteWorkerAddressResponse>* call);
  Status GetRemoteWorkerAddressSync(
      const GetRemoteWorkerAddressRequest* request,
      GetRemoteWorkerAddressResponse* response);

  ::grpc::ServerCompletionQueue* cq_;
  grpc::UcxService::AsyncService ucx_service_;
  mutex shutdown_mu_;
  bool is_shutdown_ GUARDED_BY(shutdown_mu_);
  ::grpc::Alarm* shutdown_alarm_;
  // not owned
  ucxMgr* ucx_mgr_;
  const WorkerEnv* const worker_env_;

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcUcxService);
};

// Create a GrpcUcxService, then assign it to a given handle.
void SetNewUcxService(GrpcUcxService** handle, const WorkerEnv* worker_env,
                      ::grpc::ServerBuilder* builder);

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_UCX
#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_UCX_GRPC_UCX_SERVICE_H_
