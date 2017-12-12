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

#include "grpc++/alarm.h"
#include "grpc++/grpc++.h"
#include "grpc++/server_builder.h"

#include "tensorflow/contrib/ucx/grpc_ucx_service.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_util.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"

namespace tensorflow {

GrpcUcxService::GrpcUcxService(const WorkerEnv* worker_env,
                                   ::grpc::ServerBuilder* builder)
    : is_shutdown_(false), worker_env_(worker_env) {
  builder->RegisterService(&ucx_service_);
  cq_ = builder->AddCompletionQueue().release();
}

GrpcUcxService::~GrpcUcxService() {
  delete shutdown_alarm_;
  delete cq_;
}

void GrpcUcxService::Shutdown() {
  bool did_shutdown = false;
  {
    mutex_lock l(shutdown_mu_);
    if (!is_shutdown_) {
      LOG(INFO) << "Shutting down GrpcWorkerService.";
      is_shutdown_ = true;
      did_shutdown = true;
    }
  }
  if (did_shutdown) {
    shutdown_alarm_ =
        new ::grpc::Alarm(cq_, gpr_now(GPR_CLOCK_MONOTONIC), nullptr);
  }
}

// This macro creates a new request for the given RPC method name
// (e.g., `ENQUEUE_REQUEST(GetRemoteAddress, false);`), and enqueues it on
// `this->cq_`.
//
// This macro is invoked one or more times for each RPC method to
// ensure that there are sufficient completion queue entries to
// handle incoming requests without blocking.
//
// The implementation of the request handler for each RPC method
// must ensure that it calls ENQUEUE_REQUEST() for that RPC method,
// to keep accepting new requests.
#define ENQUEUE_REQUEST(method, supports_cancel)                             \
  do {                                                                       \
    mutex_lock l(shutdown_mu_);                                              \
    if (!is_shutdown_) {                                                     \
      Call<GrpcUcxService, grpc::UcxService::AsyncService,               \
           method##Request, method##Response>::                              \
          EnqueueRequest(&ucx_service_, cq_,                               \
                         &grpc::UcxService::AsyncService::Request##method, \
                         &GrpcUcxService::method##Handler,                 \
                         (supports_cancel));                                 \
    }                                                                        \
  } while (0)

// This method blocks forever handling requests from the completion queue.
void GrpcUcxService::HandleRPCsLoop() {
  for (int i = 0; i < 10; ++i) {
    ENQUEUE_REQUEST(GetRemoteAddress, false);
  }

  void* tag;
  bool ok;

  while (cq_->Next(&tag, &ok)) {
    UntypedCall<GrpcUcxService>::Tag* callback_tag =
        static_cast<UntypedCall<GrpcUcxService>::Tag*>(tag);
    if (callback_tag) {
      callback_tag->OnCompleted(this, ok);
    } else {
      cq_->Shutdown();
    }
  }
}

void GrpcUcxService::GetRemoteAddressHandler(
    WorkerCall<GetRemoteAddressRequest, GetRemoteAddressResponse>* call) {
  Status s = GetRemoteAddressSync(&call->request, &call->response);
  call->SendResponse(ToGrpcStatus(s));
  ENQUEUE_REQUEST(GetRemoteAddress, false);
}

// Create a GrpcUcxService, then assign it to a given handle.
void SetNewUcxService(GrpcUcxService** handle, const WorkerEnv* worker_env,
                        ::grpc::ServerBuilder* builder) {
  *handle = new GrpcUcxService(worker_env, builder);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_USE_UCX
