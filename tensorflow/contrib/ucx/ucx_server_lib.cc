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

#include "tensorflow/contrib/ucx/ucx_server_lib.h"
#include "tensorflow/contrib/ucx/ucx_rendezvous_mgr.h"

#include <string>
#include <utility>

#include "grpc/support/alloc.h"

#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

namespace {
// static utility function
RendezvousMgrInterface* NewUCXRendezvousMgr(const WorkerEnv* env) {
  // Runtime check to disable the UCX path
  const char* ucxenv = getenv("UCX_DISABLED");
  if (ucxenv && ucxenv[0] == '1') {
    LOG(INFO) << "UCX path disabled by environment variable\n";
    return new RpcRendezvousMgr(env);
  } else {
    return new UcxRendezvousMgr(env);
  }
}

}  // namespace

UCXServer::UCXServer(const ServerDef& server_def, Env* env)
    : GrpcServer(server_def, env) {}

UCXServer::~UCXServer() {
  TF_CHECK_OK(Stop());
  TF_CHECK_OK(Join());
}

Status UCXServer::Init(ServiceInitFunction service_func,
                       RendezvousMgrCreationFunction rendezvous_mgr_func) {
  Status s = GrpcServer::Init(service_func, rendezvous_mgr_func);
  return s;
}

Status UCXServer::Start() {
  Status s = GrpcServer::Start();
  return s;
}

Status UCXServer::Join() {
  Status s = GrpcServer::Join();
  return s;
}

/* static */
Status UCXServer::Create(const ServerDef& server_def, Env* env,
                         std::unique_ptr<ServerInterface>* out_server) {
  std::unique_ptr<UCXServer> ret(new UCXServer(server_def, Env::Default()));
  ServiceInitFunction service_func = nullptr;
  TF_RETURN_IF_ERROR(ret->Init(service_func, NewUCXRendezvousMgr));
  *out_server = std::move(ret);
  return Status::OK();
}

namespace {

class UCXServerFactory : public ServerFactory {
 public:
  bool AcceptsOptions(const ServerDef& server_def) override {
    return server_def.protocol() == "grpc+ucx";
  }

  Status NewServer(const ServerDef& server_def,
                   std::unique_ptr<ServerInterface>* out_server) override {
    return UCXServer::Create(server_def, Env::Default(), out_server);
  }
};

// Registers a `ServerFactory` for `UCXServer` instances.
class UCXServerRegistrar {
 public:
  UCXServerRegistrar() {
    gpr_allocation_functions alloc_fns;
    alloc_fns.malloc_fn = port::Malloc;
    alloc_fns.realloc_fn = port::Realloc;
    alloc_fns.free_fn = port::Free;
    gpr_set_allocation_functions(alloc_fns);
    ServerFactory::Register("UCX_SERVER", new UCXServerFactory());
  }
};
static UCXServerRegistrar registrar;

}  // namespace
}  // namespace tensorflow

#endif  // TENSORFLOW_USE_UCX
