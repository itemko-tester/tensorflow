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

#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"

namespace tensorflow {

namespace {
// static utility function
RendezvousMgrInterface* NewUCXRendezvousMgr(const WorkerEnv* env) {
  return new UcxRendezvousMgr(env);
}

}  // namespace

UCXServer::UCXServer(const ServerDef& server_def, Env* env)
    : GrpcServer(server_def, env), ucx_state_(DISCONNECTED) {}

UCXServer::~UCXServer() {
  TF_CHECK_OK(Stop());
  TF_CHECK_OK(Join());
  delete ucx_mgr_;
  delete ucx_service_;
  delete channel_cache_;
}

Status UCXServer::ChannelCacheFactory(const ServerDef& server_def,
                                      GrpcChannelCache** channel_cache) {
  string name_prefix =
      strings::StrCat("/job:", server_def.job_name(), "/replica:0", "/task:",
                      server_def.task_index());

  GrpcChannelSpec channel_spec;
  TF_RETURN_IF_ERROR(ParseChannelSpec(server_def, &channel_spec));

  *channel_cache =
      NewGrpcChannelCache(channel_spec, GetChannelCreationFunction());

  const string host_port = (*channel_cache)->TranslateTask(name_prefix);
  int requested_port;

  if (!strings::safe_strto32(str_util::Split(host_port, ':')[1],
                             &requested_port)) {
    return errors::Internal("Could not parse port for local server from \"",
                            (*channel_cache)->TranslateTask(name_prefix),
                            "\".");
  }
  if (requested_port != bound_port()) {
    return errors::InvalidArgument("Requested port ", requested_port,
                                   " differs from expected port ",
                                   bound_port());
  }

  return Status::OK();
}

Status UCXServer::Init(ServiceInitFunction service_func,
                       RendezvousMgrCreationFunction rendezvous_mgr_func) {
  Status s = GrpcServer::Init(service_func, rendezvous_mgr_func);
  TF_CHECK_OK(s) << "GrpcServer::Init failed with status: " << s.ToString();
  {
    mutex_lock l(mu_);
    CHECK_EQ(ucx_state_, DISCONNECTED);
    CHECK(ChannelCacheFactory(server_def(), &channel_cache_).ok());
    ucx_mgr_ = new UcxMgr(worker_env(), channel_cache_);
    // set ucx_mgr for ucx_service and ucx_rendezvous_mgr
    ucx_service_->SetUcxMgr(ucx_mgr_);
    dynamic_cast<UcxRendezvousMgr*>(worker_env()->rendezvous_mgr)
        ->SetUcxMgr(ucx_mgr_);
  }
  return s;
}

Status UCXServer::Start() {
  Status s = GrpcServer::Start();
  {
    mutex_lock l(mu_);
    if (ucx_state_ == DISCONNECTED) {
      // ucx_thread needs to be initiated
      // before ucx_mgr sets up the ucx channels.
      ucx_thread_.reset(worker_env()->env->StartThread(
          ThreadOptions(), "TF_ucx_service",
          [this] { ucx_service_->HandleRPCsLoop(); }));
      ucx_mgr_->SetupChannels();
      ucx_mgr_->GetAdapter()->UcxProgress();
      ucx_state_ = CONNECTED;
    }
  }
  return s;
}

Status UCXServer::Join() {
  Status s = GrpcServer::Join();
  {
    mutex_lock l(mu_);
    if (ucx_state_ == CONNECTED) {
      ucx_state_ = DISCONNECTED;
      ucx_thread_.reset();
    }
  }
  return s;
}

/* static */
Status UCXServer::Create(const ServerDef& server_def, Env* env,
                         std::unique_ptr<ServerInterface>* out_server) {
  std::unique_ptr<UCXServer> ret(new UCXServer(server_def, Env::Default()));
  ServiceInitFunction service_func = [&ret](const WorkerEnv* worker_env,
                                            ::grpc::ServerBuilder* builder) {
    return SetNewUcxService(&ret->ucx_service_, worker_env, builder);
  };
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
