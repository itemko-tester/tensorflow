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
#include "tensorflow/contrib/ucx/grpc_ucx_client.h"
#include "tensorflow/contrib/ucx/ucx_service.pb.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_worker_cache.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/lib/core/status.h"
#include <thread>

namespace tensorflow {

void RequestInit(void* request);
extern size_t request_size;

UcxMgr::UcxMgr(const WorkerEnv* const worker_env,
               GrpcChannelCache* const channel_cache)
    : worker_env_(worker_env), channel_cache_(channel_cache) {
  ucp_params_t ucp_params;
  ucp_worker_params_t worker_params;
  ucp_config_t* config;
  ucs_status_t status;
  ucp_address_t* local_addr_;
  size_t local_addr_len_;

  /* UCP initialization */
  status = ucp_config_read(NULL, NULL, &config);
  CHECK(status == UCS_OK) << "ucp_config_read failed with status: " << status;

  ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES |
                          UCP_PARAM_FIELD_REQUEST_SIZE |
                          UCP_PARAM_FIELD_REQUEST_INIT;
  ucp_params.features = UCP_FEATURE_TAG;
  ucp_params.request_size = request_size;
  ucp_params.request_init = RequestInit;

  status = ucp_init(&ucp_params, config, &ucp_context_);
  CHECK(status == UCS_OK) << "ucp_init failed with status: " << status;
  ucp_config_print(config, stdout, NULL, UCS_CONFIG_PRINT_CONFIG);

  ucp_config_release(config);

  memset(&worker_params, 0, sizeof(worker_params));
  worker_params.field_mask = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
  worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;

  status = ucp_worker_create(ucp_context_, &worker_params, &ucp_worker_);
  CHECK(status == UCS_OK) << "worker_create failed";

  local_worker_ = worker_env_->session_mgr->LegacySession()->worker_name;
  std::vector<string> workers;
  worker_env_->session_mgr->LegacySession()->worker_cache->ListWorkers(
      &workers);
  num_remote_workers_ = workers.size() - 1;
  ucp_worker_get_address(ucp_worker_, &local_addr_, &local_addr_len_);

  ucx_addr_ = std::move(UcxAddress(local_addr_, local_addr_len_));
  for (size_t i = 0; i < workers.size(); i++) {
    if (local_worker_.compare(workers[i]) != 0) {
      channel_table_.insert(
          {workers[i], new UcxChannel(ucx_addr_, ucp_worker_)});
    }
  }
  ucx_adapter_ = new UcxAdapter(ucp_worker_, GetMutex());
}

// Find a channel via the given name.
// Args:
//   name: peer name, e.g. worker1
// Returns
//   channel object that is connected to the named peer.
UcxChannel* UcxMgr::FindChannel(const string& name) {
  ChannelTable::iterator iter = channel_table_.find(name);
  CHECK(iter != channel_table_.end());
  return iter->second;
}

void UcxMgr::SetupChannels() {
  for (const auto& p : channel_table_) {
    string worker_name = p.first;
    LOG(INFO) << "connecting to remote node " << worker_name;
    UcxChannel* uc = p.second;
    GetRemoteWorkerAddressRequest req;
    GetRemoteWorkerAddressResponse resp;
    // get the channel cache
    SharedGrpcChannelPtr client_channel =
        channel_cache_->FindWorkerChannel(worker_name);
    GrpcUcxClient* client = new GrpcUcxClient(client_channel);
    // setting up request
    req.set_host_name(local_worker_);
    WorkerAddress* addr_info = req.mutable_addr();
    addr_info->set_addr_len(ucx_addr_.get_size());
    addr_info->set_worker_addr(ucx_addr_.get_addr(), ucx_addr_.get_size());

    // synchronous call
    Status s = client->GetRemoteWorkerAddress(&req, &resp);
    // save obtained remote addresses
    // connect to the remote channel
    if (s.ok()) {
      CHECK(worker_name.compare(resp.host_name()) == 0);
      UcxAddress ra(resp.addr().worker_addr(), resp.addr().addr_len());
      uc->SetRemoteAddress(ra);
      uc->Connect();
    } else {
      LOG(ERROR) << "GetRemoteWorkerAddress failed with error: "
                 << s.error_message();
    }
    delete client;
  }
}

UcxMgr::~UcxMgr() {
  ucp_worker_release_address(ucp_worker_, ucx_addr_.get_addr());
  ucp_worker_destroy(ucp_worker_);
  ucp_cleanup(ucp_context_);
  delete ucx_adapter_;
}

UcxAdapter::~UcxAdapter() { progress_thread_.reset(); }

void UcxAdapter::UcxProgress() {
  progress_thread_.reset(Env::Default()->StartThread(
      ThreadOptions(), "UcxAdapterProgressThread", [this] {
        while (1) {
          mtx_.lock();
          while (ucp_worker_progress(ucp_worker_)){};
          mtx_.unlock();
          std::this_thread::yield();
        }
      }));
}

}  // end namespace tensorflow

#endif  // TENSORFLOW_USE_UCX
