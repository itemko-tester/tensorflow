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
  ucp_params_t ucp_params;
  ucp_worker_params_t worker_params;
  ucp_config_t* config;
  ucs_status_t status;

  /* UCP initialization */
  status = ucp_config_read(NULL, NULL, &config);
  CHECK(status == UCS_OK) << "ucp_config_read failed with status: " << status;

  ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES |
                          UCP_PARAM_FIELD_REQUEST_SIZE |
                          UCP_PARAM_FIELD_REQUEST_INIT;
  ucp_params.features = UCP_FEATURE_TAG;
  // ucp_params.request_size    = sizeof(int);
  // ucp_params.request_init    = request_init;

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
  VLOG(2) << "ucx_mgr on local worker: " << local_worker_;
  ucp_worker_get_address(ucp_worker_, &local_addr_, &local_addr_len_);

  for (size_t i = 0; i < workers.size(); i++) {
    if (local_worker_.compare(workers[i]) != 0) {
      channel_table_.insert(
          {workers[i], new UcxChannel(local_addr_, local_addr_len_,
                                      local_worker_, workers[i], ucp_worker_)});
    }
  }
}

UcxMgr::~UcxMgr() {
  ucp_worker_release_address(ucp_worker_, local_addr_);
  ucp_worker_destroy(ucp_worker_);
  ucp_cleanup(ucp_context_);
}

}  // end namespace tensorflow

#endif  // TENSORFLOW_USE_UCX
