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

#include "tensorflow/contrib/ucx/ucx_channel.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

UcxChannel::UcxChannel(const UcxAddress& ucx_addr, ucp_worker_h ucp_worker)
    : self_addr_(ucx_addr), ucp_worker_(ucp_worker) {}

UcxChannel::~UcxChannel() { ucp_ep_destroy(ep_); }

void UcxChannel::SetRemoteAddress(const UcxAddress& ra) {
  remote_addr_ = ra;
  remote_set_ = true;
}

void UcxChannel::Connect() {
  {
    mutex_lock lock{mu_};
    CHECK(remote_set_) << "remote channel is not set";
  }
  Connect(remote_addr_);
}

void UcxChannel::Connect(const UcxAddress& remoteAddr) {
  ucp_ep_params_t ep_params;
  ucs_status_t status;

  ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
  ep_params.address = remoteAddr.get_addr();

  status = ucp_ep_create(ucp_worker_, &ep_params, &ep_);
  CHECK(status == UCS_OK) << "EP creation failed!!! status: " << status << "("
                          << ucs_status_string(status) << ")";
  LOG(INFO) << "UCX channel connected!";
}
}
