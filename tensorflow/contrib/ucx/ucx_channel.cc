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

UcxChannel::UcxChannel(ucp_address_t *local_addr, size_t local_addr_len,
                       const std::string local_name,
                       const std::string remote_name, ucp_worker_h ucp_worker)
    : local_name_(local_name), remote_name_(remote_name) {
  ucp_ep_params_t ep_params;
  self_addr_ = UcxAddress(local_addr, local_addr_len);

  // TODO send the address to remote
  // TODO recv the address from remote

  ep_params.field_mask =
      UCP_EP_PARAM_FIELD_REMOTE_ADDRESS | UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE;
  ep_params.address = remote_addr_.get_addr();
  // ep_params.err_mode        = err_handling_opt.ucp_err_mode;

  ucp_ep_create(ucp_worker, &ep_params, &ep_);
}

UcxChannel::~UcxChannel() { ucp_ep_destroy(ep_); }

/*
void UcxChannel::SetRemoteAddress(const UcxAddress& ra) {
  remote_addr_(ra);

}
 */
