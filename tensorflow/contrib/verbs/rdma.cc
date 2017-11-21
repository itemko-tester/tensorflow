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

//#ifdef TENSORFLOW_USE_VERBS

#include "tensorflow/contrib/verbs/rdma.h"
#include "tensorflow/contrib/verbs/rdma_memory_mgr.h"
#include <cstdlib>
#include <fcntl.h>
#include "tensorflow/contrib/verbs/verbs_util.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/common_runtime/gpu/gpu_util.h"
#include "tensorflow/core/common_runtime/gpu/process_state.h"
#include "tensorflow/core/distributed_runtime/rendezvous_mgr_interface.h"
#include "tensorflow/core/distributed_runtime/session_mgr.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/random/random.h"

namespace tensorflow {

#define RoCE_V2 "RoCE v2"

namespace {
// hash name to 32-bit integer
uint32_t NameHash(const string& name) {
  return Hash32(name.data(), name.size(), 0x1234ABCD);
}

// convenience function for printing message
string MessageTypeToString(RdmaMessageType rmt) {
  switch (rmt) {
    case RDMA_MESSAGE_TENSOR_CONTENT_REQUEST:
      return "RDMA_MESSAGE_TENSOR_CONTENT_REQUEST";
      break;
    case RDMA_MESSAGE_TENSOR_META_DATA_REQUEST:
      return "RDMA_MESSAGE_TENSOR_META_DATA_REQUEST";
      break;
    case RDMA_MESSAGE_TENSOR_META_DATA_RESPONSE:
      return "RDMA_MESSAGE_TENSOR_META_DATA_RESPONSE";
      break;
    default:
      return "UNKNOWN MESSAGE";
  }
}
}  // namespace

// Function to get environment variable
// Args:
//    var_name - the name of the environmental variable
// Returns:
//    string with it's value or empty string if not set
string get_env_var(char const* var_name) {
  char const* var_temp = getenv(var_name);

  return (var_temp == NULL) ? string() : string(var_temp);
}

// Function to open device
// Args:
//   ibv_dev device to open
// Returns:
//   context of the opened device
ibv_context* open_device(ibv_device* ibv_dev) {
  ibv_context* context = ibv_open_device(ibv_dev);

  CHECK(context) << "Open context failed for " << ibv_get_device_name(ibv_dev);
  return context;
}

// Function to count the number of active ports for device
// Args:
//   device - to check active ports
// Returns:
//   number of active ports of the given device
int get_dev_active_port_count(ibv_device* device) {
  ibv_device_attr device_att;
  ibv_port_attr port_attr;
  ibv_context* context = NULL;
  int rc, port_index, active_ports = 0;

  context = ibv_open_device(device);
  CHECK(context) << "Open context failed for " << ibv_get_device_name(device);
  rc = ibv_query_device(context, &device_att);
  CHECK(!rc) << "Failed to query the device";

  for (port_index = 1; port_index <= device_att.phys_port_cnt; port_index++) {
    rc = ibv_query_port(context, port_index, &port_attr);
    CHECK(!rc) << "Failed to query the port" << port_index;
    if (port_attr.state == IBV_PORT_ACTIVE) {
      active_ports++;
    }
  }
  ibv_close_device(context);
  return active_ports;
}

// Function to set device. If RDMA_DEVICE not set, search for device with active
// port.
// Fails if more than one device with active port was found.
// Returns:
//   device to use
ibv_device* set_device() {
  ibv_device** dev_list;
  int dev_num, device_index, device_to_open = 0;
  int num_devs_with_active_port = 0;
  string env_p_rdma_device, str_port_num;

  dev_list = ibv_get_device_list(&dev_num);
  CHECK(dev_list) << "No InfiniBand device found";

  env_p_rdma_device = get_env_var("RDMA_DEVICE");
  if (!env_p_rdma_device.empty()) {
    for (device_index = 0; device_index < dev_num; device_index++) {
      if (!env_p_rdma_device.compare(
               ibv_get_device_name(dev_list[device_index]))) {
        CHECK(get_dev_active_port_count(dev_list[device_index]) != 0)
            << "Device " << ibv_get_device_name(dev_list[device_index])
            << " has no active ports";
        return dev_list[device_index];
      }
    }
    // check validity of input device
    CHECK(false) << "The device " << env_p_rdma_device << " wasn't found";
  } else {
  // set default device
    str_port_num = get_env_var("RDMA_DEVICE_PORT");
    CHECK(str_port_num.empty())
        << "RDMA_DEVICE should be provided if RDMA_DEVICE_PORT is set by user";
    for (device_index = 0; device_index < dev_num; device_index++) {
      // get port_num
      if (get_dev_active_port_count(dev_list[device_index]) > 0) {
        num_devs_with_active_port++;
        CHECK(num_devs_with_active_port <= 1) << ". More than one device with "
                                                 "active port in the system. "
                                                 "Please enter RDMA_DEVICE";
        // found device with at least 1 active port
        device_to_open = device_index;
      }
    }
    CHECK(num_devs_with_active_port > 0)
        << "There is no active port in the system";
    return dev_list[device_to_open];
  }
  CHECK(false) << "No device was set!";
  return NULL;  // never happens
}

// Function to set port for device.
// If RDMA_DEVICE_PORT not set, first active port of the device will be set.
// Args:
//   context of the device
// Returns:
//   port to use
uint8_t set_port(ibv_context* context) {
  uint8_t port_num = 0; //0 is illegal port number
  string str_port_num;
  ibv_device_attr device_att;
  ibv_port_attr port_attr;
  int rc, port_index;

  rc = ibv_query_device(context, &device_att);
  CHECK(!rc) << "Failed to query the device\n";

  str_port_num = get_env_var("RDMA_DEVICE_PORT");
  // user defined port
  if (!str_port_num.empty()) {
    port_num = stoi(str_port_num);
    CHECK(port_num > 0) << "RDMA_DEVICE_PORT should be positive";
    CHECK(port_num <= device_att.phys_port_cnt) << "RDMA_DEVICE_PORT should be "
                                                   "less or equal to amount of "
                                                   "available ports";
    rc = ibv_query_port(context, port_num, &port_attr);
    CHECK(!rc) << "Failed to query the port" << port_num;
    // check if port id active
    CHECK(port_attr.state == IBV_PORT_ACTIVE)
        << "Selected RDMA_DEVICE_PORT is not active";
  }
  // set default port
  else {
    for (port_index = 1; port_index <= device_att.phys_port_cnt; port_index++) {
      rc = ibv_query_port(context, port_index, &port_attr);
      CHECK(!rc) << "Failed to query the port" << port_index;
      if (port_attr.state == IBV_PORT_ACTIVE) {
        port_num = port_index;
        break;
      }
    }
    CHECK_GT(port_num, 0) << "No active ports";
  }
  return port_num;
}

// Function read from sysfs file
// Args:
//   dir - directory
//   file - file
//   buff - buffer for the result
//   size - buffer size
// Returns:
//   number of bytes were read or -1 if failed
int read_sysfs_file(const char* dir, const char* file, char* buf, size_t size) {
  char* path;
  int fd;
  int len;

  if (asprintf(&path, "%s/%s", dir, file) < 0) return -1;

  fd = open(path, O_RDONLY);
  if (fd < 0) {
    free(path);
    return -1;
  }

  len = read(fd, buf, size);

  close(fd);
  free(path);

  if (len > 0 && buf[len - 1] == '\n') buf[--len] = '\0';

  return len;
}

// Function to check if GID index support RoCE V2
// Args:
//   context - device context
//   port_num - port number
//   index -  GID index
// Returns:
//   if GID supports RoCE V2 - true, otherwise - false.
bool is_gid_type_roce_v2(ibv_context* context, uint8_t port_num,
                         uint8_t index) {
  char name[32];
  char buff[41];

  snprintf(name, sizeof(name), "ports/%d/gid_attrs/types/%d", port_num, index);
  if (read_sysfs_file(context->device->ibdev_path, name, buff, sizeof(buff)) <=
      0) {
    return false;
  }
  return !strcmp(buff, RoCE_V2);
}

// Function to set GID index.
// If the port link is IB, no GID index should be selected.
// If Ethernet but RDMA_GID_INDEX not set gid index that supports
//   RoCE V2 will be chosen(fails if more then one IP is configured)
// Args:
//   context - device context
//   port_num - port number
// Returns:
//   GID index to use
uint8_t set_gid(uint8_t port_num, ibv_context* context) {
  ibv_port_attr port_attr;
  string gid_str;
  int rc, i, gids_num = 0, v2_ip_num = 0;
  union ibv_gid gid;
  uint8_t gid_index = 0;

  rc = ibv_query_port(context, port_num, &port_attr);
  CHECK(!rc) << "Failed to query the port" << port_num;

  for (i = 0; i < port_attr.gid_tbl_len; i++) {
    rc = ibv_query_gid(context, port_num, i, &gid);
    CHECK(!rc) << "Failed to query gid to port " << (int)port_num << " index "
               << i;
    if (gid.global.interface_id) {
      gids_num++;
      if (gid.global.subnet_prefix == 0 &&
          is_gid_type_roce_v2(context, port_num, i)) {
        if (v2_ip_num == 0) {
          // can be overwritten by RDMA_GID_INDEX later
          gid_index = i;
        }
        v2_ip_num++;
      }
    }
  }
  switch (port_attr.link_layer) {
    case(IBV_LINK_LAYER_ETHERNET) :
      gid_str = get_env_var("RDMA_GID_INDEX");
      if (!gid_str.empty()) {
        gid_index = stoi(gid_str);
        CHECK(gid_index < gids_num)
            << "RDMA_GID_INDEX should be less than GIDs amount" << gids_num;
      } else {
        CHECK(v2_ip_num <= 1)
            << "More than one IP is available, please specify GID_INDEX";
      }
      break;
    case(IBV_LINK_LAYER_INFINIBAND) :  // no need in GID index
      break;
    default:
      LOG(INFO) << "Unknown port link layer. Currently supporting Ethernet and "
                   "InfiniBand only. ";
  }
  if (!is_gid_type_roce_v2(context, port_num, gid_index)) {
    LOG(INFO) << "RoCE v2 is not configured for GID_INDEX " << (int)gid_index;
  }
  return gid_index;
}

// set the default or environment value to the configuration parameter.
// Args:
//   default_val- the default value for this parameter
//   env_param- the environment parameter's name
// Returns:
//   32-bit value
uint32_t set_param(uint32_t default_val, const char* env_param) {
  uint32_t val = default_val;
  string val_s;

  val_s = get_env_var(env_param);

  if (!val_s.empty()) {
    val = stoi(val_s);
  }
  return val;
}

enum ibv_mtu set_mtu(uint8_t port_num, ibv_context* context) {
  ibv_port_attr port_attr;
  enum ibv_mtu mtu;
  string mtu_s;
  int rc, mtu_i;

  rc = ibv_query_port(context, port_num, &port_attr);
  CHECK(!rc) << "Failed to query the port" << port_num;

  mtu_s = get_env_var("RDMA_MTU");

  if (!mtu_s.empty()) {
    mtu_i = stoi(mtu_s);
    switch (mtu_i) {
      case 256:
        mtu = IBV_MTU_256;
        break;
      case 512:
        mtu = IBV_MTU_512;
        break;
      case 1024:
        mtu = IBV_MTU_1024;
        break;
      case 2048:
        mtu = IBV_MTU_2048;
        break;
      case 4096:
        mtu = IBV_MTU_4096;
        break;
      default:
        CHECK(0) << "Error: MTU input value must be one of the following: 256, "
                    "512, 1024, 2048, 4096. MTU " << mtu << " is invalid\n";
        break;
    }
    CHECK(mtu < port_attr.active_mtu)
        << "MTU configuration for the QPs is larger than active MTU";
  } else {
    mtu = port_attr.active_mtu;
  }
  return mtu;
}

RdmaParams params_init(ibv_context* context) {
  RdmaParams params;

  params.port_num = set_port(context);
  params.sgid_index = set_gid(params.port_num, context);
  params.pkey_index = (uint8_t)set_param(PKEY_DEFAULT, "RDMA_PKEY");
  params.queue_depth = set_param(QUEUE_DEPTH_DEFAULT, "RDMA_QUEUE_DEPTH");
  params.timeout = (uint8_t)set_param(TIMEOUT_DEFAULT, "RDMA_TIMEOUT");
  params.retry_cnt = (uint8_t)set_param(RETRY_CNT_DEFAULT, "RDMA_RETRY_CNT");
  params.sl = (uint8_t)set_param(SL_DEFAULT, "RDMA_SL");
  CHECK(params.sl <= 7) << "SL value is " << (int)params.sl
                        << ". Valid values are 0-7.";
  params.mtu = set_mtu(params.port_num, context);
  params.traffic_class = set_param(TRAFFIC_CLASS, "RDMA_TRAFFIC_CLASS");
  return params;
}

ibv_pd* alloc_protection_domain(ibv_context* context) {
  ibv_pd* pd = ibv_alloc_pd(context);
  CHECK(pd) << "Failed to allocate protection domain";
  return pd;
}

RdmaAdapter::RdmaAdapter(const WorkerEnv* worker_env)
    : context_(open_device(set_device())),
      params_(params_init(context_)),
      pd_(alloc_protection_domain(context_)),
      worker_env_(worker_env) {
  event_channel_ = ibv_create_comp_channel(context_);
  CHECK(event_channel_) << "Failed to create completion channel";
  cq_ = ibv_create_cq(context_, MAX_CONCURRENT_WRITES * 2, NULL, event_channel_,
                      0);
  CHECK(cq_) << "Failed to create completion queue";
  CHECK(!ibv_req_notify_cq(cq_, 0)) << "Failed to request CQ notification";
  polling_thread_.reset(Env::Default()->StartThread(
      ThreadOptions(), "RdmaAdapterCQThread", [this] { Process_CQ(); }));
  VLOG(2) << "Start RdmaAdapter: " << name();
}

RdmaAdapter::~RdmaAdapter() {
  polling_thread_.reset();
  CHECK(!ibv_destroy_cq(cq_)) << "Failed to destroy CQ";
  CHECK(!ibv_destroy_comp_channel(event_channel_))
      << "Failed to destroy channel";
  CHECK(!ibv_dealloc_pd(pd_)) << "Failed to deallocate PD";
  CHECK(!ibv_close_device(context_)) << "Failed to release context";
}

string RdmaAdapter::name() const { return string(context_->device->name); }

class WriteContextDesc {
public:
  WriteContextDesc(uint64_t write_type, void* write_context)
    : write_type_(write_type)
    , write_context_(write_context) {
  }

  uint64_t write_type_;
  void* write_context_;
};


RdmaTensorRequest* RdmaChannel::CreatePendingRequest() {
  mutex_lock l(pending_requests_mu_);
  auto it = pending_requests_.emplace(pending_request_serial, RdmaTensorRequest(pending_request_serial));
  ++pending_request_serial;
  return &it.first->second;
}
RdmaTensorRequest* RdmaChannel::PopPendingRequest(int request_num) {
  mutex_lock l(pending_requests_mu_);
  auto it = pending_requests_.find(request_num);
  CHECK(it != pending_requests_.end());
  pending_requests_.erase(it);
  return &it->second;
}
RdmaTensorRequest* RdmaChannel::GetPendingRequest(int request_num) {
  mutex_lock l(pending_requests_mu_);
  auto it = pending_requests_.find(request_num);
  CHECK(it != pending_requests_.end());
  return &it->second;
}

RdmaTensorResponse* RdmaChannel::CreatePendingResponse(int request_index) {
  mutex_lock l(pending_responses_mu_);
  auto it = pending_responses_.emplace(request_index, RdmaTensorResponse(request_index));
  CHECK(it.second);  // Make sure not called twice for request
  RdmaTensorResponse& response = it.first->second;
  return &response;
}
RdmaTensorResponse* RdmaChannel::DeletePendingResponse(int request_index) {
  mutex_lock l(pending_responses_mu_);
  auto it = pending_responses_.find(request_index);
  CHECK(it != pending_responses_.end());
  pending_responses_.erase(it);
  return &it->second;
}
RdmaTensorResponse* RdmaChannel::GetPendingResponse(int request_index) {
  mutex_lock l(pending_responses_mu_);
  auto it = pending_responses_.find(request_index);
  if (it == pending_responses_.end()) {
    return nullptr;
  }
  return &it->second;
}

// Function to process incoming messages
// There are two types of messages:
// 1. IBV_WC_RECV_RDMA_WITH_IMM (receive)
// 2. IBV_WC_RDMA_WRITE (send))
void RdmaAdapter::Process_CQ() {
  while (true) {
    ibv_cq* cq;
    void* cq_context;
    CHECK(!ibv_get_cq_event(event_channel_, &cq, &cq_context));
    CHECK(cq == cq_);
    ibv_ack_cq_events(cq, 1);
    CHECK(!ibv_req_notify_cq(cq_, 0));

    int ne =
        ibv_poll_cq(cq_, MAX_CONCURRENT_WRITES * 2, static_cast<ibv_wc*>(wc_));
    CHECK_GE(ne, 0);
    for (int i = 0; i < ne; ++i) {
      if (wc_[i].status != IBV_WC_SUCCESS) {
          WriteContextDesc* desc = reinterpret_cast<WriteContextDesc*>(wc_[i].wr_id);
          int imm_data = desc->write_type_;
          LOG(INFO) << "ERROR: " << ibv_wc_status_str(wc_[i].status) << ": "
                     << "OPCODE: " << wc_[i].opcode << " "
                     << "WR-ID: " << std::hex << "0x" << wc_[i].wr_id << " "
                     << "IMM: 0x" << std::hex << imm_data;

          if ((imm_data == RDMA_IMM_TYPE_MESSAGE) ||
              (imm_data == RDMA_IMM_TYPE_ACK)) {

            RdmaBuffer* rb = reinterpret_cast<RdmaBuffer*>(desc->write_context_);
            LOG(INFO) << "MESSAGE FAILED. SRC-BUFFER: " << rb->buffer_;
          } else {
            TensorBuffer* src_buffer = reinterpret_cast<TensorBuffer*>(desc->write_context_);
            LOG(INFO) << "TENSOR WRITE FAILED. SRC-BUFFER: " << ((src_buffer == nullptr) ? (void*)0 : src_buffer->data());
          }
          LOG(FATAL) << "Exiting.";
      }

//      LOG(INFO) << "EVENT " << ibv_wc_status_str(wc_[i].status) << " "
//      << "OPCODE: " << wc_[i].opcode << " "
//      << "WR-ID: " << wc_[i].wr_id << " "
//      << "IMM: " << std::hex << wc_[i].imm_data;

      uint32_t imm_data = wc_[i].imm_data;


      if (wc_[i].opcode == IBV_WC_RECV_RDMA_WITH_IMM) {
        RdmaChannel* rc = reinterpret_cast<RdmaChannel*>(wc_[i].wr_id);
        // put back a recv wr.
        rc->Recv();
        // imm_data is the index of RX buffer in the buffer table.
        if (imm_data == RDMA_IMM_TYPE_ACK)
        {
          // receive an ack to a message
          RdmaBuffer* rb = rc->tx_message_buffer_;
          rb->SetBufferStatus(remote, idle);
          rb->SendNextItem();

        }
        else if (imm_data == RDMA_IMM_TYPE_MESSAGE) {

          RdmaMessage rm;
          RdmaMessage::ParseMessage(rm, rc->rx_message_buffer_->buffer_);
          int request_index = rm.request_index_;

          // received a request-for-tensor message
          // send ack to release remote tx message buffer
          RdmaBuffer* ab = rc->tx_ack_buffer_;
          ab->SendNextItem();

          string key = rm.name_;
          int64 step_id = rm.step_id_;
          Rendezvous::ParsedKey parsed;
          Rendezvous::ParseKey(key, &parsed);

          if (rm.type_ == RDMA_MESSAGE_TENSOR_META_DATA_REQUEST)
          {
            //****************************************
            // RDMA_MESSAGE_TENSOR_META_DATA_REQUEST:
            //****************************************
//            LOG(INFO) << "STEP 0x" << std::hex << step_id << std::dec << ": RECEIVED META-DATA REQUEST #" << request_index << ": " << key;

            Rendezvous::DoneCallback cb = [rc, key, step_id, request_index](const Status& status,
                                                                            const Rendezvous::Args& send_args,
                                                                            const Rendezvous::Args& recv_args,
                                                                            const Tensor& in, bool is_dead) {

              CHECK(status.ok()) << "RecvLocalAsync was not ok: " << status.error_message();
              RdmaTensorResponse* response = rc->CreatePendingResponse(request_index);
              response->in_ = std::move(in);
              response->send_args_ = send_args;
              response->recv_args_ = recv_args;
              response->is_dead_ = is_dead;

              RdmaTensorBuffer::SendTensorMetaDataResponse(rc, key, step_id, response);
            };
            worker_env_->rendezvous_mgr->RecvLocalAsync(step_id, parsed, std::move(cb));
          }
          else if (rm.type_ == RDMA_MESSAGE_TENSOR_CONTENT_REQUEST)
          {
            //**************************************
            // RDMA_MESSAGE_TENSOR_CONTENT_REQUEST:
            //**************************************
//            LOG(INFO) << "STEP 0x" << std::hex << step_id << std::dec << ": RECEIVED CONTENT REQUEST #" << request_index << ": " << key;
            RdmaTensorResponse* response = rc->GetPendingResponse(request_index);
            // send the next tensor
            if (response == nullptr)
            {
              Rendezvous::DoneCallback cb = [rc, rm, request_index, parsed](const Status& status,
                                                const Rendezvous::Args& send_args,
                                                const Rendezvous::Args& recv_args,
                                                const Tensor& in, bool is_dead)
              {
                CHECK(status.ok()) << "RecvLocalAsync was not ok: " << status.error_message();
                RdmaTensorBuffer::SendTensorContentResponse(rc, send_args, recv_args, in, is_dead,
                                                            rm.remote_addr_, rm.rkey_, rm.name_, rm.step_id_, request_index, parsed);
              };
              worker_env_->rendezvous_mgr->RecvLocalAsync(step_id, parsed, std::move(cb));
            }
            else
            {
              RdmaTensorResponse r = *response;
              rc->DeletePendingResponse(request_index);
              RdmaTensorBuffer::SendTensorContentResponse(rc, r.send_args_, r.recv_args_, r.in_, r.is_dead_,
                                                          rm.remote_addr_, rm.rkey_, rm.name_, rm.step_id_, request_index, parsed);
            }

          } else if (rm.type_ == RDMA_MESSAGE_TENSOR_META_DATA_RESPONSE) {
            //*****************************************
            // RDMA_MESSAGE_TENSOR_META_DATA_RESPONSE:
            //*****************************************
//            LOG(INFO) << "STEP 0x" << std::hex << rm.step_id_ << std::dec
//                      << ": RECEIVED META-DATA RESPONSE #" << request_index << ": " << key
//                      << " (TYPE = " << DataTypeString(rm.data_type_) << ". SHAPE = " << rm.tensor_shape_.DebugString() << ").";

            RdmaTensorRequest* request = rc->GetPendingRequest(request_index);
            worker_env_->compute_pool->Schedule([request, rm]() {
              (*request->recv_meta_data_callback_)(rm.data_type_, rm.tensor_shape_);
            });
          }
        }
        else
        {
          //*****************************************
          // RDMA_MESSAGE_TENSOR_CONTENT_RESPONSE:
          //*****************************************
          RdmaTensorRequest* request = rc->GetPendingRequest(imm_data);
          worker_env_->compute_pool->Schedule([request]() {
            (*request->recv_content_callback_)();
          });
        }
      }

      else if (wc_[i].opcode == IBV_WC_RDMA_WRITE)
      {
        WriteContextDesc* desc = reinterpret_cast<WriteContextDesc*>(wc_[i].wr_id);
        imm_data = desc->write_type_;
//        LOG(INFO) << "WRITE COMPLETED ON " << desc->write_context_ << " IMM: 0x" << std::hex << desc->write_type_;

        if ((imm_data == RDMA_IMM_TYPE_MESSAGE) ||
        	(imm_data == RDMA_IMM_TYPE_ACK))
        {
          RdmaBuffer* rb = reinterpret_cast<RdmaBuffer*>(desc->write_context_);
          rb->SetBufferStatus(local, idle);
          if (imm_data != RDMA_IMM_TYPE_ACK)
          {
            worker_env_->compute_pool->Schedule([rb]() { rb->SendNextItem(); });
          }
        }
        else
        {
          TensorBuffer* src_buffer = (TensorBuffer*)desc->write_context_;
          if (src_buffer != nullptr) {
  //          void* src_addr = src_buffer->data();
            src_buffer->Unref();
          }
//          LOG(INFO) << "WRITE TENSOR COMPLETE FROM " << src_addr;
        }
        delete desc;
      }
    }
  }
}

RdmaChannel::RdmaChannel(const RdmaAdapter* adapter, const string local_name,
                         const string remote_name)
    : adapter_(adapter), local_name_(local_name), remote_name_(remote_name) {
  // Create queue pair
  {
    struct ibv_qp_init_attr attr;
    memset(&attr, 0, sizeof(ibv_qp_init_attr));
    attr.send_cq = adapter_->cq_;
    attr.recv_cq = adapter_->cq_;
    attr.cap.max_send_wr = adapter_->params_.queue_depth;
    attr.cap.max_recv_wr = adapter_->params_.queue_depth;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.qp_type = IBV_QPT_RC;

    qp_ = ibv_create_qp(adapter_->pd_, &attr);
    CHECK(qp_) << "Failed to create queue pair";
  }

  // Init queue pair
  {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(ibv_qp_attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = adapter_->params_.pkey_index;
    attr.port_num = adapter_->params_.port_num;
    attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE;

    int mask =
        IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    CHECK(!ibv_modify_qp(qp_, &attr, mask)) << "Failed to set QP to INIT";
  }

  // Local address
  {
    struct ibv_port_attr attr;
    CHECK(
        !ibv_query_port(adapter_->context_, adapter_->params_.port_num, &attr))
        << "Query port";
    self_.lid = attr.lid;
    self_.qpn = qp_->qp_num;
    self_.psn = static_cast<uint32_t>(random::New64()) & 0xffffff;
    union ibv_gid gid;
    CHECK(!ibv_query_gid(adapter_->context_, adapter_->params_.port_num,
                         adapter_->params_.sgid_index, &gid))
        << "Query gid";
    self_.snp = gid.global.subnet_prefix;
    self_.iid = gid.global.interface_id;
  }

  // create message and ack buffers, then initialize the tables.
  {
    const string buffer_names[] = {"tx_message_buffer", "rx_message_buffer",
                                   "tx_ack_buffer",     "rx_ack_buffer"};
    tx_message_buffer_ = new RdmaMessageBuffer(this, buffer_names[0]);
    rx_message_buffer_ = new RdmaMessageBuffer(this, buffer_names[1]);
    tx_ack_buffer_ = new RdmaAckBuffer(this, buffer_names[2]);
    rx_ack_buffer_ = new RdmaAckBuffer(this, buffer_names[3]);
    message_buffers_.reserve(kNumMessageBuffers);
    message_buffers_.push_back(tx_message_buffer_);
    message_buffers_.push_back(rx_message_buffer_);
    message_buffers_.push_back(tx_ack_buffer_);
    message_buffers_.push_back(rx_ack_buffer_);
    // create buffer on host
    tx_message_buffer_->CreateCPUBuffer(RdmaMessage::kRdmaMessageBufferSize);
    rx_message_buffer_->CreateCPUBuffer(RdmaMessage::kRdmaMessageBufferSize);
    tx_ack_buffer_->CreateCPUBuffer(RdmaMessage::kRdmaAckBufferSize);
    rx_ack_buffer_->CreateCPUBuffer(RdmaMessage::kRdmaAckBufferSize);
    LOG(INFO) << "ELAD: tx_message_buffer_ = " << tx_message_buffer_->buffer_;
    LOG(INFO) << "ELAD: rx_message_buffer_ = " << rx_message_buffer_->buffer_;
    LOG(INFO) << "ELAD: tx_ack_buffer_ = " << tx_ack_buffer_->buffer_;
    LOG(INFO) << "ELAD: rx_ack_buffer_ = " << rx_ack_buffer_->buffer_;

    // bt_mu_.lock() is not used in constructor.
    for (int i = 0; i < kNumMessageBuffers; i++) {
      uint32_t index = NameHash(buffer_names[i]);
      buffer_table_.insert({index, message_buffers_[i]});
      buffer_index_name_table_.insert({index, buffer_names[i]});
      buffer_name_index_table_.insert({buffer_names[i], index});
    }

    // Initiate recv
    for (int i = 0; i < 100; i++) {
      Recv();
    }
  }
}

RdmaChannel::~RdmaChannel() {
  CHECK(!ibv_destroy_qp(qp_)) << "Failed to destroy QP";
  delete tx_message_buffer_;
  delete rx_message_buffer_;
  delete tx_ack_buffer_;
  delete rx_ack_buffer_;
}

void RdmaChannel::SetRemoteAddress(const RdmaAddress& ra, bool override) {
  mutex_lock lock{mu_};
  if ((override) || (!remote_set_)) {
    remote_.lid = ra.lid;
    remote_.qpn = ra.qpn;
    remote_.psn = ra.psn;
    remote_.snp = ra.snp;
    remote_.iid = ra.iid;
    remote_set_ = true;
  } else {
    CHECK(remote_.lid == ra.lid);
    CHECK(remote_.qpn == ra.qpn);
    CHECK(remote_.psn == ra.psn);
    CHECK(remote_.snp == ra.snp);
    CHECK(remote_.iid == ra.iid);
  }
}

// Adding tokens to the completion queue
// Tokens are needed to process future messages.
void RdmaChannel::Recv() {
  struct ibv_recv_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t) this;
  struct ibv_recv_wr* bad_wr;
  CHECK(!ibv_post_recv(qp_, &wr, &bad_wr)) << "Failed to post recv";
}

// Lookup 32-bit buffer index from buffer name
// Args:
//   buffer_name: name of the buffer
// Returns:
//   32-bit index
uint32_t RdmaChannel::LookupBufferIndex(const string& buffer_name) {
  mutex_lock lock{bt_mu_};
  BufferNameIndexTable::iterator iter =
      buffer_name_index_table_.find(buffer_name);
  CHECK(iter != buffer_name_index_table_.end());
  return iter->second;
}

// Find a buffer by its 32-bit index
// Args:
//   index: 32-bit hash code of the tensor buffer name
// Returns:
//   name of the tensor buffer
RdmaBuffer* RdmaChannel::FindBuffer(const uint32_t index) {
  mutex_lock lock{bt_mu_};
  BufferTable::iterator iter = buffer_table_.find(index);
  CHECK(iter != buffer_table_.end());
  return iter->second;
}

// Find a buffer by its name
// Args:
//   name: name of the buffer
// Returns:
//   the named rdma buffer
RdmaBuffer* RdmaChannel::FindBuffer(const string& name) {
  uint32_t index = LookupBufferIndex(name);
  return FindBuffer(index);
}

// Find a buffer if it exists, otherwise create one.
// The memory inside the created buffer is not allocated.
// Args:
//   name: the name of the buffer
//   buffer_type: TENSOR, MESSAGE or ACK.
// Returns:
//   the named buffer
RdmaBuffer* RdmaChannel::FindOrCreateBuffer(const string& name,
                                            BufferType buffer_type) {
  mutex_lock lock{bt_mu_};
  RdmaBuffer* rb;
  // find index
  BufferNameIndexTable::iterator iter = buffer_name_index_table_.find(name);
  if (iter != buffer_name_index_table_.end()) {
    uint32_t index = iter->second;
    // find buffer
    BufferTable::iterator iter = buffer_table_.find(index);
    CHECK(iter != buffer_table_.end());
    rb = iter->second;
  } else {
    uint32_t index = NameHash(name);
    if (buffer_type == TENSOR) {
      rb = new RdmaTensorBuffer(this, name);
    } else if (buffer_type == MESSAGE) {
      rb = new RdmaMessageBuffer(this, name);
    } else if (buffer_type == ACK) {
      rb = new RdmaAckBuffer(this, name);
    }
    buffer_name_index_table_.insert({name, index});
    buffer_index_name_table_.insert({index, name});
    buffer_table_.insert({index, rb});
  }
  CHECK(rb);
  return rb;
}


void RdmaChannel::Connect() {
  {
    mutex_lock lock{mu_};
    CHECK(remote_set_) << "remote channel is not set";
  }
  Connect(remote_);
}

// Setup channel to a remote node
// Args:
//   remoteAddr: the rdma address of a remote channel.
// Returns:
//   None
void RdmaChannel::Connect(const RdmaAddress& remoteAddr) {
  mutex_lock lock{mu_};
  if (!connected_) {
    struct ibv_qp_attr attr;
    memset(&attr, 0, sizeof(ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTR;

    // This assumes both QP's ports are configured with the same MTU
    attr.path_mtu = adapter_->params_.mtu;
    attr.dest_qp_num = remoteAddr.qpn;
    attr.rq_psn = remoteAddr.psn;
    attr.max_dest_rd_atomic = 1;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 1;
    attr.ah_attr.grh.dgid.global.subnet_prefix = remoteAddr.snp;
    attr.ah_attr.grh.dgid.global.interface_id = remoteAddr.iid;
    attr.ah_attr.grh.flow_label = 0;
    attr.ah_attr.grh.hop_limit = 255;
    attr.ah_attr.dlid = remoteAddr.lid;
    attr.ah_attr.sl = adapter_->params_.sl;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = adapter_->params_.port_num;
    attr.ah_attr.grh.sgid_index = adapter_->params_.sgid_index;
    attr.ah_attr.grh.traffic_class = adapter_->params_.traffic_class;

    int r;
    CHECK(!(r = ibv_modify_qp(qp_, &attr, IBV_QP_STATE | IBV_QP_AV |
                                              IBV_QP_PATH_MTU |
                                              IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                              IBV_QP_MAX_DEST_RD_ATOMIC |
                                              IBV_QP_MIN_RNR_TIMER)))
        << "QP to Ready to Receive " << r;

    memset(&attr, 0, sizeof(ibv_qp_attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = self_.psn;
    attr.timeout = adapter_->params_.timeout;
    attr.retry_cnt = adapter_->params_.retry_cnt;
    attr.rnr_retry = 7; /* infinite */
    attr.max_rd_atomic = 1;

    CHECK(!(r = ibv_modify_qp(qp_, &attr, IBV_QP_STATE | IBV_QP_TIMEOUT |
                                              IBV_QP_RETRY_CNT |
                                              IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                              IBV_QP_MAX_QP_RD_ATOMIC)))
        << "QP to Ready to Send " << r;

    connected_ = true;
  } else {
    LOG(INFO) << "channel already connected";
  }
}

RdmaBuffer::RdmaBuffer(RdmaChannel* channel, string name)
    : channel_(channel), name_(name) {}

RdmaBuffer::~RdmaBuffer() {
  CHECK(!ibv_dereg_mr(self_)) << "ibv_dereg_mr failed";
  FreeBuffer();
}

void RdmaBuffer::FreeBuffer() {
  if ((buffer_ != nullptr) && buffer_on_host_) {
    free(buffer_);
  }
  // TODO
  // release buffer if it is on device.
  // We don't support RDMABuffer on device at this moment.
}

// Allocate CPU memory for the Rdma buffer
// Args:
//   size: to-be-allocated memory size
//   lock: whether or not mutex_lock the process to protect concurrency.
// Returns:
//   None
void RdmaBuffer::CreateCPUBuffer(size_t size, bool lock) {
  CHECK(size > 0);
  if (lock) {
    mu_.lock();
  }
  if (local_status_ != none) {
    // delete existing buffer
    CHECK(!ibv_dereg_mr(self_)) << "ibv_dereg_mr failed";
    FreeBuffer();
  }
  size_ = size;
  buffer_ = malloc(size_);
  self_ = ibv_reg_mr(channel_->adapter_->pd_, buffer_, size_,
                     IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
  CHECK(self_) << "Failed to register memory region";
  buffer_on_host_ = true;
  local_status_ = idle;
  if (lock) {
    mu_.unlock();
  }
}

// Set address of remote memory region
// Args:
//   rmr: address of remote memory region
//   override: whether override existing information
// Returns:
//   None
void RdmaBuffer::SetRemoteMR(RemoteMR rmr, bool override) {
  mutex_lock lock{mu_};
  if ((override) || (remote_status_ == none)) {
    remote_.remote_addr = rmr.remote_addr;
    remote_.rkey = rmr.rkey;
    remote_status_ = idle;
  } else {
    CHECK(remote_.remote_addr == rmr.remote_addr);
    CHECK(remote_.rkey == rmr.rkey);
  }
}

// Put a task in the buffer's job queue
void RdmaBuffer::EnqueueItem(string item) {
  mutex_lock lock{mu_};
  queue_.push(item);
}

// Rdma-Write the content of the buffer
void RdmaBuffer::Write(uint32_t imm_data, size_t buffer_size) {
  struct ibv_sge list;
  list.addr = (uint64_t)buffer_;
  list.length = buffer_size;
  list.lkey = self_->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t)new WriteContextDesc(imm_data, this);
  wr.sg_list = &list;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = imm_data;
  wr.wr.rdma.remote_addr = (uint64_t)remote_.remote_addr;
  wr.wr.rdma.rkey = remote_.rkey;

  struct ibv_send_wr* bad_wr;
  CHECK(!ibv_post_send(channel_->qp_, &wr, &bad_wr)) << "Failed to post send";
}

RdmaAckBuffer::RdmaAckBuffer(RdmaChannel* channel, string name)
    : RdmaBuffer(channel, name) {}

RdmaMessageBuffer::RdmaMessageBuffer(RdmaChannel* channel, string name)
    : RdmaBuffer(channel, name) {}

RdmaTensorBuffer::RdmaTensorBuffer(RdmaChannel* channel, string name)
    : RdmaBuffer(channel, name) {}

RdmaTensorBuffer::~RdmaTensorBuffer() {
  for (Itable it = retable.begin(); it != retable.end(); ++it) {
    delete (it->second);
  }
}

// Send the next ack from the buffer's job queue.
void RdmaAckBuffer::SendNextItem() {
  uint32_t imm_data = RDMA_IMM_TYPE_ACK;
//  RdmaMessage rm;
//  rm.name_ = "rx_ack_buffer";
//  rm.name_size_ = rm.name_.size();
//  string message = RdmaMessage::CreateMessage(rm);
//  memcpy(buffer_, message.data(), message.size());
  Write(imm_data, 0); //message.size());
}

// Send the next message from the buffer's job queue.
void RdmaMessageBuffer::SendNextItem() {
  uint32_t imm_data = RDMA_IMM_TYPE_MESSAGE;
  mu_.lock();
  if (!queue_.empty() && (local_status_ == idle) && (remote_status_ == idle)) {
    local_status_ = busy;
    remote_status_ = busy;
    string message = queue_.front();
    queue_.pop();
    // local/remote_status_ won't be set back to idle
    // unitl Write() is successful
    mu_.unlock();
    memcpy(buffer_, message.data(), message.size());
    Write(imm_data, message.size());
  } else {
    mu_.unlock();
  }
}


void RdmaTensorBuffer::SendTensorMetaDataResponse(const RdmaChannel* channel,
                                                  const string& key,
                                                  int64 step_id,
                                                  const RdmaTensorResponse* response)
{

  RdmaMessage rm;
  rm.type_ = RDMA_MESSAGE_TENSOR_META_DATA_RESPONSE;
  rm.name_size_ = key.size();
  rm.name_ = key;
  rm.tensor_shape_ = response->in_.shape();
  rm.data_type_ = response->in_.dtype();
  rm.is_dead_ = response->is_dead_;
  rm.step_id_ = step_id;
  rm.request_index_ = response->index_;

//  LOG(INFO) << "STEP 0x" << std::hex << step_id << std::dec << ": SENDING META-DATA RESPONSE #" << response->index_ << ": " << key
//            << " (TYPE = " << DataTypeString(rm.data_type_) << ". SHAPE = " << rm.tensor_shape_.DebugString() << ").";

  string message = RdmaMessage::CreateMessage(rm);
  channel->tx_message_buffer_->EnqueueItem(message);
  channel->tx_message_buffer_->SendNextItem();
}


void RdmaTensorBuffer::SendTensorContentResponse(
    const RdmaChannel* channel,
    const Rendezvous::Args& send_args,
    const Rendezvous::Args& recv_args,
    const Tensor& in, bool is_dead,
    uint64_t remote_addr, uint32_t rkey,
    const string& key, int64 step_id,
    int request_index, const Rendezvous::ParsedKey& parsed) {

  // Figures out which device the tensor is hosted on.
  Device* src_dev = nullptr;
  Status s = channel->adapter_->worker_env_->device_mgr->LookupDevice(
      parsed.src_device, &src_dev);
  CHECK(s.ok()) << "src device not found";
  // Does the device have the right incarnation number we expect?
  CHECK(src_dev->attributes().incarnation() == parsed.src_incarnation)
      << "RecvTensor expects a different device incarnation: "
      << parsed.src_incarnation << " vs. "
      << src_dev->attributes().incarnation()
      << ". Your worker job was probably restarted. Check your "
      << "worker job for the reason why it was restarted.";

  bool can_memcpy = DataTypeCanUseMemcpy(in.dtype());
  // string tensor needs to be serialized
  TensorProto proto;
  if (src_dev->tensorflow_gpu_device_info() && (!send_args.alloc_attrs.on_host()))
  {
    CHECK(send_args.device_context) << "send dev name: " << src_dev->name()
                                    << " gpu_info: "
                                    << src_dev->tensorflow_gpu_device_info();

    if (can_memcpy) {
      if (RdmaMemoryMgr::Singleton().FindMemoryRegion((void*)DMAHelper::base(&in), in.TotalBytes()))
      {
        // Queue the RDMA write on stream
        GPUUtil::StreamOperation(send_args.device_context, src_dev,
            [channel, remote_addr, rkey, proto, key, in, step_id,
             is_dead, request_index, send_args, recv_args](const Status& s) {
              PostCopyOperations(channel, remote_addr, rkey, true, key, proto, in,
                                 step_id, is_dead, request_index,
                                 send_args, recv_args);
            });
      }
      else {
        AllocatorAttributes host_alloc_attrs;
        host_alloc_attrs.set_gpu_compatible(true);
        host_alloc_attrs.set_on_host(true);
        Allocator* alloc = ProcessState::singleton()->GetCUDAHostAllocator(0);
        Tensor* copy = new Tensor(alloc, in.dtype(), in.shape());

        static uint64_t num_proxies = 0;
        static uint64_t num_proxy_bytes = 0;
        num_proxy_bytes += in.TotalBytes();
        if (++num_proxies % 1000 == 0) {
          LOG(INFO) << "Number of SENDER proxies: " << num_proxies << " (" << num_proxy_bytes << " Bytes).";
        }

        GPUUtil::CopyGPUTensorToCPU(
            src_dev, send_args.device_context, &in, copy,
            [channel, remote_addr, rkey, copy, proto, key, in, step_id,
             is_dead, request_index, send_args, recv_args](const Status& s) {
              CHECK(s.ok()) << "copy tensor from gpu sync";
              PostCopyOperations(channel, remote_addr, rkey, true, key, proto, *copy,
                                 step_id, is_dead, request_index,
                                 send_args, recv_args);
              delete copy;
            });
      }
    } else {
      // "val" is on a GPU. No longer uses GPUUtil to fill the proto, use
      // aync instead
      GPUUtil::SetProtoFromGPU(
          in, src_dev, send_args.device_context, &proto, is_dead,
          [channel, remote_addr, rkey, proto, key, in, step_id,
          is_dead, request_index, send_args, recv_args](const Status& s) mutable {
            CHECK(s.ok()) << "copy proto from gpu sync";
            auto tensor_bytes = proto.ByteSize();
            PostCopyOperations(channel, remote_addr, rkey, false, key, proto, in,
                               step_id, is_dead, request_index,
                               send_args, recv_args);
          });
    }
  } else {
    // tensor is in CPU memory.
    if (!can_memcpy) {
      in.AsProtoTensorContent(&proto);
    }
    PostCopyOperations(channel, remote_addr, rkey, can_memcpy,  key, proto, in,
                       step_id, is_dead, request_index,
                       send_args, recv_args);
  }
}

void RdmaTensorBuffer::PostCopyOperations(
    const RdmaChannel* channel, uint64_t remote_addr, uint32_t rkey,
    bool can_memcpy, const string& key, const TensorProto& proto,
    const Tensor& in, int64 step_id, bool is_dead, int request_index,
    const Rendezvous::Args& send_args, const Rendezvous::Args& recv_args) {

  uint32_t imm_data = request_index;
//  LOG(INFO) << "ELAD: BUFFER SIZE = " << buffer_size << " TENSOR BYTES = " << tensor_bytes;
  const TensorBuffer* src_buffer = DMAHelper::buffer(&in);
  if (is_dead) {
    LOG(FATAL) << "TENSOR IS DEAD.";
  }

  void* src_addr;
  size_t write_size;
  void* write_context;
  ibv_mr* mr;
  if (can_memcpy)
  {
    write_size = in.TotalBytes();
    if (write_size > 0)
    {
      src_addr = src_buffer->data();
      src_buffer->Ref(); // Keep the buffer alive until the write completes.
      write_context = (void*)src_buffer;
      mr = RdmaMemoryMgr::Singleton().FindMemoryRegion(src_addr, write_size);
      CHECK(mr != nullptr) << " NO MEMORY REGION FOUND FOR " << src_addr << ": " << key;
    }
    else
    {
      src_addr = 0;
      mr = nullptr;
      write_context = nullptr;
    }
  }
  else
  {
    write_size = proto.ByteSize() + 4;
    src_addr = ProcessState::singleton()->GetCPUAllocator(0)->AllocateRaw(32, write_size);
    *(int*)src_addr = proto.ByteSize();
//    LOG(INFO) << "DECODING " << key << " TO PROTO OF SIZE: " << proto.ByteSize();
    proto.SerializeToArray(src_addr + 4, write_size);
    write_context = nullptr;
    mr = RdmaMemoryMgr::Singleton().FindMemoryRegion(src_addr, write_size);
    CHECK(mr != nullptr) << " NO MEMORY REGION FOUND FOR " << src_addr << ": " << key;
  }


  struct ibv_sge list;
  list.addr = (uint64_t)src_addr;
  list.length = write_size;
  list.lkey = (mr == nullptr) ? 0 : mr->lkey;

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  wr.wr_id = (uint64_t)new WriteContextDesc(imm_data, write_context);
  wr.sg_list = &list;
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE_WITH_IMM;
  wr.send_flags = IBV_SEND_SIGNALED;
  wr.imm_data = imm_data;
  wr.wr.rdma.remote_addr = remote_addr;
  wr.wr.rdma.rkey = rkey;

//  LOG(INFO) << "STEP 0x" << std::hex << step_id << std::dec
//            << ": SENDING CONTENT RESPONSE #" << request_index
//            << " FROM " << std::hex << src_addr << " (0x" << list.lkey << ") TO 0x" << remote_addr << " (0x" << rkey << "): "
//            << key << " (SIZE: 0x" << std::hex << write_size << ")";

  struct ibv_send_wr* bad_wr;
  CHECK(!ibv_post_send(channel->qp_, &wr, &bad_wr)) << "Failed to post send";
}

// Create a RdmaMessage according to the pre-defined format
// Args:
//   rm: the message structure
// Returns:
//   message in string format
string RdmaMessage::CreateMessage(const RdmaMessage& rm) {
  // Rdma Message format
  // type|name_size|name|step_id|buffer_size|remote_addr|rkey|is_dead|...
  //   1B|    2B   | 512|  8B   |    8B     |       8B  | 4B |    1B |...
  // ...|data_type|tensor_shape|tensor_bytes|tensor_buffer
  // ...|   XB    |    XB      |    8B      |...
  //
  // ACK:             type|13|"rx_ack_buffer"
  // TENSOR_REQUEST:  type|name_size|tensor_name|step_id
  // TENSOR_WRITE:    type|name_size|tensor_name|step_id|...|is_dead
  //                 |data_type|tensor_shape|tensor_bytes
  // BUFFER_IDLE:     type|name_size|buffer_name
  // BUFFER_REQUEST:
  // type|name_size|buffer_name|...|buffer_size|remote_addr|rkey|
  // BUFFER_RESPONSE:
  // type|name_size|buffer_name|...|buffer_size|remote_addr|rkey|
  char message[kMessageTotalBytes];
  // type
  message[kTypeStartIndex] = static_cast<char>(rm.type_) & 0xff;
  // size of name
  memcpy(&message[kNameSizeStartIndex], &rm.name_size_, sizeof(rm.name_size_));
  // name
  memcpy(&message[kNameStartIndex], rm.name_.data(), rm.name_.size());
  // buffer_size, remote_addr, rkey
  memcpy(&message[kBufferSizeStartIndex], &rm.buffer_size_,
         sizeof(rm.buffer_size_));
  memcpy(&message[kRemoteAddrStartIndex], &rm.remote_addr_,
         sizeof(rm.remote_addr_));
  memcpy(&message[kRkeyStartIndex], &rm.rkey_, sizeof(rm.rkey_));
  // step_id
  memcpy(&message[kStepIdStartIndex], &rm.step_id_, sizeof(rm.step_id_));
  // is_dead, data_type, tensor_shape, tensor_bytes
  memcpy(&message[kIsDeadStartIndex], &rm.is_dead_, sizeof(rm.is_dead_));

  memcpy(&message[kDataTypeStartIndex], &rm.data_type_,
         sizeof(rm.data_type_));
  memcpy(&message[kTensorShapeStartIndex], &rm.tensor_shape_,
         sizeof(rm.tensor_shape_));
  memcpy(&message[kTensorBytesStartIndex], &rm.request_index_,
         sizeof(rm.request_index_));
  return string(message, kMessageTotalBytes);
}

// Parse a RdmaMessage according to the pre-defined format
// Args:
//   rm: the message structure where the parsed message will be saved
//   buffer: the place where the raw message is stored
// Returns:
//   None
void RdmaMessage::ParseMessage(RdmaMessage& rm, void* buffer) {
  char* message = static_cast<char*>(buffer);
  // type
  rm.type_ = static_cast<RdmaMessageType>(message[kTypeStartIndex]);
  // name_size_
  memcpy(&rm.name_size_, &message[kNameSizeStartIndex], sizeof(rm.name_size_));
  // name
  rm.name_ = string(&message[kNameStartIndex], rm.name_size_);
  // buffer_size, remote_addr, rkey
  memcpy(&rm.buffer_size_, &message[kBufferSizeStartIndex],
         sizeof(rm.buffer_size_));
  memcpy(&rm.remote_addr_, &message[kRemoteAddrStartIndex],
         sizeof(rm.remote_addr_));
  memcpy(&rm.rkey_, &message[kRkeyStartIndex], sizeof(rm.rkey_));
  // step_id
  memcpy(&rm.step_id_, &message[kStepIdStartIndex], sizeof(rm.step_id_));
  memcpy(&rm.is_dead_, &message[kIsDeadStartIndex], sizeof(rm.is_dead_));
  memcpy(&rm.data_type_, &message[kDataTypeStartIndex],
         sizeof(rm.data_type_));
  memcpy(&rm.tensor_shape_, &message[kTensorShapeStartIndex],
         sizeof(rm.tensor_shape_));
  memcpy(&rm.request_index_, &message[kTensorBytesStartIndex],
         sizeof(rm.request_index_));
}

}  // end namespace tensorflow

//#endif
