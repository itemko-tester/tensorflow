/*
 * ucx_channel.h
 *
 *  Created on: Dec 7, 2017
 *      Author: root
 */

#ifndef TENSORFLOW_CONTRIB_UCX_UCX_CHANNEL_H_
#define TENSORFLOW_CONTRIB_UCX_UCX_CHANNEL_H_

#include <ucp/api/ucp.h>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include "tensorflow/core/platform/mutex.h"

namespace tensorflow {

class UcxAddress {
 public:
  UcxAddress() : addr_len_(0), worker_addr_(nullptr) {}

  UcxAddress(ucp_address_t* worker_addr, size_t addr_len)
      : addr_len_(addr_len), worker_addr_((ucp_address_t*)malloc(addr_len)) {
    std::memcpy(worker_addr_, worker_addr, addr_len_);
  }

  UcxAddress(const string& worker_addr, size_t addr_len)
      : addr_len_(addr_len), worker_addr_((ucp_address_t*)malloc(addr_len_)) {
    std::memcpy(worker_addr_, worker_addr.c_str(), addr_len_);
  }

  UcxAddress(const UcxAddress& other)
      : addr_len_(other.addr_len_),
        worker_addr_((ucp_address_t*)malloc(addr_len_)) {
    std::memcpy(worker_addr_, other.worker_addr_, addr_len_);
  }

  UcxAddress& operator=(const UcxAddress& other) {
    addr_len_ = other.addr_len_;
    worker_addr_ = (ucp_address_t*)malloc(addr_len_);
    std::memcpy(worker_addr_, other.worker_addr_, addr_len_);
    return *this;
  }

  // move
  UcxAddress& operator=(UcxAddress&& other) {
    if (&other != this) {
      addr_len_ = other.addr_len_;
      worker_addr_ = other.worker_addr_;
      other.worker_addr_ = nullptr;
    }
    return *this;
  }

  ~UcxAddress() {
    if (worker_addr_ != nullptr) free(worker_addr_);
  }

  ucp_address_t* get_addr() const { return worker_addr_; }

  size_t get_size() const { return addr_len_; }

  std::ostream& print(std::ostream& out) const {
    for (uint i = 0; i < addr_len_; i++) {
      out << std::hex << std::setw(2) << std::setfill('0')
          << (int)((unsigned char*)worker_addr_)[i] << " ";
      if (!i % 64) {
        out << std::endl;
      }
    }
    out << std::endl;
    return out;
  }

 protected:
  size_t addr_len_;
  ucp_address_t* worker_addr_;
};

inline std::ostream& operator<<(std::ostream& out, const UcxAddress& addr) {
  return addr.print(out);
}

// Class that represents a connection to a remote Ucx peer.
// Responsible for connecting End Points.
class UcxChannel {
  friend class UcxMgr;
  friend class UcxRemoteRendezvous;

 public:
  UcxChannel(const UcxAddress& ucx_addr, ucp_worker_h ucp_worker);
  ~UcxChannel();
  const UcxAddress& self_addr() const { return self_addr_; }

  void Connect(const UcxAddress& remoteAddr);
  void Connect();
  void Recv();
  void SetRemoteAddress(const UcxAddress& ra);
  const ucp_ep_h GetEp() const { return ep_; }

 protected:
  UcxAddress self_addr_;
  ucp_worker_h ucp_worker_;
  UcxAddress remote_addr_;
  mutex mu_;
  bool remote_set_ GUARDED_BY(bt_mu_) = false;
  ucp_ep_h ep_ = nullptr;
};
}
#endif /* TENSORFLOW_CONTRIB_UCX_UCX_CHANNEL_H_ */
