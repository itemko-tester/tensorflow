/*
 * ucx_channel.h
 *
 *  Created on: Dec 7, 2017
 *      Author: root
 */

#ifndef TENSORFLOW_CONTRIB_UCX_UCX_CHANNEL_H_
#define TENSORFLOW_CONTRIB_UCX_UCX_CHANNEL_H_

#include <string>
#include <mutex>
#include <ucp/api/ucp.h>

class UcxAddress {

 public:
  UcxAddress() : addr_len_(0), worker_addr_(nullptr) {}
  UcxAddress(ucp_address_t* worker_addr, size_t addr_len)
      : addr_len_(addr_len), worker_addr_((ucp_address_t*)malloc(addr_len_)) {}
  UcxAddress(const UcxAddress& addr)
      : addr_len_(addr.addr_len_), worker_addr_(addr.worker_addr_) {}
  ~UcxAddress() {
    if (worker_addr_ != nullptr) free(worker_addr_);
  }
  ucp_address_t* get_addr() { return worker_addr_; }

 private:
  size_t addr_len_;
  ucp_address_t* worker_addr_;
};

// Class that represents a connection to a remote Ucx peer.
// Responsible for connecting End Points.
class UcxChannel {

  friend class UcxMgr;
  friend class UcxRemoteRendezvous;

 public:
  UcxChannel(ucp_address_t* local_addr, size_t local_addr_len,
             const std::string local_name, const std::string remote_name,
             ucp_worker_h ucp_worker);
  ~UcxChannel();
  const UcxAddress& self_addr() const { return self_addr_; }

  void Connect(const UcxAddress& remoteAddr);
  void Connect();
  void Recv();
  //  void SetRemoteAddress(const UcxAddress& ra);

 protected:
  ucp_ep_h ep_;
  std::string local_name_;
  std::string remote_name_;
  UcxAddress remote_addr_;
  UcxAddress self_addr_;
};

#endif /* TENSORFLOW_CONTRIB_UCX_UCX_CHANNEL_H_ */
