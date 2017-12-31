# Description:
#   Ucx RDMA communication interfaces and implementations for TensorFlow.

package(default_visibility = [
    "//tensorflow:__subpackages__",
])

licenses(["notice"])  # Apache 2.0

load("//tensorflow:tensorflow.bzl", "tf_cuda_library")

exports_files(["LICENSE"])

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)

filegroup(
    name = "c_srcs",
    data = glob([
        "**/*.cc",
        "**/*.h",
    ]),
)

# For platform specific build config
load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library_cc",
)

tf_proto_library_cc(
    name = "ucx_service_proto",
    srcs = ["ucx_service.proto"],
    has_services = 1,
    cc_api_version = 2,
    protodeps = ["//tensorflow/core:protos_all"],
    visibility = [
        "//tensorflow:__subpackages__",
    ],
)

cc_library(
    name = "grpc_ucx_service",
    srcs = ["grpc_ucx_service.cc"],
    hdrs = ["grpc_ucx_service.h"],
    deps = [
        ":grpc_ucx_service_impl",
        ":ucx_mgr",
        ":ucx_service_proto_cc",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/distributed_runtime:session_mgr",
        "//tensorflow/core/distributed_runtime/rpc:async_service_interface",
        "//tensorflow/core/distributed_runtime/rpc:grpc_call",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
        "@grpc//:grpc++_unsecure",
    ],
    alwayslink = 1,
)

cc_library(
    name = "grpc_ucx_service_impl",
    srcs = ["grpc_ucx_service_impl.cc"],
    hdrs = ["grpc_ucx_service_impl.h"],
    deps = [
        ":ucx_service_proto_cc",
        "@grpc//:grpc++_unsecure",
        "//tensorflow/core:lib"
    ],
)

cc_library(
    name = "grpc_ucx_client",
    srcs = ["grpc_ucx_client.cc"],
    hdrs = ["grpc_ucx_client.h"],
    deps = [
        ":grpc_ucx_service_impl",
        ":ucx_service_proto_cc",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:call_options",
        "//tensorflow/core/distributed_runtime/rpc:grpc_util",
    ],
    alwayslink = 1,
)

cc_library(
    name = "ucx_rendezvous_mgr",
    srcs = ["ucx_rendezvous_mgr.cc"],
    hdrs = ["ucx_rendezvous_mgr.h"],
    linkopts = select({
        "//tensorflow:with_verbs_support": ["-lucs", "-lucp", "-luct", "-lucm"],
        "//conditions:default": [],
    }),
    deps = [
        ":ucx_mgr",
        ":ucx_util",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:base_rendezvous_mgr",
        "//tensorflow/core/distributed_runtime:worker_env",
    ],
)

cc_library(
    name = "ucx_util",
    srcs = ["ucx_util.cc"],
    hdrs = ["ucx_util.h"],
    deps = [
        "//tensorflow/core:framework",
        "//tensorflow/core:lib",
    ],
)

cc_library(
    name = "ucx_mgr",
    srcs = ["ucx_mgr.cc"],
    hdrs = ["ucx_mgr.h"],
    deps = [
        ":ucx_channel",
        ":ucx_service_proto_cc",
        ":grpc_ucx_client",
        ":grpc_ucx_service_impl",
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/distributed_runtime:session_mgr",
        "//tensorflow/core/distributed_runtime:worker_env",
        "//tensorflow/core/distributed_runtime/rpc:grpc_channel",
        "//tensorflow/core/distributed_runtime/rpc:grpc_worker_cache",
    ],
)

tf_cuda_library(
    name = "ucx_channel",
    srcs = ["ucx_channel.cc"],
    hdrs = ["ucx_channel.h"],
    deps = [
        "//tensorflow/core:core_cpu_internal",
        "//tensorflow/core:framework",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:gpu_runtime",
        "//tensorflow/core:lib",
        "//tensorflow/core:lib_internal",
        "//tensorflow/core/distributed_runtime:rendezvous_mgr_interface",
        "//tensorflow/core/distributed_runtime:session_mgr",
        "//tensorflow/core/distributed_runtime:worker_env",
    ],
)

cc_library(
    name = "ucx_server_lib",
    srcs = ["ucx_server_lib.cc"],
    hdrs = ["ucx_server_lib.h"],
    linkstatic = 1,  # Seems to be needed since alwayslink is broken in bazel
    deps = [
        ":ucx_mgr",
        ":ucx_rendezvous_mgr",
        ":ucx_util",
        ":grpc_ucx_service",
        "//tensorflow/core:lib",
        "//tensorflow/core/distributed_runtime:server_lib",
        "//tensorflow/core/distributed_runtime/rpc:grpc_server_lib",
    ],
    alwayslink = 1,
)
