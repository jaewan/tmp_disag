load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@rules_proto//proto:defs.bzl", "proto_library")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@pip//:requirements.bzl", "requirement")
load("@pybind11_bazel//:build_defs.bzl", "pybind_extension")

package(default_visibility = ["//visibility:public"])

# Proto definitions
proto_library(
    name = "remote_proto",
    srcs = ["proto/remote.proto"],
)

# C++ core libraries
cc_library(
    name = "remote_device_lib",
    srcs = ["csrc/remote_device.cc"],
    hdrs = ["csrc/remote_device.h"],
    deps = [
        "@libtorch",
    ],
)

# Python extension module
pybind_extension(
    name = "remote_cuda_ext",
    srcs = ["csrc/python_bindings.cc"],
    deps = [
        ":remote_device_lib",
        "@libtorch",
    ],
)

# Python package
py_library(
    name = "remote_cuda",
    srcs = glob(["remote_cuda/*.py"]),
    data = [":remote_cuda_ext.so"],
    imports = ["."],
    deps = [
        requirement("torch"),
    ],
)

# Example application
py_binary(
    name = "example",
    srcs = ["example.py"],
    deps = [
        ":remote_cuda",
        requirement("torch"),
    ],
)
