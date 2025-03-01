load("@rules_python//python:defs.bzl", "py_test")

# Python Unit tests
# run "bazel test //:test_device"
py_test(
    name = "test_device",
    srcs = ["test_device.py"],
    imports = ["."],
    deps = [
        ":remote_cuda",  # Use the py_library target, not the extension
    ],
)


