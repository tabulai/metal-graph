# Vendored metal-cpp provenance

This directory is an unmodified copy of Apple's Apache-2.0-licensed
[`apple/metal-cpp`](https://github.com/apple/metal-cpp) headers, plus this provenance
file.

- Upstream commit: `27c4382b7151d55a51692cdcb27aaa98752240de`
- Upstream commit date: 2026-06-08
- Reported `METALCPP_VERSION`: 381.0.0
- License: `LICENSE.txt` in this directory

The headers are committed deliberately: on macOS, CMake treats the Metal runner and its
tests as required and fails configuration if this dependency is incomplete. Updating the
dependency should replace the full tree from a reviewed upstream commit and update this
file in the same change.
