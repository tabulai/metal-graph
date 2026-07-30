#!/usr/bin/env bash
set -euo pipefail

output_path="${1:?output path is required}"
project_path="${2:?project path is required}"
python_executable="$(command -v python || command -v python3)"

mkdir -p "$(dirname "${output_path}")"

{
  echo "build_identifier=${CIBUILDWHEEL_BUILD_IDENTIFIER:?}"
  echo "commit=$(git -C "${project_path}" rev-parse HEAD)"
  echo "architecture=$(uname -m)"
  echo "deployment_target=${MACOSX_DEPLOYMENT_TARGET:?}"
  echo
  echo "[macOS]"
  sw_vers
  echo
  echo "[Xcode]"
  echo "developer_directory=$(xcode-select -p)"
  xcodebuild -version
  echo
  echo "[macOS SDK]"
  echo "version=$(xcrun --sdk macosx --show-sdk-version)"
  echo "path=$(xcrun --sdk macosx --show-sdk-path)"
  if sdk_build_version="$(xcrun --sdk macosx --show-sdk-build-version 2>/dev/null)"; then
    echo "build_version=${sdk_build_version}"
  fi
  echo
  echo "[CMake]"
  echo "executable=$(command -v cmake)"
  cmake --version
  echo
  echo "[Python]"
  "${python_executable}" --version
  echo "executable=$("${python_executable}" -c 'import sys; print(sys.executable)')"
  echo
  echo "[Python build packages]"
  "${python_executable}" - <<'PY'
from importlib.metadata import version

for package in ("build", "cmake", "nanobind", "ninja", "scikit-build-core"):
    print(f"{package}={version(package)}")
PY
} > "${output_path}"
