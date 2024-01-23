#!/usr/bin/env python3

import os
import subprocess
import sys
from typing import Literal, Optional


def run_backend_discovery_and_copy_test(build_dir: str, backend: Optional[Literal['CUDA']], enabled: bool):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    celerity_dir = os.path.realpath(os.path.join(script_dir, "../.."))

    # Configure and build
    # TODO: Make sure only one backend is enabled at a time (once we have more than one)
    backend_config = [f"-DCELERITY_ENABLE_CUDA_BACKEND={'ON' if backend == 'CUDA' and enabled else 'OFF'}"]
    cmake_args = ["-DCELERITY_DETAIL_INTEGRATION_TESTING=ON"] + backend_config
    subprocess.run(["cmake", celerity_dir] + cmake_args, cwd=build_dir, check=True)
    subprocess.run(["cmake", "--build", ".", "--target", "backend", "-j", str(os.cpu_count())], cwd=build_dir, check=True)

    exe = os.path.join(build_dir, "test/integration/backend")

    # Run once without arguments to get list of available devices
    output = subprocess.check_output(exe).decode()
    print("\nAvailable devices:\n", output)
    devices = [line.strip() for line in output.split("\n") if line.strip()]

    # Try to find a device for the backend as well as an unrelated device
    backend_device_idx = None
    other_device_idx = None
    for i, device in enumerate(devices):
        if (backend is not None and backend.lower() in device.lower()):
            if backend_device_idx is None:
                backend_device_idx = i
        elif other_device_idx is None:
            other_device_idx = i

    if backend is not None and backend_device_idx is None:
        raise RuntimeError(
            f"No matching device found for backend {backend}")

    env = os.environ.copy()
    env["CELERITY_LOG_LEVEL"] = "debug"
    # for unit / system tests, LD_PRELOAD is set by `/root/capture-backtrace.sh`
    if "CI_LD_PRELOAD" in env:
        env["LD_PRELOAD"] = env["CI_LD_PRELOAD"]

    # Check that optimized backend is available (or not) for matching device
    if backend is not None:
        output = subprocess.run([exe, str(backend_device_idx)], stdout=subprocess.PIPE,
                                env=env, check=True).stdout.decode()
        print(output)
        if enabled:
            if f"Using {backend} backend for selected platform" not in output:
                raise RuntimeError(
                    f"Optimized {backend} backend was not selected for device '{devices[backend_device_idx]}'")
        else:
            if f"is compatible with specialized {backend} backend, but it has not been compiled" not in output:
                raise RuntimeError(f"Did not receive warning about {backend} not being available")

    # Check that no optimized backend is available for other device (if one exists)
    if other_device_idx is not None:
        output = subprocess.run([exe, str(other_device_idx)], stdout=subprocess.PIPE,
                                env=env, check=True).stdout.decode()
        print(output)
        if "No backend specialization available" not in output:
            raise RuntimeError("Did not receive warning about no backend specialization being available")


# We currently require an existing build directory to configure and compile
# integration tests. This is a bit hacky but it avoids having to restructure
# the entire CI setup...
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <celerity build directory> <platform>")
        sys.exit(1)
    build_dir = sys.argv[1]
    platform = sys.argv[2]

    # Using backend names like "CUDA" or "Level-Zero" might make more sense here,
    # but the platform is again what we have available in our current CI setup.
    if platform not in ("intel", "nvidia"):
        print(f"Invalid platform '{platform}'; must be 'intel' or 'nvidia'")
        sys.exit(1)

    # Level-Zero is NYI.
    backend = "CUDA" if platform == "nvidia" else None

    # Run once with backend module enabled
    run_backend_discovery_and_copy_test(build_dir, backend, True)

    # And once with module disabled
    if backend is not None:
        run_backend_discovery_and_copy_test(build_dir, backend, False)
