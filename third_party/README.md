# Third-party dependencies

This repository does not vendor third-party source for Milestone 0.

CMake downloads GLFW, Dear ImGui, glad, and nlohmann/json into the build tree through `FetchContent`. CUDA and OpenGL are resolved from the local system/toolkit.

If future milestones copy or adapt code from NVIDIA CUDA Samples or Lenia3D, record that in a notice file and preserve any required upstream copyright or license headers.
