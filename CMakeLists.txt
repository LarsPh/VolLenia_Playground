cmake_minimum_required(VERSION 3.24)

project(VolLenia_Playground
    VERSION 0.1.0
    DESCRIPTION "Volumetric 3D Lenia playground bootstrap"
    LANGUAGES C CXX CUDA
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

include(FetchContent)
include(GNUInstallDirs)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(VOLLENIA_PYTHON_VENV "${CMAKE_BINARY_DIR}/python-venv")
if(WIN32)
    set(VOLLENIA_VENV_PYTHON "${VOLLENIA_PYTHON_VENV}/Scripts/python.exe")
else()
    set(VOLLENIA_VENV_PYTHON "${VOLLENIA_PYTHON_VENV}/bin/python")
endif()

if(NOT EXISTS "${VOLLENIA_VENV_PYTHON}")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m venv "${VOLLENIA_PYTHON_VENV}"
        RESULT_VARIABLE VOLLENIA_VENV_RESULT
    )
    if(NOT VOLLENIA_VENV_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to create Python venv for glad generator")
    endif()
endif()

execute_process(
    COMMAND "${VOLLENIA_VENV_PYTHON}" -c "import jinja2"
    RESULT_VARIABLE VOLLENIA_JINJA2_IMPORT_RESULT
    OUTPUT_QUIET
    ERROR_QUIET
)
if(NOT VOLLENIA_JINJA2_IMPORT_RESULT EQUAL 0)
    execute_process(
        COMMAND "${VOLLENIA_VENV_PYTHON}" -m pip install --disable-pip-version-check "jinja2>=3.1,<4"
        RESULT_VARIABLE VOLLENIA_JINJA2_INSTALL_RESULT
    )
    if(NOT VOLLENIA_JINJA2_INSTALL_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to install jinja2 into build-local Python venv for glad")
    endif()
endif()
set(Python_EXECUTABLE "${VOLLENIA_VENV_PYTHON}" CACHE FILEPATH "Python interpreter used by glad" FORCE)

set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(NFD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NFD_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4
)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.92.8
)

FetchContent_Declare(
    glad
    GIT_REPOSITORY https://github.com/Dav1dde/glad.git
    GIT_TAG v2.0.8
    SOURCE_SUBDIR cmake
)

FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)

FetchContent_Declare(
    nativefiledialog_extended
    GIT_REPOSITORY https://github.com/btzy/nativefiledialog-extended.git
    GIT_TAG v1.3.0
)

FetchContent_MakeAvailable(glfw imgui glad nlohmann_json nativefiledialog_extended)

glad_add_library(glad_gl_core_33 STATIC REPRODUCIBLE API gl:core=3.3)

add_library(imgui_backend STATIC
    "${imgui_SOURCE_DIR}/imgui.cpp"
    "${imgui_SOURCE_DIR}/imgui_draw.cpp"
    "${imgui_SOURCE_DIR}/imgui_tables.cpp"
    "${imgui_SOURCE_DIR}/imgui_widgets.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp"
    "${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp"
)

target_include_directories(imgui_backend PUBLIC
    "${imgui_SOURCE_DIR}"
    "${imgui_SOURCE_DIR}/backends"
)

target_compile_definitions(imgui_backend PUBLIC
    IMGUI_IMPL_OPENGL_LOADER_GLAD2
)

target_link_libraries(imgui_backend PUBLIC
    glfw
    glad_gl_core_33
    OpenGL::GL
)

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/core")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/Version.h.in"
    "${CMAKE_CURRENT_BINARY_DIR}/generated/core/Version.h"
    @ONLY
)

add_executable(VolLenia_Playground
    src/main.cpp
    src/app/App.cpp
    src/app/Camera.cpp
    src/app/UiPanel.cpp
    src/io/CellResampler.cu
    src/io/CellVolumeFile.cpp
    src/io/LeniaAnimalCatalog.cpp
    src/model/ModelSpec.cpp
    src/render/CudaVolumeRenderer.cu
    src/render/CudaPbo.cpp
    src/render/GlDisplay.cpp
    src/render/PboSmokeTest.cu
    src/render/SyntheticVolume.cu
    src/sim/DeviceMultiVolume.cu
    src/sim/DeviceVolume.cu
    src/sim/ExpandedFlowSimulation.cu
    src/sim/FlowTransport.cu
    src/sim/KernelBank.cu
    src/sim/LeniaSeeds.cu
    src/sim/LeniaSimulation.cu
)

target_include_directories(VolLenia_Playground PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
)

target_compile_definitions(VolLenia_Playground PRIVATE
    GLFW_INCLUDE_NONE
)

target_link_libraries(VolLenia_Playground PRIVATE
    CUDA::cudart
    CUDA::cufft
    OpenGL::GL
    glfw
    glad_gl_core_33
    imgui_backend
    nfd
    nlohmann_json::nlohmann_json
)

if(MSVC)
    target_compile_options(VolLenia_Playground PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive->
    )
else()
    target_compile_options(VolLenia_Playground PRIVATE -Wall -Wextra -Wpedantic)
endif()
