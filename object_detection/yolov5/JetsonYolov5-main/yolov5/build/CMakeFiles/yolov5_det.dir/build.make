# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/matteo/anaconda3/envs/tensorRT/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/matteo/anaconda3/envs/tensorRT/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/build

# Include any dependencies generated for this target.
include CMakeFiles/yolov5_det.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/yolov5_det.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/yolov5_det.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/yolov5_det.dir/flags.make

CMakeFiles/yolov5_det.dir/yolov5_det.cpp.o: CMakeFiles/yolov5_det.dir/flags.make
CMakeFiles/yolov5_det.dir/yolov5_det.cpp.o: /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/yolov5_det.cpp
CMakeFiles/yolov5_det.dir/yolov5_det.cpp.o: CMakeFiles/yolov5_det.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/yolov5_det.dir/yolov5_det.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/yolov5_det.dir/yolov5_det.cpp.o -MF CMakeFiles/yolov5_det.dir/yolov5_det.cpp.o.d -o CMakeFiles/yolov5_det.dir/yolov5_det.cpp.o -c /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/yolov5_det.cpp

CMakeFiles/yolov5_det.dir/yolov5_det.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolov5_det.dir/yolov5_det.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/yolov5_det.cpp > CMakeFiles/yolov5_det.dir/yolov5_det.cpp.i

CMakeFiles/yolov5_det.dir/yolov5_det.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolov5_det.dir/yolov5_det.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/yolov5_det.cpp -o CMakeFiles/yolov5_det.dir/yolov5_det.cpp.s

CMakeFiles/yolov5_det.dir/src/calibrator.cpp.o: CMakeFiles/yolov5_det.dir/flags.make
CMakeFiles/yolov5_det.dir/src/calibrator.cpp.o: /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/src/calibrator.cpp
CMakeFiles/yolov5_det.dir/src/calibrator.cpp.o: CMakeFiles/yolov5_det.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/yolov5_det.dir/src/calibrator.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/yolov5_det.dir/src/calibrator.cpp.o -MF CMakeFiles/yolov5_det.dir/src/calibrator.cpp.o.d -o CMakeFiles/yolov5_det.dir/src/calibrator.cpp.o -c /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/src/calibrator.cpp

CMakeFiles/yolov5_det.dir/src/calibrator.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolov5_det.dir/src/calibrator.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/src/calibrator.cpp > CMakeFiles/yolov5_det.dir/src/calibrator.cpp.i

CMakeFiles/yolov5_det.dir/src/calibrator.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolov5_det.dir/src/calibrator.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/src/calibrator.cpp -o CMakeFiles/yolov5_det.dir/src/calibrator.cpp.s

CMakeFiles/yolov5_det.dir/src/model.cpp.o: CMakeFiles/yolov5_det.dir/flags.make
CMakeFiles/yolov5_det.dir/src/model.cpp.o: /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/src/model.cpp
CMakeFiles/yolov5_det.dir/src/model.cpp.o: CMakeFiles/yolov5_det.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/yolov5_det.dir/src/model.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/yolov5_det.dir/src/model.cpp.o -MF CMakeFiles/yolov5_det.dir/src/model.cpp.o.d -o CMakeFiles/yolov5_det.dir/src/model.cpp.o -c /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/src/model.cpp

CMakeFiles/yolov5_det.dir/src/model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolov5_det.dir/src/model.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/src/model.cpp > CMakeFiles/yolov5_det.dir/src/model.cpp.i

CMakeFiles/yolov5_det.dir/src/model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolov5_det.dir/src/model.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/src/model.cpp -o CMakeFiles/yolov5_det.dir/src/model.cpp.s

CMakeFiles/yolov5_det.dir/src/postprocess.cpp.o: CMakeFiles/yolov5_det.dir/flags.make
CMakeFiles/yolov5_det.dir/src/postprocess.cpp.o: /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/src/postprocess.cpp
CMakeFiles/yolov5_det.dir/src/postprocess.cpp.o: CMakeFiles/yolov5_det.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/yolov5_det.dir/src/postprocess.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/yolov5_det.dir/src/postprocess.cpp.o -MF CMakeFiles/yolov5_det.dir/src/postprocess.cpp.o.d -o CMakeFiles/yolov5_det.dir/src/postprocess.cpp.o -c /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/src/postprocess.cpp

CMakeFiles/yolov5_det.dir/src/postprocess.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/yolov5_det.dir/src/postprocess.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/src/postprocess.cpp > CMakeFiles/yolov5_det.dir/src/postprocess.cpp.i

CMakeFiles/yolov5_det.dir/src/postprocess.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/yolov5_det.dir/src/postprocess.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/src/postprocess.cpp -o CMakeFiles/yolov5_det.dir/src/postprocess.cpp.s

CMakeFiles/yolov5_det.dir/src/preprocess.cu.o: CMakeFiles/yolov5_det.dir/flags.make
CMakeFiles/yolov5_det.dir/src/preprocess.cu.o: CMakeFiles/yolov5_det.dir/includes_CUDA.rsp
CMakeFiles/yolov5_det.dir/src/preprocess.cu.o: /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/src/preprocess.cu
CMakeFiles/yolov5_det.dir/src/preprocess.cu.o: CMakeFiles/yolov5_det.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CUDA object CMakeFiles/yolov5_det.dir/src/preprocess.cu.o"
	/usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT CMakeFiles/yolov5_det.dir/src/preprocess.cu.o -MF CMakeFiles/yolov5_det.dir/src/preprocess.cu.o.d -x cu -c /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/src/preprocess.cu -o CMakeFiles/yolov5_det.dir/src/preprocess.cu.o

CMakeFiles/yolov5_det.dir/src/preprocess.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/yolov5_det.dir/src/preprocess.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/yolov5_det.dir/src/preprocess.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/yolov5_det.dir/src/preprocess.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target yolov5_det
yolov5_det_OBJECTS = \
"CMakeFiles/yolov5_det.dir/yolov5_det.cpp.o" \
"CMakeFiles/yolov5_det.dir/src/calibrator.cpp.o" \
"CMakeFiles/yolov5_det.dir/src/model.cpp.o" \
"CMakeFiles/yolov5_det.dir/src/postprocess.cpp.o" \
"CMakeFiles/yolov5_det.dir/src/preprocess.cu.o"

# External object files for target yolov5_det
yolov5_det_EXTERNAL_OBJECTS =

yolov5_det: CMakeFiles/yolov5_det.dir/yolov5_det.cpp.o
yolov5_det: CMakeFiles/yolov5_det.dir/src/calibrator.cpp.o
yolov5_det: CMakeFiles/yolov5_det.dir/src/model.cpp.o
yolov5_det: CMakeFiles/yolov5_det.dir/src/postprocess.cpp.o
yolov5_det: CMakeFiles/yolov5_det.dir/src/preprocess.cu.o
yolov5_det: CMakeFiles/yolov5_det.dir/build.make
yolov5_det: libmyplugins.so
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_dnn_objdetect.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_dnn_superres.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_face.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_hfs.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_img_hash.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_quality.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_tracking.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_text.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_dnn.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_video.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.4.2.0
yolov5_det: /usr/lib/x86_64-linux-gnu/libopencv_core.so.4.2.0
yolov5_det: CMakeFiles/yolov5_det.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable yolov5_det"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/yolov5_det.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/yolov5_det.dir/build: yolov5_det
.PHONY : CMakeFiles/yolov5_det.dir/build

CMakeFiles/yolov5_det.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/yolov5_det.dir/cmake_clean.cmake
.PHONY : CMakeFiles/yolov5_det.dir/clean

CMakeFiles/yolov5_det.dir/depend:
	cd /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5 /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5 /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/build /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/build /home/matteo/Documents/unibo/Tesi/Ubicomp_2023/code/MatteoQuantizing/object_detection/yolov5/JetsonYolov5-main/yolov5/build/CMakeFiles/yolov5_det.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/yolov5_det.dir/depend

