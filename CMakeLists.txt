cmake_minimum_required(VERSION 2.8)
project( 2d-image-stitching )
find_package( OpenCV REQUIRED )
add_executable( stitch stitch.cpp )
target_link_libraries( stitch ${OpenCV_LIBS} )
