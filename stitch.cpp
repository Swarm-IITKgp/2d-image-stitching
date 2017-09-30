#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
//#include <opencv2/nonfree/features2d.hpp>



using namespace std;
using namespace cv;

int debug = 0;
void readme();


int main( int argc, char** argv )
{

		if( argc != 3)
		{
				readme();
				return -1;
		}


		// Load the images, pass the images as arguments from left to right
		Mat image_l = imread(argv[1],  CV_LOAD_IMAGE_UNCHANGED);
		Mat image_r = imread(argv[2],  CV_LOAD_IMAGE_UNCHANGED);

		
		// Resize each image to a fixed size
		Size size(1024,780);

		resize(image_l,image_l,size);
		resize(image_r,image_r,size);

		

		// Covert to Grayscale
		Mat gray_image_l;
		Mat gray_image_r;
		cvtColor( image_l, gray_image_l, CV_RGB2GRAY );
		cvtColor( image_r, gray_image_r, CV_RGB2GRAY );

		
		if ( !gray_image_l.data || !gray_image_r.data )
		{
				std::cout << " --(!) Error reading images " << std::endl;
				return -1;
		}


		/* 1. Detect the keypoints using a feature detector algorithm like ORB/SURF */

		int minHessian = 400;

		OrbFeatureDetector detector( minHessian );

		std::vector< KeyPoint > keypoints_l, keypoints_r;

		detector.detect( gray_image_l, keypoints_l );
		detector.detect( gray_image_r, keypoints_r );

		

		// In debug mode, show the keypoints of left and right image
		if(debug)
		{
				Mat img_keypoints_left, img_keypoints_right;
				drawKeypoints(gray_image_l, keypoints_l, img_keypoints_left, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
				drawKeypoints(gray_image_r, keypoints_r, img_keypoints_right, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
				imshow("Left Keypoints", img_keypoints_left);
				imshow("Right keypoints", img_keypoints_right);
		}


		/* 2.  Calculate Descriptors */
		
		OrbDescriptorExtractor extractor;

		Mat descriptors_l,descriptors_r;

		extractor.compute( gray_image_l, keypoints_l, descriptors_l );
		extractor.compute( gray_image_r, keypoints_r, descriptors_r );

		// Converting to CV_32F type for use by FLANN matcher
		descriptors_l.convertTo(descriptors_l, CV_32F);
		descriptors_r.convertTo(descriptors_r, CV_32F); 

		/* 3.  Matching descriptor vectors using FLANN matcher */
		
		FlannBasedMatcher matcher;
		std::vector< DMatch > matches;
		matcher.match( descriptors_l, descriptors_r, matches );


		double max_dist = 0;
		double min_dist = 100;

		// Calculation of min-max distances between keypoints
		for(int i =0; i < descriptors_l.rows ; i++)
		{
				double dist = matches[i].distance;
				if( dist < min_dist ) min_dist = dist;
				if( dist > max_dist ) max_dist = dist;
		}

		

		// Use only "good" matches (i.e. whose distance is less than 3 X min_dist )
		std::vector< DMatch > good_matches;

		for(int i =0 ; i < descriptors_l.rows ; i++)
		{
				if( matches[i].distance < 3*min_dist )
				{
						good_matches.push_back( matches[i] );
				}
		}


		// In debug mode, display min/max distance of keypoint and show the (good) matches 
		if(debug)
		{
	
				printf("Max dist = %f \n", max_dist );
				printf("Min dist = %f \n", min_dist );
				Mat img_matches;
				drawMatches(gray_image_l, keypoints_l, gray_image_r, keypoints_r,
					good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
					vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
				imshow("Good matches", img_matches);
		}




		/* 4. Find the Homography Matrix using RANSAC */
		
		std::vector< Point2f > l;
		std::vector< Point2f > r;



		for( int i = 0; i < good_matches.size(); i++)
		{
				// Get the keypoints from the good matches
				l.push_back( keypoints_l[good_matches[i].queryIdx].pt );
				r.push_back( keypoints_r[good_matches[i].trainIdx].pt );
		}

		// Homography is computed from the right image to the left image
		Mat H = findHomography( r, l, CV_RANSAC );

		

		/* 5. Use the homography Matrix to warp the images */
		
		cv::Mat result;

		// Warp the right image into left's perspective and store it in result
		warpPerspective( image_r, result, H, cv::Size( image_l.cols+image_r.cols, image_l.rows) );
		
		// Place the left image in the left half of result
		cv::Mat half(result, cv::Rect(0, 0, image_l.cols, image_l.rows) );
		image_l.copyTo(half);

		// To remove the black portion after stitching, and confine in a rectangular region //

		// Vector with all non-black point positions
		std::vector<cv::Point> nonBlackList;
		nonBlackList.reserve(result.rows*result.cols);

		// Add all non-black points to the vector
		
		for(int j=0; j<result.rows; ++j)
				for(int i=0; i<result.cols; ++i)
				{
						// If not black: add to the list
						if(result.at<cv::Vec3b>(j,i) != cv::Vec3b(0,0,0))
						{
								nonBlackList.push_back(cv::Point(i,j));
						}
				}

		// Create bounding rect around those points
		cv::Rect bb = cv::boundingRect(nonBlackList);

		// Display result and save it
		cv::imshow("Result", result(bb));
		cv::imwrite("./Result.jpg", result(bb));

		

		waitKey(0);
		return 0;
}

// Function readme 
void readme()
{
		std::cout << " Usage: ./a.out < img1 > < img2 > " <<std::endl;
}
