#include "opencv2/highgui/highgui.hpp"
//#include "opencv/cxcore.h"
#include "opencv/cv.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/features2d.hpp"

using namespace std;
using namespace cv;

void main(){

	std::string str;
	str = "c:/cvimagetest/test_img_org.png";

	//load image
	Mat image;
	image = cv::imread(str, CV_LOAD_IMAGE_COLOR);

	//exeption handling
	if (image.empty()){
		cout << "파일 읽기 실패" << endl;
	}


	//apply Canny
	Mat canny;
	Canny(image, canny, 50, 100);

	//sift image descriptor
	SiftDescriptorExtractor detector;
	vector<KeyPoint> keypoints;
	detector.detect(canny, keypoints);
	drawKeypoints(canny, keypoints, canny);

	//match features
	BFMatcher matcher;

	//matcher.match()



	cv::imshow("영상확인", canny);
	cv::waitKey(0);
}