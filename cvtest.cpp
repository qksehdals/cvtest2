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

	Mat image;
	image = cv::imread(str, CV_LOAD_IMAGE_COLOR);

	if (image.empty()){
		cout << "파일 읽기 실패" << endl;
	}

	Mat canny;
	Canny(image, canny, 50, 100);

	SiftDescriptorExtractor detector;
	vector<KeyPoint> keypoints;
	detector.detect(canny, keypoints);
	drawKeypoints(canny, keypoints, canny);

	BFMatcher matcher;
	int a = 123;
	//matcher.match()



	cv::imshow("영상확인", canny);
	cv::waitKey(0);
}