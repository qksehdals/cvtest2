#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/features2d.hpp"
//#include "opencv/cxcore.h"

using namespace std;
using namespace cv;

/*
struct extendedDMatch{
	DMatch match;
	Point2i queryAddr;
	Point2i trainAddr;

};

typedef struct extendedDMatch{
	DMatch match;
	Point2i queryAddr;
	Point2i trainAddr;
} eDMatch;
*/

void main(){
	int index = 0;
START:
	std::string str1, str2;
	str1 = "c:/cvimagetest/white_img_org.png";
	index += 1;
	str2 = "c:/cvimagetest/";
	str2 += to_string(index);
	str2 += ".png";
	cout << "processing " << str2 << "..." << endl;
	//load image
	Mat image_template;
	Mat image1, image2;
	image1 = cv::imread(str1, CV_LOAD_IMAGE_COLOR);
	image2 = cv::imread(str2, CV_LOAD_IMAGE_COLOR);
	image_template = cv::imread(str2, CV_LOAD_IMAGE_COLOR);

	//exeption handling
	if (image1.empty()){
		cout << "파일 읽기 실패" << endl;
		exit(-1);
	}
	if (image2.empty()){
		cout << "파일 읽기 실패" << endl;
		exit(-1);
	}

	//apply Canny
	Mat canny1, canny2;
	Canny(image1, canny1, 50, 100);
	Canny(image2, canny2, 50, 100);

	//sift feature detect (get coordinate only?)
	SiftFeatureDetector siftdetector;

	vector<KeyPoint> siftkeypoints1, siftkeypoints2;
	siftdetector.detect(canny1, siftkeypoints1);
	drawKeypoints(canny1, siftkeypoints1, canny1);
	siftdetector.detect(canny2, siftkeypoints2);
	drawKeypoints(canny2, siftkeypoints2, canny2);
	
	//sift feature extract
	SiftDescriptorExtractor extractor;
	Mat descriptors1, descriptors2;
	extractor.compute(canny1, siftkeypoints1, descriptors1);
	extractor.compute(canny2, siftkeypoints2, descriptors2);
	
	
	//match features (brute force)
	BFMatcher siftmatcher;
	//FlannBasedMatcher matcher;
	std::vector<DMatch> siftmatches;
	siftmatcher.match(descriptors1, descriptors2, siftmatches);
	
	/*
	int chunksize1 = MIN(image1.rows, image1.cols) / 15;
	int grid_rows1 = (image1.rows / chunksize1) + 1;
	int grid_cols1 = (image1.cols / chunksize1) + 1;

	
	vector<DMatch>** image1_block = (vector<DMatch>**) malloc(sizeof(vector<DMatch>*) * MIN(image1.rows, image1.cols));
	vector<DMatch> siftmatches_copied(siftmatches);
	free(image1_block);
	*/
	
	std::vector<DMatch> slope_matches;
	for (int i = 0; i < (int)siftmatches.size(); i++){
		if (siftmatches.at(i).distance < 250){
			slope_matches.push_back(siftmatches.at(i));
			DMatch cur = slope_matches.back();

			//float dx = (image1.cols - siftkeypoints1.at(cur.queryIdx).pt.x + siftkeypoints2.at(cur.trainIdx).pt.x);
			//float dy = (image1.rows - siftkeypoints1.at(cur.queryIdx).pt.y + siftkeypoints2.at(cur.trainIdx).pt.y);
			int dy = ( (-siftkeypoints1.at(cur.queryIdx).pt.y) + siftkeypoints2.at(cur.trainIdx).pt.y);
			//printf("%f \n", (dx / dy));
			if (!(dy < 5 && dy > -5)) slope_matches.pop_back();
		}
	}
	//cout << slope_matches.size() << endl;

	
	std::vector<DMatch> good_matches(slope_matches);
	/*
	double max_dist = 0;
	double min_dist = 100;
	
	for (int i = 0; i < descriptors1.rows; i++){
		double dist = siftmatches[i].distance;

		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}

	printf("max dist : %f \n", max_dist);
	printf("min dist : %f \n", min_dist);

	srand(time(NULL));
	for (int i = 0; i < descriptors1.rows; i++){
		//if (siftmatches[i].distance < 300){
		if (rand()%5 == 0){
			good_matches.push_back(siftmatches[i]);
		}
	}
	*/

	Mat view;
	drawMatches(image1, siftkeypoints1, image2, siftkeypoints2, good_matches, view, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	

	double thres = 3.0;
//B1:
	vector<Point2f> obj;
	vector<Point2f> scene;
	for (unsigned i = 0; i < good_matches.size(); i++){
		obj.push_back(siftkeypoints1[good_matches[i].queryIdx].pt);
		scene.push_back(siftkeypoints2[good_matches[i].trainIdx].pt);
	}

	Mat H = findHomography(obj, scene, CV_RANSAC, thres);

	vector<Point2f> obj_corners(4);
	obj_corners[0] = cvPoint(0, 0);
	obj_corners[1] = cvPoint(image1.cols, 0);
	obj_corners[2] = cvPoint(image1.cols, image1.rows);
	obj_corners[3] = cvPoint(0, image1.rows);

	vector<Point2f> scene_corners(4);
	perspectiveTransform(obj_corners, scene_corners, H);
	
	/*
	cout << scene_corners[0] << endl;
	cout << scene_corners[1] << endl;
	cout << scene_corners[2] << endl;
	cout << scene_corners[3] << endl;
	*/
	//Draw lines between the corners (the mapped object in the scene - image_2 )
	line(view, scene_corners[0] + Point2f(image1.cols, 0), scene_corners[1] + Point2f(image1.cols, 0), Scalar(0, 255, 0), 4);
	line(view, scene_corners[1] + Point2f(image1.cols, 0), scene_corners[2] + Point2f(image1.cols, 0), Scalar(0, 255, 0), 4);
	line(view, scene_corners[2] + Point2f(image1.cols, 0), scene_corners[3] + Point2f(image1.cols, 0), Scalar(0, 255, 0), 4);
	line(view, scene_corners[3] + Point2f(image1.cols, 0), scene_corners[0] + Point2f(image1.cols, 0), Scalar(0, 255, 0), 4);
	
	Mat W = getPerspectiveTransform(obj_corners, scene_corners);
	Mat warped_image;
	warpPerspective(image2, warped_image, W.inv(), Size(image1.cols, image1.rows));

	Vec3b seoul = warped_image.at<Vec3b>(Point(215, 160));
	Vec3b gyeonggi = warped_image.at<Vec3b>(Point(230, 190));
	Vec3b incheon = warped_image.at<Vec3b>(Point(170, 140));
	Vec3b daejeon = warped_image.at<Vec3b>(Point(245, 275));
	Vec3b sejong = warped_image.at<Vec3b>(Point(235, 255));
	Vec3b daegu = warped_image.at<Vec3b>(Point(341, 322));
	Vec3b ulsan = warped_image.at<Vec3b>(Point(390, 350));
	Vec3b busan = warped_image.at<Vec3b>(Point(380, 382));
	Vec3b gwangju = warped_image.at<Vec3b>(Point(203, 388));
	
	Vec3b gangwon = warped_image.at<Vec3b>(Point(311, 137));
	Vec3b chungbuk = warped_image.at<Vec3b>(Point(268, 233));
	Vec3b chungnam = warped_image.at<Vec3b>(Point(200, 262));
	Vec3b jeonbuk = warped_image.at<Vec3b>(Point(228, 333));
	Vec3b jeonnam = warped_image.at<Vec3b>(Point(200, 425));
	Vec3b gyeongbuk = warped_image.at<Vec3b>(Point(350, 260));
	Vec3b gyeongnam = warped_image.at<Vec3b>(Point(310, 375));
	Vec3b jeju = warped_image.at<Vec3b>(Point(180, 560));

	
	circle(warped_image, Point(215,160), 3, Scalar(255, 0, 0));
	circle(warped_image, Point(230, 190), 3, Scalar(255, 0, 0));
	circle(warped_image, Point(170, 140), 3, Scalar(255, 0, 0));
	circle(warped_image, Point(245, 275), 3, Scalar(255, 0, 0));
	circle(warped_image, Point(235, 255), 3, Scalar(255, 0, 0));
	circle(warped_image, Point(341, 322), 3, Scalar(255, 0, 0));
	circle(warped_image, Point(390, 350), 3, Scalar(255, 0, 0));
	circle(warped_image, Point(380, 382), 3, Scalar(255, 0, 0));
	circle(warped_image, Point(203, 388), 3, Scalar(255, 0, 0));

	circle(warped_image, Point(311, 137), 3, Scalar(255, 0, 0));
	circle(warped_image, Point(268, 233), 3, Scalar(255, 0, 0));
	circle(warped_image, Point(200, 262), 3, Scalar(255, 0, 0));
	circle(warped_image, Point(228, 333), 3, Scalar(255, 0, 0));
	circle(warped_image, Point(200, 425), 3, Scalar(255, 0, 0));
	circle(warped_image, Point(350, 260), 3, Scalar(255, 0, 0));
	circle(warped_image, Point(310, 375), 3, Scalar(255, 0, 0));
	circle(warped_image, Point(180, 560), 3, Scalar(255, 0, 0));

	cout << "result of " << str2 << endl;
	cout << "서울 : " << ((seoul[0] + seoul[1] + seoul[2])/3) << " " << seoul << endl;
	cout << "경기 : " << ((gyeonggi[0] + gyeonggi[1] + gyeonggi[2]) / 3) << " " << gyeonggi << endl;
	cout << "인천 : " << ((incheon[0] + incheon[1] + incheon[2]) / 3) << " " << incheon << endl;
	cout << "대전 : " << ((daejeon[0] + daejeon[1] + daejeon[2]) / 3) << " " << daejeon << endl;
	cout << "세종 : " << ((sejong[0] + sejong[1] + sejong[2]) / 3) << " " << sejong << endl;
	cout << "대구 : " << ((daegu[0] + daegu[1] + daegu[2]) / 3) << " " << daegu << endl;
	cout << "울산 : " << ((ulsan[0] + ulsan[1] + ulsan[2]) / 3) << " " << ulsan << endl;
	cout << "부산 : " << ((busan[0] + busan[1] + busan[2]) / 3) << " " << busan << endl;
	cout << "광주 : " << ((gwangju[0] + gwangju[1] + gwangju[2]) / 3) << " " << gwangju << endl;

	cout << "강원 : " << ((gangwon[0] + gangwon[1] + gangwon[2]) / 3) << " " << gangwon << endl;
	cout << "충북 : " << ((chungbuk[0] + chungbuk[1] + chungbuk[2]) / 3) << " " << chungbuk << endl;
	cout << "충남 : " << ((chungnam[0] + chungnam[1] + chungnam[2]) / 3) << " " << chungnam << endl;
	cout << "전북 : " << ((jeonbuk[0] + jeonbuk[1] + jeonbuk[2]) / 3) << " " << jeonbuk << endl;
	cout << "전남 : " << ((jeonnam[0] + jeonnam[1] + jeonnam[2]) / 3) << " " << jeonnam << endl;
	cout << "경북 : " << ((gyeongbuk[0] + gyeongbuk[1] + gyeongbuk[2]) / 3) << " " << gyeongbuk << endl;
	cout << "경남 : " << ((gyeongnam[0] + gyeongnam[1] + gyeongnam[2]) / 3) << " " << gyeongnam << endl;
	cout << "제주 : " << ((jeju[0] + jeju[1] + jeju[2]) / 3) << " " << jeju << endl;

	imshow("warped", warped_image);
	imshow("view", view);
	cv::waitKey(0);
	goto START;

}

