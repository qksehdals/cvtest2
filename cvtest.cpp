#include "opencv2/highgui/highgui.hpp"
#include "opencv/cv.h"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/nonfree/features2d.hpp"

#include <iostream>
#include <fstream>
//#include "opencv/cxcore.h"

using namespace std;
using namespace cv;


/*	function prototypes	*/
bool is_white(cv::Vec3b);
void imshow_seq(vector<cv::Mat>);
vector<cv::Mat> load_mask(void);
vector<cv::Vec3b> get_values_from_map(Mat image, vector<cv::Mat> mask_set, bool print);
vector<int> make_grayscale_value(vector<cv::Vec3b> input);
void print_same_region(vector<int> input, int index);


/*	Main function	*/
void main(){
	
	int index = 0;
START:
	std::string str1, str2;
	str1 = "c:/cvimagetest/template.png";
	index += 1;
	str2 = "c:/cvimagetest/new_chart/";
	str2 += to_string(index);
	str2 += ".png";
	
	ofstream output("c:/cvimagetest/output.txt", ios::app);
	output << to_string(index) << ".png" << endl;
	

	cout << to_string(index) << ".png" << endl;
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

	//Draw lines between corners
	line(view, scene_corners[0] + Point2f(image1.cols, 0), scene_corners[1] + Point2f(image1.cols, 0), Scalar(0, 255, 0), 4);
	line(view, scene_corners[1] + Point2f(image1.cols, 0), scene_corners[2] + Point2f(image1.cols, 0), Scalar(0, 255, 0), 4);
	line(view, scene_corners[2] + Point2f(image1.cols, 0), scene_corners[3] + Point2f(image1.cols, 0), Scalar(0, 255, 0), 4);
	line(view, scene_corners[3] + Point2f(image1.cols, 0), scene_corners[0] + Point2f(image1.cols, 0), Scalar(0, 255, 0), 4);
	
	Mat W = getPerspectiveTransform(obj_corners, scene_corners);
	Mat warped_image;
	warpPerspective(image2, warped_image, W.inv(), Size(image1.cols, image1.rows));
	
	//imshow("view", view);
	//imshow("canny1", canny1);
	//imshow("canny2", canny2);

	/*
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
	*/
	
	/*			지역코드		*/
	/*	서울 : 0	충북 : 10	*/
	/*	경기 : 1	충남 : 11	*/
	/*	인천 : 2	전북 : 12	*/
	/*	대전 : 3	전남 : 13	*/
	/*	세종 : 4	경북 : 14	*/
	/*	대구 : 5	경남 : 15	*/
	/*	울산 : 6	제주 : 16	*/
	/*	부산 : 7	---------	*/
	/*	광주 : 8	---------	*/
	/*	강원 : 9	---------	*/

	imshow("result", warped_image);

	vector<Mat> mask_set = load_mask();

	//cout << "result of original image" << endl;
	//get_values_from_map(image2, mask_set, true);
	vector<Vec3b> regional_result = get_values_from_map(warped_image, mask_set, true);
	
	cout << endl;	
	output << endl;
	output.close();
	
	vector<int> grayscale_result = make_grayscale_value(regional_result);

	//cout << grayscale_result.size() << endl;

	print_same_region(grayscale_result, 10);
	waitKey(0);
goto START;

}


bool is_white(cv::Vec3b pixel){
	if (pixel[0] == 255){
		if (pixel[1] == 255){
			if (pixel[2] == 255){
				return true;
			}
		}
	}
	return false;
}

void imshow_seq(vector<cv::Mat> image_set){
	for (int i = 0; i < image_set.size(); i++){
		imshow("mask", image_set.at(i));
		waitKey(0);
	}
}

vector<cv::Mat> load_mask(void){

	vector<cv::Mat> mask_set;
	mask_set.clear();

	std::string mask_names[17] = { "mask_su", "mask_gg", "mask_ic", "mask_dj", 
		"mask_sj", "mask_dg", "mask_us", "mask_bs", "mask_gj", "mask_gw", 
		"mask_cb", "mask_cn", "mask_jb", "mask_jn", "mask_gb", "mask_gn", 
		"mask_jj" };

	/*			지역코드		*/
	/*	서울 : 0	충북 : 10	*/
	/*	경기 : 1	충남 : 11	*/
	/*	인천 : 2	전북 : 12	*/
	/*	대전 : 3	전남 : 13	*/
	/*	세종 : 4	경북 : 14	*/
	/*	대구 : 5	경남 : 15	*/
	/*	울산 : 6	제주 : 16	*/
	/*	부산 : 7	---------	*/
	/*	광주 : 8	---------	*/
	/*	강원 : 9	---------	*/	
	cv::Mat mask;
	for (int i = 0; i < 17; i++){
		std::string path = "c:/cvimagetest/mask/";
		path += mask_names[i];
		path += ".png";
		mask = cv::imread(path, CV_LOAD_IMAGE_COLOR);
		if (mask.empty()){
			cout << "Failed to read mask file : " << path << endl;
			exit(-1);
		}

		mask_set.push_back(mask);
	}

	return mask_set;
}

vector<cv::Vec3b> get_values_from_map(Mat image, vector<cv::Mat> mask_set, bool print){
	vector<cv::Vec3b> result;
	result.clear();
	//
	vector<int> pixels_dominant;
	pixels_dominant.clear();
	//
	/*
	vector<int> mask_ground_vec;
	mask_ground_vec.clear();
	*/


	/*			지역코드		*/
	/*	서울 : 0	충북 : 10	*/
	/*	경기 : 1	충남 : 11	*/
	/*	인천 : 2	전북 : 12	*/
	/*	대전 : 3	전남 : 13	*/
	/*	세종 : 4	경북 : 14	*/
	/*	대구 : 5	경남 : 15	*/
	/*	울산 : 6	제주 : 16	*/
	/*	부산 : 7	---------	*/
	/*	광주 : 8	---------	*/
	/*	강원 : 9	---------	*/
	for (int i = 0; i < mask_set.size(); i++){
		cv::Mat mask_cur = mask_set[i];
		vector<cv::Vec3b> colors;
		vector<int> counts;
		colors.clear();
		counts.clear();
	
		//int mask_ground_counter = 0;

		for (int m = 0; m < mask_cur.cols; m++){
			for (int n = 0; n < mask_cur.rows; n++){
				Vec3b color_cur = Vec3b(mask_cur.at<Vec3b>(m, n)); // mask color at (m,n) (black or white)

				if ( is_white(color_cur) ){
					//mask_ground_counter += 1;
					if (colors.size() == 0){
						colors.push_back(Vec3b(image.at<Vec3b>(m, n)));
						counts.push_back(0);
					}
					else{
						for (int k = 0; k < colors.size(); k++){
							if (colors[k] == Vec3b(image.at<Vec3b>(m, n))){
								counts[k] += 1;
								break;
							}else{
								if (k == colors.size() - 1){
									colors.push_back(Vec3b(image.at<Vec3b>(m, n)));
									counts.push_back(0);
								}
							}
						}
					}
				}
				
			}

		}

		if (colors.size() != counts.size()) {
			cout << "may be error!" << endl;
			exit(-1);
		}
		
		Vec3b dominant_color;
		int dominant_index = 0;
		for (int p = 0; p < counts.size(); p++){
			
			if (counts[p] > counts[dominant_index]){
				dominant_index = p;
			}
			
		}
		dominant_color = colors[dominant_index];

		result.push_back(dominant_color);
		//
		pixels_dominant.push_back(counts[dominant_index]);


		//mask_ground_vec.push_back(mask_ground_counter);
	}

	/*
	for (int i = 0; i < mask_ground_vec.size(); i++){
		cout << mask_ground_vec[i] << endl;
	}
	*/
	
	int mask_ground[17] = {325, 6305, 478, 300, 248, 491, 616, 362, 277, 10631, 4509, 5023, 4868, 6797, 11913, 6281, 1132};
	if (print){
		/*
		cout << "지역명\t" << "grayscale\t" << "[B,G,R]\t" << "valid_pixels\t" << "error(%)\t" << endl;
		cout << "서울\t" << ((result[0][0] + result[0][1] + result[0][2]) / 3) << "\t" << result[0] << "\t" << pixels_dominant[0] << "\t" <<(((float)(mask_ground[0] - pixels_dominant[0]))/(float)mask_ground[0]) * 100 << "%" << endl;
		cout << "경기\t" << ((result[1][0] + result[1][1] + result[1][2]) / 3) << "\t" << result[1] << "\t" << pixels_dominant[1] << "\t" << (((float)(mask_ground[1] - pixels_dominant[1])) / (float)mask_ground[1]) * 100 << "%" << endl;
		cout << "인천\t" << ((result[2][0] + result[2][1] + result[2][2]) / 3) << "\t" << result[2] << "\t" << pixels_dominant[2] << "\t" << (((float)(mask_ground[2] - pixels_dominant[2])) / (float)mask_ground[2]) * 100 << "%" << endl;
		cout << "대전\t" << ((result[3][0] + result[3][1] + result[3][2]) / 3) << "\t" << result[3] << "\t" << pixels_dominant[3] << "\t" << (((float)(mask_ground[3] - pixels_dominant[3])) / (float)mask_ground[3]) * 100 << "%" << endl;
		cout << "세종\t" << ((result[4][0] + result[4][1] + result[4][2]) / 3) << "\t" << result[4] << "\t" << pixels_dominant[4] << "\t" << (((float)(mask_ground[4] - pixels_dominant[4])) / (float)mask_ground[4]) * 100 << "%" << endl;
		cout << "대구\t" << ((result[5][0] + result[5][1] + result[5][2]) / 3) << "\t" << result[5] << "\t" << pixels_dominant[5] << "\t" << (((float)(mask_ground[5] - pixels_dominant[5])) / (float)mask_ground[5]) * 100 << "%" << endl;
		cout << "울산\t" << ((result[6][0] + result[6][1] + result[6][2]) / 3) << "\t" << result[6] << "\t" << pixels_dominant[6] << "\t" << (((float)(mask_ground[6] - pixels_dominant[6])) / (float)mask_ground[6]) * 100 << "%" << endl;
		cout << "부산\t" << ((result[7][0] + result[7][1] + result[7][2]) / 3) << "\t" << result[7] << "\t" << pixels_dominant[7] << "\t" << (((float)(mask_ground[7] - pixels_dominant[7])) / (float)mask_ground[7]) * 100 << "%" << endl;
		cout << "광주\t" << ((result[8][0] + result[8][1] + result[8][2]) / 3) << "\t" << result[8] << "\t" << pixels_dominant[8] << "\t" << (((float)(mask_ground[8] - pixels_dominant[8])) / (float)mask_ground[8]) * 100 << "%" << endl;

		cout << "강원\t" << ((result[9][0] + result[9][1] + result[9][2]) / 3) << "\t" << result[9] << "\t" << pixels_dominant[9] << "\t" << (((float)(mask_ground[9] - pixels_dominant[9])) / (float)mask_ground[9]) * 100 << "%" << endl;
		cout << "충북\t" << ((result[10][0] + result[10][1] + result[10][2]) / 3) << "\t" << result[10] << "\t" << pixels_dominant[10] << "\t" << (((float)(mask_ground[10] - pixels_dominant[10])) / (float)mask_ground[10]) * 100 << "%" << endl;
		cout << "충남\t" << ((result[11][0] + result[11][1] + result[11][2]) / 3) << "\t" << result[11] << "\t" << pixels_dominant[11] << "\t" << (((float)(mask_ground[11] - pixels_dominant[11])) / (float)mask_ground[11]) * 100 << "%" << endl;
		cout << "전북\t" << ((result[12][0] + result[12][1] + result[12][2]) / 3) << "\t" << result[12] << "\t" << pixels_dominant[12] << "\t" << (((float)(mask_ground[12] - pixels_dominant[12])) / (float)mask_ground[12]) * 100 << "%" << endl;
		cout << "전남\t" << ((result[13][0] + result[13][1] + result[13][2]) / 3) << "\t" << result[13] << "\t" << pixels_dominant[13] << "\t" << (((float)(mask_ground[13] - pixels_dominant[13])) / (float)mask_ground[13]) * 100 << "%" << endl;
		cout << "경북\t" << ((result[14][0] + result[14][1] + result[14][2]) / 3) << "\t" << result[14] << "\t" << pixels_dominant[14] << "\t" << (((float)(mask_ground[14] - pixels_dominant[14])) / (float)mask_ground[14]) * 100 << "%" << endl;
		cout << "경남\t" << ((result[15][0] + result[15][1] + result[15][2]) / 3) << "\t" << result[15] << "\t" << pixels_dominant[15] << "\t" << (((float)(mask_ground[15] - pixels_dominant[15])) / (float)mask_ground[15]) * 100 << "%" << endl;
		cout << "제주\t" << ((result[16][0] + result[16][1] + result[16][2]) / 3) << "\t" << result[16] << "\t" << pixels_dominant[16] << "\t" << (((float)(mask_ground[16] - pixels_dominant[16])) / (float)mask_ground[16]) * 100 << "%" << endl;
		*/


		cout << "지역명\t" << "grayscale\t" << "[B,G,R]\t" << endl;
		cout << "서울\t" << ((result[0][0] + result[0][1] + result[0][2]) / 3) << "\t\t" << result[0] << "\t" << endl;
		cout << "경기\t" << ((result[1][0] + result[1][1] + result[1][2]) / 3) << "\t\t" << result[1] << "\t" << endl;
		cout << "인천\t" << ((result[2][0] + result[2][1] + result[2][2]) / 3) << "\t\t" << result[2] << "\t" << endl;
		cout << "대전\t" << ((result[3][0] + result[3][1] + result[3][2]) / 3) << "\t\t" << result[3] << "\t" << endl;
		cout << "세종\t" << ((result[4][0] + result[4][1] + result[4][2]) / 3) << "\t\t" << result[4] << "\t" << endl;
		cout << "대구\t" << ((result[5][0] + result[5][1] + result[5][2]) / 3) << "\t\t" << result[5] << "\t" << endl;
		cout << "울산\t" << ((result[6][0] + result[6][1] + result[6][2]) / 3) << "\t\t" << result[6] << "\t" << endl;
		cout << "부산\t" << ((result[7][0] + result[7][1] + result[7][2]) / 3) << "\t\t" << result[7] << "\t" << endl;
		cout << "광주\t" << ((result[8][0] + result[8][1] + result[8][2]) / 3) << "\t\t" << result[8] << "\t" << endl;

		cout << "강원\t" << ((result[9][0] + result[9][1] + result[9][2]) / 3) << "\t\t" << result[9] << "\t" << endl;
		cout << "충북\t" << ((result[10][0] + result[10][1] + result[10][2]) / 3) << "\t\t" << result[10] << "\t" << endl;
		cout << "충남\t" << ((result[11][0] + result[11][1] + result[11][2]) / 3) << "\t\t" << result[11] << "\t" << endl;
		cout << "전북\t" << ((result[12][0] + result[12][1] + result[12][2]) / 3) << "\t\t" << result[12] << "\t" << endl;
		cout << "전남\t" << ((result[13][0] + result[13][1] + result[13][2]) / 3) << "\t\t" << result[13] << "\t" << endl;
		cout << "경북\t" << ((result[14][0] + result[14][1] + result[14][2]) / 3) << "\t\t" << result[14] << "\t" << endl;
		cout << "경남\t" << ((result[15][0] + result[15][1] + result[15][2]) / 3) << "\t\t" << result[15] << "\t" << endl;
		cout << "제주\t" << ((result[16][0] + result[16][1] + result[16][2]) / 3) << "\t\t" << result[16] << "\t" << endl;
	}

	
	ofstream output("c:/cvimagetest/output.txt", ios::app);
	output << "서울\t" << ((result[0][0] + result[0][1] + result[0][2]) / 3) << "\t" << result[0] << "\t" << pixels_dominant[0] << "\t" << (((float)(mask_ground[0] - pixels_dominant[0])) / (float)mask_ground[0]) * 100 << "" << endl;
	output << "경기\t" << ((result[1][0] + result[1][1] + result[1][2]) / 3) << "\t" << result[1] << "\t" << pixels_dominant[1] << "\t" << (((float)(mask_ground[1] - pixels_dominant[1])) / (float)mask_ground[1]) * 100 << "" << endl;
	output << "인천\t" << ((result[2][0] + result[2][1] + result[2][2]) / 3) << "\t" << result[2] << "\t" << pixels_dominant[2] << "\t" << (((float)(mask_ground[2] - pixels_dominant[2])) / (float)mask_ground[2]) * 100 << "" << endl;
	output << "대전\t" << ((result[3][0] + result[3][1] + result[3][2]) / 3) << "\t" << result[3] << "\t" << pixels_dominant[3] << "\t" << (((float)(mask_ground[3] - pixels_dominant[3])) / (float)mask_ground[3]) * 100 << "" << endl;
	output << "세종\t" << ((result[4][0] + result[4][1] + result[4][2]) / 3) << "\t" << result[4] << "\t" << pixels_dominant[4] << "\t" << (((float)(mask_ground[4] - pixels_dominant[4])) / (float)mask_ground[4]) * 100 << "" << endl;
	output << "대구\t" << ((result[5][0] + result[5][1] + result[5][2]) / 3) << "\t" << result[5] << "\t" << pixels_dominant[5] << "\t" << (((float)(mask_ground[5] - pixels_dominant[5])) / (float)mask_ground[5]) * 100 << "" << endl;
	output << "울산\t" << ((result[6][0] + result[6][1] + result[6][2]) / 3) << "\t" << result[6] << "\t" << pixels_dominant[6] << "\t" << (((float)(mask_ground[6] - pixels_dominant[6])) / (float)mask_ground[6]) * 100 << "" << endl;
	output << "부산\t" << ((result[7][0] + result[7][1] + result[7][2]) / 3) << "\t" << result[7] << "\t" << pixels_dominant[7] << "\t" << (((float)(mask_ground[7] - pixels_dominant[7])) / (float)mask_ground[7]) * 100 << "" << endl;
	output << "광주\t" << ((result[8][0] + result[8][1] + result[8][2]) / 3) << "\t" << result[8] << "\t" << pixels_dominant[8] << "\t" << (((float)(mask_ground[8] - pixels_dominant[8])) / (float)mask_ground[8]) * 100 << "" << endl;

	output << "강원\t" << ((result[9][0] + result[9][1] + result[9][2]) / 3) << "\t" << result[9] << "\t" << pixels_dominant[9] << "\t" << (((float)(mask_ground[9] - pixels_dominant[9])) / (float)mask_ground[9]) * 100 << "" << endl;
	output << "충북\t" << ((result[10][0] + result[10][1] + result[10][2]) / 3) << "\t" << result[10] << "\t" << pixels_dominant[10] << "\t" << (((float)(mask_ground[10] - pixels_dominant[10])) / (float)mask_ground[10]) * 100 << "" << endl;
	output << "충남\t" << ((result[11][0] + result[11][1] + result[11][2]) / 3) << "\t" << result[11] << "\t" << pixels_dominant[11] << "\t" << (((float)(mask_ground[11] - pixels_dominant[11])) / (float)mask_ground[11]) * 100 << "" << endl;
	output << "전북\t" << ((result[12][0] + result[12][1] + result[12][2]) / 3) << "\t" << result[12] << "\t" << pixels_dominant[12] << "\t" << (((float)(mask_ground[12] - pixels_dominant[12])) / (float)mask_ground[12]) * 100 << "" << endl;
	output << "전남\t" << ((result[13][0] + result[13][1] + result[13][2]) / 3) << "\t" << result[13] << "\t" << pixels_dominant[13] << "\t" << (((float)(mask_ground[13] - pixels_dominant[13])) / (float)mask_ground[13]) * 100 << "" << endl;
	output << "경북\t" << ((result[14][0] + result[14][1] + result[14][2]) / 3) << "\t" << result[14] << "\t" << pixels_dominant[14] << "\t" << (((float)(mask_ground[14] - pixels_dominant[14])) / (float)mask_ground[14]) * 100 << "" << endl;
	output << "경남\t" << ((result[15][0] + result[15][1] + result[15][2]) / 3) << "\t" << result[15] << "\t" << pixels_dominant[15] << "\t" << (((float)(mask_ground[15] - pixels_dominant[15])) / (float)mask_ground[15]) * 100 << "" << endl;
	output << "제주\t" << ((result[16][0] + result[16][1] + result[16][2]) / 3) << "\t" << result[16] << "\t" << pixels_dominant[16] << "\t" << (((float)(mask_ground[16] - pixels_dominant[16])) / (float)mask_ground[16]) * 100 << "" << endl;
	output.close();
	


	return result;
}

vector<int> make_grayscale_value(vector<cv::Vec3b> input){
	/*			지역코드		*/
	/*	서울 : 0	충북 : 10	*/
	/*	경기 : 1	충남 : 11	*/
	/*	인천 : 2	전북 : 12	*/
	/*	대전 : 3	전남 : 13	*/
	/*	세종 : 4	경북 : 14	*/
	/*	대구 : 5	경남 : 15	*/
	/*	울산 : 6	제주 : 16	*/
	/*	부산 : 7	---------	*/
	/*	광주 : 8	---------	*/
	/*	강원 : 9	---------	*/
	vector<int> result;
	result.clear();

	for (int i = 0; i < input.size(); i++){
		result.push_back(((input[i][0] + input[i][1] + input[i][2]) / 3));
	}

	return result;
}

void print_same_region(vector<int> input, int index){
	/*			지역코드		*/
	/*	서울 : 0	충북 : 10	*/
	/*	경기 : 1	충남 : 11	*/
	/*	인천 : 2	전북 : 12	*/
	/*	대전 : 3	전남 : 13	*/
	/*	세종 : 4	경북 : 14	*/
	/*	대구 : 5	경남 : 15	*/
	/*	울산 : 6	제주 : 16	*/
	/*	부산 : 7	---------	*/
	/*	광주 : 8	---------	*/
	/*	강원 : 9	---------	*/
	String region_name[17] = { "서울", "경기", "인천", "대전", "세종", "대구", "울산", "부산", "광주", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주" };

	if (index >= 17){
		cout << "Index value must smaller than 17!" << endl;
		return;
	}

	cout << "선택된 지역 : " << region_name[index] << endl;
	cout << "선택된 지역과 같은 값을 가지는 지역들 : ";
	
	int selected_value = input[index];

	for (int i = 0; i < input.size(); i++){	
		if (input[i] == selected_value){
			if (i == index) continue;

			cout << region_name[i] << " ";
		}
	}
	cout << endl;

	return;
}