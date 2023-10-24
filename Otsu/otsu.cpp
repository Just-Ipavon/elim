#include <stdlib.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<double> normalizedHistogram(Mat& src){

	vector<double> his(256,0);
	for(int i=0; i<src.rows; i++)
		for(int j=0; j<src.cols; j++)
			his[src.at<uchar>(i,j)]++;

	for(int i=0; i<256; i++)
		his[i] /= src.rows*src.cols;

	return his;
}

int otsu(Mat& src){

    vector<double> his = normalizedHistogram(src);

	double gMean = 0.0f;
	for(int i=0; i<256; i++)
		gMean += i*his[i];

	double currProb1 = 0.0f;
	double currCumMean = 0.0f;
	double currIntVar = 0.0f;
	double maxVar = 0.0f;
	int kstar = 0;
	for(int i=0; i<256; i++){
		currProb1 += his[i];
		currCumMean += i*his[i];
		currIntVar = pow(gMean*currProb1-currCumMean,2)/(currProb1*(1-currProb1));
		if(currIntVar > maxVar){
			maxVar = currIntVar;
			kstar = i;
		}
	}

	return kstar;
}

int main(int argc, char** argv){

	Mat src = imread(argv[1],IMREAD_GRAYSCALE);
	if(src.empty()) return -1;

	Mat dst, gsrc;
	GaussianBlur(src,gsrc,Size(3,3),0,0);
	int th = otsu(src);
	threshold(src,dst,th,255,THRESH_BINARY);

	imshow("src",src);
	imshow("otsu", dst);
    waitKey(0);
	return 0;
}