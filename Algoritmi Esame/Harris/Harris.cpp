#include <opencv2/opencv.hpp>
#include <stdlib.h>

using namespace std;
using namespace cv;

void circleCorners(Mat& src, Mat& dst, int th){
    for(int i=0; i<src.rows; i++){
        for(int j=0; j<src.cols; j++){
            if((int)src.at<float>(i,j)>th)
                circle(dst,Point(j,i),5,Scalar(0),1);
        }
    }
}

void harris(Mat& src, Mat &dst, int ksize, int th, float k){

	//1
	Mat Dx, Dy;
	Sobel(src,Dx,CV_32FC1,1,0,ksize);
	Sobel(src,Dy,CV_32FC1,0,1,ksize);

	//2
	Mat Dx2, Dy2, DxDy;
	pow(Dx,2,Dx2);
	pow(Dy,2,Dy2);
	multiply(Dx,Dy,DxDy);

	//3-4
	Mat C00, C01, C10, C11;
	GaussianBlur(Dx2,C00,Size(7,7),2,0);
	GaussianBlur(Dy2,C11,Size(7,7),0,2);
	GaussianBlur(DxDy,C01,Size(7,7),2,2);
	C10 = C01;

	//5
	Mat det, trace, trace2, R, PPD, PSD;

	multiply(C00,C11,PPD);
	multiply(C01,C10,PSD);
	det = PPD-PSD;

	trace = C00+C11;
	pow(trace,2,trace2);

	R = det-k*trace2;

	//6
	normalize(R,R,0,255,NORM_MINMAX,CV_32FC1);
	convertScaleAbs(R,dst);

	//7
	circleCorners(R,dst,th);

}

int main(int argc, char** argv){

	Mat src = imread(argv[1],IMREAD_GRAYSCALE);
	if(src.empty()) return -1;

	Mat dst;
	int ksize = 3, th = 150;
	float k = 0.117;
	harris(src,dst,ksize,th,k);

	imshow("src",src);
	imshow("dst",dst);
	waitKey(0);

	return 0;
}
