#include <stdlib.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<double> normalizedHistogram(Mat& src){
    vector<double> his(256,0.0f);
    for(int i=0; i<src.rows; i++)
        for(int j=0; j<src.cols; j++)
            his[src.at<uchar>(i,j)]++;
    for(int i=0; i<256; i++)
        his[i] /= src.rows*src.cols;

    return his;
}

vector<int> otsu2k(Mat& src){

    vector<double> his = normalizedHistogram(src);

    double gMean = 0.0f;
    for(int i=0; i<256; i++)
        gMean += i*his[i];

    vector<double> currProb(3,0.0f);
    vector<double> currCumMean(3,0.0f);
    double currIntVar = 0.0f;
    double maxVar = 0.0f;
    vector<int> kstar(2,0);
    for(int i=0; i<256-2; i++){
        currProb[0] += his[i];
        currCumMean[0] += i*his[i];
        for(int j=i+1; j<256-1; j++){
            currProb[1] += his[j];
            currCumMean[1] += j*his[j];
            for(int k=j+1; k<256; k++){
                currProb[2] += his[k];
                currCumMean[2] += k*his[k];
                currIntVar = 0.0f;
                for(int w=0; w<3; w++)
                    currIntVar += currProb[w]*pow(currCumMean[w]/currProb[w]-gMean,2);
                if(currIntVar > maxVar){
                    maxVar = currIntVar;
                    kstar[0] = i;
                    kstar[1] = j;
                }
            }
            currProb[2] = currCumMean[2] = 0.0f;
        }
        currProb[1] = currCumMean[1] = 0.0f;
    }

    return kstar;
}

void multipleThresholds(Mat& src, Mat& dst, int th1, int th2){

    cout<<th1<<"\t"<<th2<<endl;
    dst = Mat::zeros(src.rows, src.cols, CV_8U);
    for(int i=0; i<src.rows; i++)
        for(int j=0; j<src.cols; j++)
            if(src.at<uchar>(i,j) >= th2)
                dst.at<uchar>(i,j) = 255;
            else if(src.at<uchar>(i,j) >= th1)
                dst.at<uchar>(i,j) = 127;
}

int main(int argc, char** argv){

	Mat src = imread(argv[1],IMREAD_GRAYSCALE);
	if(src.empty()) return -1;

    Mat dst, gsrc;
    GaussianBlur(src,gsrc,Size(3,3),0,0);

    vector<int> ths = otsu2k(gsrc);
    multipleThresholds(gsrc,dst,ths[0],ths[1]);

    imshow("src",src);
    imshow("otsu2k",dst);
    waitKey(0);
	return 0;
}
