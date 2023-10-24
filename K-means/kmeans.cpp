#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>
#include <random>

using namespace std;
using namespace cv;

void kmeans(Mat& src, Mat& dst, int k){
    srand(time(NULL));
    vector<uchar> c(k,0);
    for(int i=0;i<k;i++){
        int randRow = rand()%src.rows;
        int randCol= rand()%src.cols;
        c[i]=src.at<uchar>(randRow,randCol);
    }
    
    bool is_c_varied = true;
    int it;
    int maxIterations = 50;
    double epsilon = 0.01f;
    vector<double> oldmean(k,0.01f);
    vector<double> newmean(k,0.01f);
    vector<vector<Point>> cluster(k);
    vector<uchar> dist(k,0);
    int minDist;

    while(is_c_varied && it++<maxIterations){
        is_c_varied = false;
        for(int i=0;i<k;i++) cluster[i].clear();
        for(int j=0;j<k;j++) oldmean[j] = newmean[j];

        for(int x=0;x<src.rows;x++)
            for(int y=0;y<src.cols;y++){
                for(int i=0;i<k;i++){
                    dist[i] = abs(c[i]-src.at<uchar>(x,y));
                }
                minDist = min_element(dist.begin(),dist.end()) -dist.begin();
                cluster[minDist].push_back(Point(x,y));
            }
        for(int i=0;i<k;i++){
            int csize = static_cast<int>(cluster[i].size());
            for(int j=0;j<csize;j++){
                int cx = cluster[i][j].x;
                int cy = cluster[i][j].y;
                newmean[i] +=src.at<uchar>(cx,cy);
            }
            newmean[i] /=csize;
            c[i] = uchar(newmean[i]);
        }
        for(int i=0;i<k;i++)
            if(!(abs(newmean[i]-oldmean[i]))<=epsilon)
                is_c_varied =true;
   
    }
    dst = src.clone();
        for(int i=0;i<k;i++){
            int csize = static_cast<int>(cluster[i].size());
            for(int j=0;j<csize;j++){
                int cx = cluster[i][j].x;
                int cy = cluster[i][j].y;
                dst.at<uchar>(cx,cy) = c[i];
            }
        }

}

int main(int argc, char** argv){
    Mat src=imread(argv[1]);
    Mat dst,gs;
    cvtColor(src,gs,COLOR_BGR2GRAY);
    kmeans(gs,dst,3);
    imshow("km",dst);
    waitKey(0);
    return 0;
}