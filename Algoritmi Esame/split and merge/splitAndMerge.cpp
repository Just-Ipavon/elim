#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

#define MIN_DEV 10.7
#define MAX_AREA 64

using namespace std;
using namespace cv;

struct Region{
	vector<Region> adj;
	bool valid;
	Scalar label;
	Rect area;
};

bool predicate(Mat src){
	Scalar dev;
	meanStdDev(src,Scalar(),dev);
	return (dev[0] < MIN_DEV || src.cols*src.rows < MAX_AREA);
}

Region split(Mat src, Rect area){
	Region R;
	R.valid = true;
	R.area = area;
	if(predicate(src)){
		Scalar mean;
		meanStdDev(src,mean,Scalar());
		R.label = mean;
	}else{
		int w = src.cols/2;
		int h = src.rows/2;
		Region r1 = split(src(Rect(0,0,w,h)),Rect(area.x,area.y,w,h));
		Region r2 = split(src(Rect(w,0,w,h)),Rect(area.x+w,area.y,w,h));
		Region r3 = split(src(Rect(0,h,w,h)),Rect(area.x,area.y+h,w,h));
		Region r4 = split(src(Rect(w,h,w,h)),Rect(area.x+w,area.y+h,w,h));
		R.adj = {r1, r2, r3, r4};
	}
	return R;
}

void mergeRegion(Mat src,Region &r1,Region &r2){
	if(r1.adj.empty() && r2.adj.empty()){
		Rect r12 = r1.area | r2.area;
		if(predicate(src(r12))){
			r1.area = r12;
			r1.label = (r1.label + r2.label) / 2;
			r2.valid = false;
		}
	}
}

void merge(Mat src,Region &r){
	if(!r.adj.empty()){
		mergeRegion(src,r.adj.at(0),r.adj.at(1));
		mergeRegion(src,r.adj.at(2),r.adj.at(3));
		mergeRegion(src,r.adj.at(0),r.adj.at(2));
		mergeRegion(src,r.adj.at(1),r.adj.at(3));
		for(auto &subregion : r.adj){
			merge(src,subregion);
		}
	}
}

void display(Mat &out, Region R){
	if(R.adj.empty() && R.valid){
		rectangle(out,R.area,R.label,FILLED);
	}
	for(auto &subregion : R.adj){
		display(out,subregion);
	}
}

int main(int argc, char* argv[]){
	Mat src = imread(argv[1], IMREAD_GRAYSCALE);
	if(src.empty()){
		cout << "Wrong PATH image selected\nPlease, try again..."<<endl;
		exit(-1);
	}
	Region r = split(src,Rect(0,0,src.cols,src.rows));
	merge(src,r);
	Mat out = src.clone();
	display(out,r);
	Mat out2 = Mat::zeros(src.size(),src.type());
	display(out2,r);
	imshow("Input",src);
	imshow("Output-1",out);
	imshow("Output-2",out2);
	waitKey(0);
	return 0;
}
