#include <opencv2/opencv.hpp>
#include <stdlib.h>

using namespace std;
using namespace cv;

void hysteresisThreshold(Mat &nms, Mat &dst, int lth, int hth)
{
    for (int i = 1; i < nms.rows - 1; i++)
    {
        for (int j = 1; j < nms.cols - 1; j++)
        {
            if (nms.at<uchar>(i, j) > hth)
            {
                dst.at<uchar>(i, j) = 255;
                for (int k = -1; k <= 1; k++)
                    for (int l = -1; l <= 1; l++)
                        if (nms.at<uchar>(i + k, j + l) > lth && nms.at<uchar>(i + k, j + l) < hth)
                            dst.at<uchar>(i + k, j + l) = 255;
            }
            else
            {
                dst.at<uchar>(i, j) = 0;
            }
        }
    }
}

void nonMaximaSuppression(Mat &mag, Mat &orientations, Mat &nms)
{
    float angle;
    for (int i = 1; i < mag.rows - 1; i++)
    {
        for (int j = 1; j < mag.cols - 1; j++)
        {
            angle = orientations.at<float>(i, j);
            angle = angle > 180 ? angle - 360 : angle;
            if ((-22.5 < angle && angle <= 22.5) || (157.5 < angle && angle <= -157.5))
            { // horizontal
                if (mag.at<uchar>(i, j) > mag.at<uchar>(i, j - 1) && mag.at<uchar>(i, j) > mag.at<uchar>(i, j + 1))
                {
                    nms.at<uchar>(i, j) = mag.at<uchar>(i, j);
                }
            }
            else if ((-67.5 < angle && angle <= -22.5) || (112.5 < angle && angle <= 157.5))
            { //+45
                if (mag.at<uchar>(i, j) > mag.at<uchar>(i - 1, j + 1) && mag.at<uchar>(i, j) > mag.at<uchar>(i + 1, j - 1))
                {
                    nms.at<uchar>(i, j) = mag.at<uchar>(i, j);
                }
            }
            else if ((-112.5 < angle && angle <= -67.5) || (67.5 < angle && angle <= 112.5))
            { // vertical
                if (mag.at<uchar>(i, j) > mag.at<uchar>(i - 1, j) && mag.at<uchar>(i, j) > mag.at<uchar>(i + 1, j))
                {
                    nms.at<uchar>(i, j) = mag.at<uchar>(i, j);
                }
            }
            else if ((-157.5 < angle && angle <= -112.5) || (67.5 < angle && angle <= 22.5))
            { //-45
                if (mag.at<uchar>(i, j) > mag.at<uchar>(i - 1, j - 1) && mag.at<uchar>(i, j) > mag.at<uchar>(i + 1, j + 1))
                {
                    nms.at<uchar>(i, j) = mag.at<uchar>(i, j);
                }
            }
        }
    }
}

void canny(Mat &src, Mat &dst, int lth, int hth, int ksize)
{

    // 1
    Mat gauss;
    GaussianBlur(src, gauss, Size(3, 3), 0, 0);

    // 2
    Mat Dx, Dy, Dx2, Dy2, mag, orientations;
    Sobel(gauss, Dx, CV_32FC1, 1, 0, ksize);
    Sobel(gauss, Dy, CV_32FC1, 0, 1, ksize);
    pow(Dx, 2, Dx2);
    pow(Dy, 2, Dy2);
    sqrt(Dx2 + Dy2, mag);
    normalize(mag, mag, 0, 255, NORM_MINMAX, CV_8UC1);
    phase(Dx, Dy, orientations, true);

    // 3
    Mat nms = Mat::zeros(src.rows, src.cols, CV_8UC1);
    nonMaximaSuppression(mag, orientations, nms);

    // 4
    dst = Mat::zeros(src.rows, src.cols, CV_8UC1);
    hysteresisThreshold(nms, dst, lth, hth);
}

int main(int argc, char **argv)
{

    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if (src.empty())
        return -1;

    Mat dst;
    int lth = 20, hth = 80, ksize = 3;
    canny(src, dst, lth, hth, ksize);

    imshow("src", src);
    imshow("dst", dst);
    waitKey(0);

    return 0;
}
