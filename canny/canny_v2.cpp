#include <opencv2/opencv.hpp>

using namespace cv;

void canny(Mat &src, Mat &dst, int lth, int hth, int ksize)
{
    // 1. Gaussian blur
    Mat gauss;
    GaussianBlur(src, gauss, Size(3, 3), 0, 0);

    // 2. Sobel edge detection
    Mat Dx, Dy, mag, orientations;
    Sobel(gauss, Dx, CV_32FC1, 1, 0, ksize);
    Sobel(gauss, Dy, CV_32FC1, 0, 1, ksize);
    magnitude(Dx, Dy, mag);
    phase(Dx, Dy, orientations, true);

    // 3. Non-maximum suppression
    Mat nms;
    copyMakeBorder(mag, nms, 1, 1, 1, 1, BORDER_REPLICATE);
    for (int i = 1; i < nms.rows - 1; i++)
    {
        for (int j = 1; j < nms.cols - 1; j++)
        {
            float angle = orientations.at<float>(i - 1, j - 1);
            if (angle < 0) angle += 360;
            int q = 255, r = 255;
            if ((0 <= angle && angle < 22.5) || (157.5 <= angle && angle <= 180))
            {
                q = nms.at<float>(i, j + 1);
                r = nms.at<float>(i, j - 1);
            }
            else if (22.5 <= angle && angle < 67.5)
            {
                q = nms.at<float>(i + 1, j - 1);
                r = nms.at<float>(i - 1, j + 1);
            }
            else if (67.5 <= angle && angle < 112.5)
            {
                q = nms.at<float>(i + 1, j);
                r = nms.at<float>(i - 1, j);
            }
            else if (112.5 <= angle && angle < 157.5)
            {
                q = nms.at<float>(i - 1, j - 1);
                r = nms.at<float>(i + 1, j + 1);
            }
            if (mag.at<float>(i - 1, j - 1) >= q && mag.at<float>(i - 1, j - 1) >= r)
            {
                nms.at<float>(i, j) = mag.at<float>(i - 1, j - 1);
            }
            else
            {
                nms.at<float>(i, j) = 0;
            }
        }
    }

    // 4. Hysteresis thresholding
    dst = Mat::zeros(src.size(), CV_8UC1);
    for (int i = 0; i < dst.rows; i++)
    {
        for (int j = 0; j < dst.cols; j++)
        {
            if (nms.at<float>(i + 1, j + 1) >= hth)
            {
                dst.at<uchar>(i, j) = 255;
            }
            else if (nms.at<float>(i + 1, j + 1) >= lth && nms.at<float>(i + 1, j + 1) < hth)
            {
                for (int k = -1; k <= 1; k++)
                {
                    for (int l = -1; l <= 1; l++)
                    {
                        if (nms.at<float>(i + 1 + k, j + 1 + l) >= hth)
                        {
                            dst.at<uchar>(i, j) = 255;
                            break;
                        }
                    }
                    if (dst.at<uchar>(i, j) == 255) break;
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    Mat dst;
    int lth = 20, hth = 80, ksize = 3;
    canny(src, dst, lth, hth, ksize);

    imshow("src", src);
    imshow("dst", dst);
    waitKey(0);

    return 0;
}