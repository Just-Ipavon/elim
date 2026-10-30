#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const float k = 0.04f;  // costante di Harris (0.04 - 0.06)
const int th = 100;     // soglia su R normalizzato 0-255

void harris(const Mat& src, Mat& dst) {
    // 1) Derivate con Sobel
    Mat Dx, Dy;
    Sobel(src, Dx, CV_32F, 1, 0, 3);
    Sobel(src, Dy, CV_32F, 0, 1, 3);

    // 2) Prodotti delle derivate
    Mat Dx2, Dy2, Dxy;
    pow(Dx, 2, Dx2);
    pow(Dy, 2, Dy2);
    multiply(Dx, Dy, Dxy);

    // 3) Smoothing gaussiano -> elementi della matrice C = [Sx2 Sxy; Sxy Sy2]
    Mat Sx2, Sy2, Sxy;
    GaussianBlur(Dx2, Sx2, Size(7, 7), 2, 2);
    GaussianBlur(Dy2, Sy2, Size(7, 7), 2, 2);
    GaussianBlur(Dxy, Sxy, Size(7, 7), 2, 2);

    // 4) R = det(C) - k * trace(C)^2
    Mat SxSy, Sxy2, trace2;
    multiply(Sx2, Sy2, SxSy);
    multiply(Sxy, Sxy, Sxy2);
    pow(Sx2 + Sy2, 2, trace2);
    Mat R = (SxSy - Sxy2) - k * trace2;

    // 5) Normalizzo R e cerchio i punti sopra soglia
    normalize(R, R, 0, 255, NORM_MINMAX, CV_32F);
    dst = src.clone();
    for (int y = 0; y < R.rows; y++)
        for (int x = 0; x < R.cols; x++)
            if (R.at<float>(y, x) > th)
                circle(dst, Point(x, y), 5, Scalar(0), 2);
}

int main(int argc, char** argv) {
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    Mat dst;
    harris(src, dst);

    imshow("src", src);
    imshow("harris", dst);
    waitKey(0);
    return 0;
}
