#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const int lth = 40;  // soglia bassa
const int hth = 90;  // soglia alta

// 4) Non-maxima suppression: tengo il pixel solo se è un massimo lungo la direzione del gradiente
void nonMaxSuppression(const Mat& mag, const Mat& angle, Mat& nms) {
    nms = Mat::zeros(mag.size(), CV_8U);
    for (int y = 1; y < mag.rows - 1; y++)
        for (int x = 1; x < mag.cols - 1; x++) {
            float a = fmod(angle.at<float>(y, x), 180);  // [0,360) -> [0,180)
            int dy, dx;                                  // passo verso il vicino lungo il gradiente
            if (a < 22.5 || a >= 157.5) { dy = 0; dx = 1;  }  // orizzontale
            else if (a < 67.5)          { dy = 1; dx = 1;  }  // diagonale 45
            else if (a < 112.5)         { dy = 1; dx = 0;  }  // verticale
            else                        { dy = 1; dx = -1; }  // diagonale 135
            uchar m = mag.at<uchar>(y, x);
            if (m >= mag.at<uchar>(y + dy, x + dx) && m >= mag.at<uchar>(y - dy, x - dx))
                nms.at<uchar>(y, x) = m;
        }
}

// 5) Isteresi: un pixel forte è bordo e "accende" i vicini deboli
void hysteresis(const Mat& nms, Mat& dst) {
    dst = Mat::zeros(nms.size(), CV_8U);
    for (int y = 1; y < nms.rows - 1; y++)
        for (int x = 1; x < nms.cols - 1; x++)
            if (nms.at<uchar>(y, x) > hth) {
                dst.at<uchar>(y, x) = 255;
                for (int dy = -1; dy <= 1; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        uchar v = nms.at<uchar>(y + dy, x + dx);
                        if (v > lth && v < hth)
                            dst.at<uchar>(y + dy, x + dx) = 255;
                    }
            }
}

void myCanny(const Mat& src, Mat& dst) {
    // 1) Smoothing gaussiano
    Mat gauss;
    GaussianBlur(src, gauss, Size(5, 5), 0, 0);

    // 2) Derivate con Sobel
    Mat Dx, Dy;
    Sobel(gauss, Dx, CV_32F, 1, 0, 3);
    Sobel(gauss, Dy, CV_32F, 0, 1, 3);

    // 3) Magnitudo (normalizzata a 0-255) e fase in gradi
    Mat mag, angle;
    magnitude(Dx, Dy, mag);
    normalize(mag, mag, 0, 255, NORM_MINMAX, CV_8U);
    phase(Dx, Dy, angle, true);

    // 4) NMS   5) Isteresi
    Mat nms;
    nonMaxSuppression(mag, angle, nms);
    hysteresis(nms, dst);
}

int main(int argc, char** argv) {
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    Mat dst;
    myCanny(src, dst);

    imshow("src", src);
    imshow("canny", dst);
    waitKey(0);
    return 0;
}
