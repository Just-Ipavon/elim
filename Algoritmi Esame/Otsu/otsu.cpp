#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// p[i] = probabilità del livello di grigio i
vector<double> normalizedHistogram(const Mat& src) {
    vector<double> p(256, 0.0);
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
            p[src.at<uchar>(y, x)]++;
    for (int i = 0; i < 256; i++)
        p[i] /= src.rows * src.cols;
    return p;
}

int otsu(const Mat& src) {
    // 1) Istogramma normalizzato
    vector<double> p = normalizedHistogram(src);

    // 2) Media globale mG
    double mG = 0.0;
    for (int i = 0; i < 256; i++)
        mG += i * p[i];

    // 3) Per ogni k: P1 (prob. cumulata), m (media cumulata), varianza inter-classe
    //    sigmaB2 = (mG*P1 - m)^2 / (P1*(1-P1))  ->  tengo il k che la massimizza
    double P1 = 0.0, m = 0.0, maxVar = 0.0;
    int kstar = 0;
    for (int k = 0; k < 256; k++) {
        P1 += p[k];
        m += k * p[k];
        if (P1 <= 0 || P1 >= 1) continue;  // evito la divisione per zero
        double sigmaB2 = pow(mG * P1 - m, 2) / (P1 * (1 - P1));
        if (sigmaB2 > maxVar) {
            maxVar = sigmaB2;
            kstar = k;
        }
    }
    return kstar;
}

int main(int argc, char** argv) {
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    Mat gauss, dst;
    GaussianBlur(src, gauss, Size(3, 3), 0, 0);
    int th = otsu(gauss);
    threshold(gauss, dst, th, 255, THRESH_BINARY);

    imshow("src", src);
    imshow("otsu", dst);
    waitKey(0);
    return 0;
}
