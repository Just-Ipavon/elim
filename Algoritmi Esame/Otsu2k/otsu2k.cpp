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

// Restituisce le due soglie {k1, k2}: classi [0,k1], (k1,k2], (k2,255]
vector<int> otsu2k(const Mat& src) {
    // 1) Istogramma normalizzato
    vector<double> p = normalizedHistogram(src);

    // 2) Media globale mG
    double mG = 0.0;
    for (int i = 0; i < 256; i++)
        mG += i * p[i];

    // 3) Per ogni coppia (k1,k2): P e m cumulati delle 3 classi
    //    sigmaB2 = somma_w P[w] * (m[w]/P[w] - mG)^2  ->  tengo la coppia che la massimizza
    double maxVar = 0.0;
    vector<int> kstar(2, 0);
    double P1 = 0.0, m1 = 0.0;
    for (int k1 = 0; k1 < 256 - 2; k1++) {
        P1 += p[k1];
        m1 += k1 * p[k1];
        double P2 = 0.0, m2 = 0.0;
        for (int k2 = k1 + 1; k2 < 256 - 1; k2++) {
            P2 += p[k2];
            m2 += k2 * p[k2];
            double P3 = 1 - P1 - P2;  // la terza classe è "tutto il resto"
            double m3 = mG - m1 - m2;
            double P[3] = {P1, P2, P3};
            double m[3] = {m1, m2, m3};
            double sigmaB2 = 0.0;
            for (int w = 0; w < 3; w++)
                if (P[w] > 0)
                    sigmaB2 += P[w] * pow(m[w] / P[w] - mG, 2);
            if (sigmaB2 > maxVar) {
                maxVar = sigmaB2;
                kstar[0] = k1;
                kstar[1] = k2;
            }
        }
    }
    return kstar;
}

// 4) Tre livelli di uscita: 0, 127, 255
void multipleThreshold(const Mat& src, Mat& dst, int k1, int k2) {
    dst = Mat::zeros(src.size(), CV_8U);
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++) {
            uchar v = src.at<uchar>(y, x);
            if (v > k2)
                dst.at<uchar>(y, x) = 255;
            else if (v > k1)
                dst.at<uchar>(y, x) = 127;
        }
}

int main(int argc, char** argv) {
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    Mat gauss, dst;
    GaussianBlur(src, gauss, Size(3, 3), 0, 0);
    vector<int> k = otsu2k(gauss);
    multipleThreshold(gauss, dst, k[0], k[1]);

    imshow("src", src);
    imshow("otsu2k", dst);
    waitKey(0);
    return 0;
}
