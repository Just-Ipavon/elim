#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const int cth1 = 100;  // soglie di Canny
const int cth2 = 112;
const int rMin = 22;   // raggi cercati: [rMin, rMax]
const int rMax = 25;
const int hth = 123;   // voti minimi per accettare un cerchio

void houghCircles(const Mat& src, Mat& dst) {
    // 1) Smoothing + bordi con Canny
    Mat gauss, edges;
    GaussianBlur(src, gauss, Size(5, 5), 0, 0);
    Canny(gauss, edges, cth1, cth2);

    // 2) Accumulatore 3D: (b, a, r) = (riga centro, colonna centro, raggio)
    int sizes[] = {src.rows, src.cols, rMax - rMin + 1};
    Mat votes(3, sizes, CV_32S, Scalar(0));

    // 3) Votazione: ogni punto di bordo vota per i centri a distanza r da sé
    for (int y = 0; y < edges.rows; y++)
        for (int x = 0; x < edges.cols; x++)
            if (edges.at<uchar>(y, x) == 255)
                for (int r = rMin; r <= rMax; r++)
                    for (int t = 0; t < 360; t++) {
                        double theta = t * CV_PI / 180;
                        int a = cvRound(x - r * cos(theta));
                        int b = cvRound(y - r * sin(theta));
                        if (a >= 0 && a < src.cols && b >= 0 && b < src.rows)
                            votes.at<int>(b, a, r - rMin)++;
                    }

    // 4) Disegno centro e circonferenza dei cerchi con abbastanza voti
    dst = src.clone();
    for (int r = rMin; r <= rMax; r++)
        for (int b = 0; b < src.rows; b++)
            for (int a = 0; a < src.cols; a++)
                if (votes.at<int>(b, a, r - rMin) >= hth) {
                    circle(dst, Point(a, b), 3, Scalar(0), 2);
                    circle(dst, Point(a, b), r, Scalar(0), 2);
                }
}

int main(int argc, char** argv) {
    Mat src = imread(argv[1], IMREAD_GRAYSCALE);
    if (src.empty()) return -1;

    Mat dst;
    houghCircles(src, dst);

    imshow("src", src);
    imshow("hough circles", dst);
    waitKey(0);
    return 0;
}
