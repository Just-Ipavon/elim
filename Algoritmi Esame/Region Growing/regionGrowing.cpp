#include <opencv2/opencv.hpp>
#include <stack>

using namespace std;
using namespace cv;

const int th = 200;                      // soglia sulla distanza di colore (al quadrato)
const int maxRegions = 100;              // numero massimo di regioni
const double minRegionAreaFactor = 0.01; // regioni sotto l'1% dell'immagine vengono ignorate

// distanza di colore al quadrato: (B-B')^2 + (G-G')^2 + (R-R')^2
int colorDist2(Vec3b a, Vec3b b) {
    int d0 = a[0] - b[0], d1 = a[1] - b[1], d2 = a[2] - b[2];
    return d0 * d0 + d1 * d1 + d2 * d2;
}

// Accresce una regione dal seed: aggiungo i vicini (8-connessi) simili al pixel corrente
void grow(const Mat& src, const Mat& dst, Mat& mask, Point seed) {
    stack<Point> front;
    front.push(seed);
    mask.at<uchar>(seed) = 1;
    while (!front.empty()) {
        Point p = front.top();
        front.pop();
        for (int dy = -1; dy <= 1; dy++)
            for (int dx = -1; dx <= 1; dx++) {
                Point q(p.x + dx, p.y + dy);
                if (q.x < 0 || q.y < 0 || q.x >= src.cols || q.y >= src.rows) continue;  // fuori immagine
                if (dst.at<uchar>(q) || mask.at<uchar>(q)) continue;                     // già visitato
                if (colorDist2(src.at<Vec3b>(p), src.at<Vec3b>(q)) < th) {
                    mask.at<uchar>(q) = 1;  // marco subito, così non lo inserisco due volte
                    front.push(q);
                }
            }
    }
}

// dst: 0 = non visitato, 1..maxRegions = etichetta della regione, 255 = ignorato
void regionGrowing(const Mat& src, Mat& dst) {
    dst = Mat::zeros(src.size(), CV_8U);
    Mat mask = Mat::zeros(src.size(), CV_8U);
    int minRegionArea = int(minRegionAreaFactor * src.rows * src.cols);
    int label = 1;

    // Ogni pixel non ancora visitato diventa un seed
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
            if (dst.at<uchar>(y, x) == 0) {
                grow(src, dst, mask, Point(x, y));
                if (countNonZero(mask) > minRegionArea) {
                    dst += mask * label;  // regione valida
                    imshow("mask", mask * 255);
                    waitKey(0);
                    if (++label > maxRegions) {
                        cout << "Run out of region labels..." << endl;
                        return;
                    }
                } else
                    dst += mask * 255;  // regione troppo piccola: ignorata
                mask.setTo(0);
            }
}

int main(int argc, char** argv) {
    Mat src = imread(argv[1], IMREAD_COLOR);
    if (src.empty()) return -1;
    if (src.rows > 500 || src.cols > 500)
        resize(src, src, Size(0, 0), 0.5, 0.5);  // dimezzo le immagini grandi (come il prof)

    Mat dst;
    regionGrowing(src, dst);

    imshow("src", src);
    imshow("regions", dst);
    waitKey(0);
    return 0;
}
