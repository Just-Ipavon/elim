#include <stdlib.h>

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// imposto i valori del threshold e della dimensione del kernel

const int th1 = 20;
const int th2 = 50;
const int ksize = 3;

void Canny(const Mat src, Mat &dst) {                               // passo la matrice sorgente e la matrice destinazione
    Mat gauss, dx, dy, magnitude, phase;                            // creo le matrici per la gaussiana, le derivate, la magnitudo e la fase
    GaussianBlur(src, gauss, Size(5, 5), 0, 0);                     // applico la gaussiana
    Sobel(gauss, dx, CV_32FC1, 1, 0, ksize);                        // applico la derivata in x CV_32FC1 è il tipo di dato 32 bit 1 canale float
    Sobel(gauss, dy, CV_32FC1, 0, 1, ksize);                        // applico la derivata in y
    cv::magnitude(dx, dy, magnitude);                               // calcolo la magnitudo
    normalize(magnitude, magnitude, 0, 255, NORM_MINMAX, CV_8UC1);  // normalizzo la magnitudo 8 bit unsigned 1 canale
    cv::phase(dx, dy, phase, true);                                 // calcolo la fase
    uchar q, r;                                                     // creo due variabili di tipo unsigned char
    for (int y = 1; y < magnitude.rows - 1; y++) {
        for (int x = 1; x < magnitude.cols - 1; x++) {
            float angle = phase.at<float>(y, x) > 180 ? phase.at<float>(y, x) - 360 : phase.at<float>(y, x);  // calcolo l'angolo
            uchar mag = magnitude.at<uchar>(y, x);                                                            // calcolo la magnitudo
            if ((angle <= -157.5 || angle > 157.5) || (angle > -22.5 && angle <= 22.5)) {                     // controllo se il gradiente è orizzontale
                q = magnitude.at<uchar>(y, x - 1);
                r = magnitude.at<uchar>(y, x + 1);
            } else if ((angle > -157.5 && angle <= -112.5) || (angle > 22.5 && angle <= 67.5)) {  // controllo se il gradiente è diagonale 45
                q = magnitude.at<uchar>(y + 1, x - 1);
                r = magnitude.at<uchar>(y - 1, x + 1);
            } else if ((angle > -67.5 && angle <= 67.5)) {  // controllo se il gradiente è verticale
                q = magnitude.at<uchar>(y + 1, x);
                r = magnitude.at<uchar>(y - 1, x);
            } else if ((angle > 67.5 && angle <= 112.5) || (angle > -112.5 && angle <= -67.5)) {  // controllo se il gradiente è diagonale -45
                q = magnitude.at<uchar>(y - 1, x - 1);
                r = magnitude.at<uchar>(y + 1, x + 1);
            }
            if (mag < r || mag < q)
                magnitude.at<uchar>(y, x) = 0;  // se la magnitudo è minore di q o r imposto il pixel a 0
        }
    }
    for (int i = 1; i < magnitude.rows - 1; i++) {  // applico il thresholding
            for (int j = 1; j < magnitude.cols - 1; j++) {
                uchar px = magnitude.at<uchar>(i, j);
                if (px >= th2)
                    px = 255;  // se il pixel è maggiore del threshold alto lo imposto a 255
                else if (px < th1)
                    px = 0;  // se il pixel è minore del threshold basso lo imposto a 0
                else {
                    bool sn = false;  // se il pixel è compreso tra i due threshold controllo se è un pixel di bordo
                    for (int x = -1; x <= 1 && !sn; x++)
                        for (int y = -1; y <= 1 && !sn; y++)
                            if (magnitude.at<uchar>(i + x, j + y) > th2)
                                sn = true;
                    if (sn)
                        px = 255;
                    else
                        px = 0;
                }
                magnitude.at<uchar>(i, j) = px;  // imposto il pixel
            }
        }
        magnitude.copyTo(dst);
}

int main(int argc, char **argv) {  // leggo l'immagine in input e la passo alla funzione Canny
    String imageName("lena.jpg");
     if( argc > 1){
        imageName = argv[1];
    }
    Mat src = imread(imageName, IMREAD_GRAYSCALE);
    Mat dst;
    Canny(src, dst);
    imshow("src", src);
    imshow("dst", dst);
    waitKey(0);
    return 0;
}
