#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

cv::Mat addPadding(cv::Mat image, int top, int bottom, int left, int right, int borderType, cv::Scalar value)
{
    cv::Mat padded;
    cv::copyMakeBorder(image, padded, top, bottom, left, right, borderType, value);
    return padded;
}

int main(int argc, char** argv){

//! [load]
    String imageName( "image.png" ); // by default
    if( argc > 1){
        imageName = argv[1];
    }
    //! [load]

    //! [mat]
    Mat image;
    //! [mat]

    //! [imread]
    image = imread( samples::findFile( imageName ), IMREAD_COLOR ); // Read the file
    //! [imread]

    if( image.empty() )                      // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }


    // Aggiungi il padding all'immagine
    cv::Scalar mean = cv::mean(image);
    cv::Mat padded = addPadding(image, 10, 10, 10, 10, cv::BORDER_CONSTANT, mean);

    // Mostra l'immagine originale e quella con il padding
    cv::imshow("Original", image);
    cv::imshow("Padded", padded);
    cv::waitKey(0);

    return 0;
}
