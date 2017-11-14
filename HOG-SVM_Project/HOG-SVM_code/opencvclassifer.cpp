//#include <iostream>
//#include <string>
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/objdetect/objdetect.hpp>
//#include <opencv2/ml/ml.hpp>
//#include <opencv2/highgui/highgui_c.h>
//
//using namespace std;
//using namespace cv;
//
//int main()
//{
//      Mat src = imread("/Users/chenxuanqi/Downloads/HOG-SVM/single-Demo/test0.jpg");
////    Mat src = imread("/Users/chenxuanqi/Downloads/finaltest/single-Demo/test1.png");
////    Mat src = imread("/Users/chenxuanqi/Downloads/finaltest/single-Demo/test2.png");
////    Mat src = imread("/Users/chenxuanqi/Downloads/finaltest/single-Demo/test3.png");
//    HOGDescriptor hog;
//    hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
//    vector<Rect> found, found_filtered;
//    hog.detectMultiScale(src, found, 0, Size(8,8), Size(32,32), 1.05, 2);
//    
//    cout<<"The number of founded rectangle："<<found.size()<<endl;
//    for(int i=0; i < found.size(); i++)
//    {
//        Rect r = found[i];
//        int j=0;
//        for(; j < found.size(); j++)
//            if(j != i && (r & found[j]) == r)
//                break;
//        if( j == found.size())
//            found_filtered.push_back(r);
//    }
//    cout<<"The number of bounding box："<<found_filtered.size()<<endl;
//    
//        for(int i=0; i<found_filtered.size(); i++)
//    {
//        Rect r = found_filtered[i];
//        r.x += cvRound(r.width*0.1);
//        r.width = cvRound(r.width*0.8);
//        r.y += cvRound(r.height*0.07);
//        r.height = cvRound(r.height*0.8);
//        rectangle(src, r.tl(), r.br(), Scalar(0,255,0), 3);
//    }
//
//    imwrite("/Users/chenxuanqi/Downloads/HOG-SVM/single-Demo/test0.jpg",src);
////    imwrite("/Users/chenxuanqi/Downloads/finaltest/single-Demo/Result1.jpg",src);
////    imwrite("/Users/chenxuanqi/Downloads/finaltest/single-Demo/Result2.jpg",src);
////    imwrite("/Users/chenxuanqi/Downloads/finaltest/single-Demo/Result3.jpg",src);
//    namedWindow("src",0);
//    imshow("src",src);
//    waitKey();
//    
//    
//    
//    system("pause");
//}
