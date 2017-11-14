#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

#include<opencv2/contrib/contrib.hpp>



using namespace std;
using namespace cv;

#define PosSamNO 2000    //The number of positive sample
#define NegSamNO 10000    //The number of negative sample

//Decide to train or not to train
//true means retrain
//false means read the SVM model in xml file
#define TRAIN true
#define CENTRAL_CROP true


//if HardExampleNO larger than 0, deal with hard examples after negtive examples
//Thus, if there is no need to use HardExample, thue number of HardExampleNO should set back to0
#define HardExampleNO 0
//#define HardExampleNO 4435



class MySVM : public CvSVM
{
public:
    double * get_alpha_vector()
    {
        return this->decision_func->alpha;
    }
    
    float get_rho()
    {
        return this->decision_func->rho;
    }
};



int main()
{
    TickMeter tm;
    tm.start();

    //The detect window size(64,128);the block size(16x16), the stride of block(8,8),the size of cell(8,8),the number of bin=9
    HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);
    //The dimension of HOG descriptor dimension, which is decided by the size of
    //image, detect window,block, the number of bins in each cell
    int DescriptorDim;
    MySVM svm;//SVM classifier
    
    
    //if train is true, retrain the classifier
    if(TRAIN)
    {
        //The Full Path of image
        string ImgName;
        //The list of file names of the positive image
        ifstream finPos("/Users/chenxuanqi/Downloads/HOG-SVM/INRIAPerson/pos.txt");
        //The list of file names of the negative image
        ifstream finNeg("/Users/chenxuanqi/Downloads/HOG-SVM/INRIAPerson/neg.txt");
        
        
        //The matrix represents the features of all samples
        //row means the number of samples
        //columns means the dimension of the HOG featues
        Mat sampleFeatureMat;
        //The array represents the label of each sample
        //1 means person while -1 means noperson
        Mat sampleLabelMat;
        
        
        //read the positive samples and generate HOG features
        for(int num=0; num<PosSamNO && getline(finPos,ImgName); num++)
        {
            
            //add the path of positive images
            ImgName = "/Users/chenxuanqi/Downloads/HOG-SVM/INRIAPerson/" + ImgName;
            cout<<"Deal with POS："<<ImgName<<endl;
            
            Mat src = imread(ImgName);//read image
            //resize the 96x160 INRIA image to 64x128 by cutting 16 pixel from the 4 corner of the image
            if(CENTRAL_CROP)
                src = src(Rect(16,16,64,128));
            //resize(src,src,Size(64,128));
            
            //HOG decriptor vector
            vector<float> descriptors;
            //Calculate the HOG descriptor the stride of the testing window is 8x8
            hog.compute(src,descriptors,Size(8,8));
            //cout<<"HOG dimension"<<descriptors.size()<<endl;
            
            
            
            // When deal with the first sample initialize the the sampleFeatureMat
            //because we can initialize the dimension of the featurs only when we no the dimention of the features
            if( 0 == num )
            {
                //HOG dimension
                DescriptorDim = descriptors.size();
                //initialize the sampleFeatureMat
                sampleFeatureMat = Mat::zeros(PosSamNO+NegSamNO+HardExampleNO, DescriptorDim, CV_32FC1);
                //initialize the sampleLabelMat
                sampleLabelMat = Mat::zeros(PosSamNO+NegSamNO+HardExampleNO, 1, CV_32FC1);
            }
            
            //copy the HOG descriptors calculated to sampleFeatureNat
            for(int i=0; i<DescriptorDim; i++){
                
                //(num,i) corresponding the ith feature element in the numth sample
                sampleFeatureMat.at<float>(num,i) = descriptors[i];
            }
            //The label equals 1 means have person
            sampleLabelMat.at<float>(num,0) = 1;
        }
        
        //read the negative samples and generate HOG features
        for(int num=0; num<NegSamNO && getline(finNeg,ImgName); num++)
        {
            cout<<"Deal with NEG："<<ImgName<<endl;
            ImgName = "/Users/chenxuanqi/Downloads/HOG-SVM/INRIAPerson/"  + ImgName;
            Mat src = imread(ImgName);
            //resize(src,img,Size(64,128));
            
            vector<float> descriptors;
            hog.compute(src,descriptors,Size(8,8));
            
            
            
            for(int i=0; i<DescriptorDim; i++)
                //the row is corresponding to the num+PosSamNO
                sampleFeatureMat.at<float>(num+PosSamNO,i) = descriptors[i];
            //the negative label is -1 corresponding to noperson
            sampleLabelMat.at<float>(num+PosSamNO,0) = -1;
        }
        
        //HardExample with negative sample
        if(HardExampleNO > 0)
        {
            ifstream finHardExample("/Users/chenxuanqi/Desktop/ECE657-CODE/INRIAPerson/annotations.txt");
            
            for(int num=0; num<HardExampleNO && getline(finHardExample,ImgName); num++)
            {
                cout<<"Deal："<<ImgName<<endl;
                ImgName = "/Users/chenxuanqi/Desktop/ECE657-CODE/INRIAPerson/Train/annotations"  + ImgName;
                Mat src = imread(ImgName);
                //resize(src,img,Size(64,128));
                
                vector<float> descriptors;
                hog.compute(src,descriptors,Size(8,8));
                
                for(int i=0; i<DescriptorDim; i++)
                    sampleFeatureMat.at<float>(num+PosSamNO+NegSamNO,i) = descriptors[i];
                sampleLabelMat.at<float>(num+PosSamNO+NegSamNO,0) = -1;
            }
        }
        
        //The requirement of stopping iteration is the number of iteration over 1000 times or error is less than FLT_EPSILON.
        CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
        //The parameter of SVM is using default parameters
        //The type of SVM classifier: C_SVC;the linear kernal funciton;Relaxation Factors C=0.01
        CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
        cout<<"Start training SVM classifier"<<endl;
        svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);//training
        cout<<"Train Completed"<<endl;
        svm.save("SVM_HOG.xml");//save the classifier as XML files
        
    }
    else
    {
        //is TRAIN is false, read the trained classifier from XML files
        svm.load("SVM_HOG_2400PosINRIA_12000Neg_HardExample.xml");
    }
    
    
    //After training linear SVM, we will have two arrays which are support vector and alpha respectively and one floating-point number rho in the result XML file.
    //Multiple the alpha matrix with the support vector matrix and get a column vector. Then add the floating-point to the end of the column vector which is the final classifier we will used to detect the pedestrians.
    //Finally, replace the default classifier in OpenCV(cv::HOGDescriptor::setSVMDetector()) with the SVM classifier we created.
    
    //HOG descriptor dimention
    DescriptorDim = svm.get_var_count();
    //The number of support vector
    int supportVectorNum = svm.get_support_vector_count();
    cout<<"The number of support vector："<<supportVectorNum<<endl;
    
    //initialize alpha vector，length equals to the number of support vector
    Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);
    //initilize support Vector Matrix
    Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);
    //initialize result matrix, the length equals to HOG descriptor dimention
    Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);
    
    //copy the support vector data to the
    for(int i=0; i<supportVectorNum; i++)
    {
        const float * pSVData = svm.get_support_vector(i);
        for(int j=0; j<DescriptorDim; j++)
        {
            //cout<<pData[j]<<" ";
            supportVectorMat.at<float>(i,j) = pSVData[j];
        }
    }
    
    //copy alpha vector data to alphaMat
    double * pAlphaData = svm.get_alpha_vector();
    for(int i=0; i<supportVectorNum; i++)
    {
        alphaMat.at<float>(0,i) = pAlphaData[i];
    }
    
    //calculate-(alphaMat * supportVectorMat),put result into resultMat
    //gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);
    resultMat = -1 * alphaMat * supportVectorMat;
    
    //setSVMDetector(const vector<float>& detector)
    vector<float> myDetector;
    //copy resultMat data into myDetector
    for(int i=0; i<DescriptorDim; i++)
    {
        myDetector.push_back(resultMat.at<float>(0,i));
    }
    //add floating point rho at the end of vector
    myDetector.push_back(svm.get_rho());
    cout<<"HOG Dimension："<<myDetector.size()<<endl;
    //set HOGDescriptor
    HOGDescriptor myHOG;
    myHOG.setSVMDetector(myDetector);
    //myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());
    
    //save
    ofstream fout("HOGDetectorForOpenCV.txt");
    for(int i=0; i<myDetector.size(); i++)
    {
        fout<<myDetector[i]<<endl;
    }
    
    
    /**************READ IMAGE AND TEST PEDESTRIAN******************/
    Mat src = imread("/Users/chenxuanqi/Downloads/HOG-SVM/single-Demo/test-source/Test.jpg");
    vector<Rect> found, found_filtered;
    cout<<"Muti-pedestrian-detector:"<<endl;
    myHOG.detectMultiScale(src, found, 0, Size(8,8), Size(32,32), 1.05, 2);
    cout<<"The number of bounding box："<<found.size()<<endl;
    
    //put all the bounding box without overlaping into found_filtered
    for(int i=0; i < found.size(); i++)
    {
        Rect r = found[i];
        int j=0;
        for(; j < found.size(); j++)
            if(j != i && (r & found[j]) == r)
                break;
        if( j == found.size())
            found_filtered.push_back(r);
    }
    
    //Draw the bounding box
    //since the bounidng box detected by the HOG descirptor is a little bit larger than the real bounidng box
    //there is some parameter adjust here.
    for(int i=0; i<found_filtered.size(); i++)
    {
        Rect r = found_filtered[i];
        r.x += cvRound(r.width*0.1);
        r.width = cvRound(r.width*0.8);
        r.y += cvRound(r.height*0.07);
        r.height = cvRound(r.height*0.8);
        rectangle(src, r.tl(), r.br(), Scalar(0,255,0), 3);
    }
    
    imwrite("ImgProcessed.jpg",src);
    namedWindow("src",0);  
    imshow("src",src);  
    waitKey();//add to shows the picture
    
    tm.stop();
    cout<<"The running time："<<tm.getTimeSec()<<endl;
    cout << tm.getTimeTicks() << endl;
    
    system("pause");

}  

