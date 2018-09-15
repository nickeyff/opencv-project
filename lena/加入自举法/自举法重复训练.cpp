#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;

#define str "E:\\INRIAPerson\\HardExample\\"
int hardExampleCount = 0;


class MySVM : public CvSVM
{
	public:
		//获得SVM的决策函数中的alpha数组
		double * get_alpha_vector()
		{
			return this->decision_func->alpha;
		}

		//获得SVM的决策函数中的rho参数,即偏移量
		float get_rho()
		{
			return this->decision_func->rho;
		}
};
int main()
{
	Mat src;
	char saveName[256];//剪裁出来的hard example图片的文件名
	string ImgName;
	//ifstream fin_detector("HOGDetectorParagram.txt");
	ifstream fin_imgList("neg.txt");//打开原始负样本图片文件列表



	
	int DescriptorDim;
	MySVM svm;
	svm.load("SVM_HOG.xml");
	DescriptorDim = svm.get_var_count();//特征向量的维数，即HOG描述子的维数
	cout<<"描述子维数："<<DescriptorDim<<endl;
	int supportVectorNum = svm.get_support_vector_count();//支持向量的个数
	cout<<"支持向量个数："<<supportVectorNum<<endl;

	//Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha向量，长度等于支持向量个数
	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//支持向量矩阵
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha向量乘以支持向量矩阵的结果

	//将支持向量的数据复制到supportVectorMat矩阵中,共有supportVectorNum个支持向量，每个支持向量的数据有DescriptorDim维(种)
	for(int i=0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//返回第i个支持向量的数据指针
		for(int j=0; j<DescriptorDim; j++)
			supportVectorMat.at<float>(i,j) = pSVData[j];//第i个向量的第j维数据
	}

	//将alpha向量的数据复制到alphaMat中
	//double * pAlphaData = svm.get_alpha_vector();//返回SVM的决策函数中的alpha向量
	double * pAlphaData = svm.get_alpha_vector();
	for(int i=0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0,i) = pAlphaData[i];//alpha向量，长度等于支持向量个数
	}

	//计算-(alphaMat * supportVectorMat),结果放到resultMat中
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//不知道为什么加负号？
	resultMat = -1 * alphaMat * supportVectorMat;

	//得到最终的setSVMDetector(const vector<float>& detector)参数中可用的检测子
	vector<float> myDetector;
	//将resultMat中的数据复制到数组myDetector中
	for(int i=0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0,i));
	}
	myDetector.push_back(svm.get_rho());//最后添加偏移量rho，得到检测子
	cout<<"检测子维数："<<myDetector.size()<<endl;
	//设置HOGDescriptor的检测子
	HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);
	hog.setSVMDetector(myDetector);



	//从文件中读入自己训练的SVM参数
	//float temp;
	//vector<float> myDetector;//自己的检测器数组
	//while(!fin_detector.eof())
	//{
		//fin_detector >> temp;
		//myDetector.push_back(temp);
	//}
	//cout<<"检测子维数："<<myDetector.size()<<endl;
	//设置检测器参数为自己训练的SVM参数
	//HOGDescriptor hog;
	//hog.setSVMDetector(myDetector);



	while(getline(fin_imgList,ImgName))
	{
		cout<<"处理："<<ImgName<<endl;
		string fullName = "E:\\INRIAPerson\\Negjpg_undesign\\" + ImgName;
		src = imread(fullName);
		Mat img = src.clone();

		vector<Rect> found;
		hog.detectMultiScale(src, found, 0, Size(8,8),Size(32,32), 1.05, 2);
		//处理得到的矩形框
		for(int i=0; i < found.size(); i++)
		{
			//将矩形框轮廓限定在图像内部，r的x、y坐标是相对于源图像src来定义的
			Rect r = found[i];
			if(r.x < 0)
				r.x = 0;
			if(r.y < 0)
				r.y = 0;
			if(r.x + r.width > src.cols)
				r.width = src.cols - r.x;
			if(r.y + r.height > src.rows)
				r.height = src.rows - r.y;

			//将矩形框框出的图片保存为难例
			Mat hardExampleImg = src(r);//从原图上截取矩形框大小的图片
			resize(hardExampleImg,hardExampleImg,Size(64,128));//将剪裁出来的图片缩放为64*128大小
			sprintf(saveName,"hardexample%09d.jpg",hardExampleCount++);//生成hard example图片的文件名
			imwrite(saveName, hardExampleImg);//保存文件

			//画矩形框，因为hog检测出的矩形框比实际人体框要稍微大些,所以这里需要做一些调整
			r.x += cvRound(r.width*0.1);
			r.width = cvRound(r.width*0.8);
			rectangle(img, r.tl(), r.br(), Scalar(0,255,0), 3);
		}
		imwrite(str+ImgName,img);
		imshow("src",src);
		waitKey(100);
	}
	system("pause");
}