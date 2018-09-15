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
		//���SVM�ľ��ߺ����е�alpha����
		double * get_alpha_vector()
		{
			return this->decision_func->alpha;
		}

		//���SVM�ľ��ߺ����е�rho����,��ƫ����
		float get_rho()
		{
			return this->decision_func->rho;
		}
};
int main()
{
	Mat src;
	char saveName[256];//���ó�����hard exampleͼƬ���ļ���
	string ImgName;
	//ifstream fin_detector("HOGDetectorParagram.txt");
	ifstream fin_imgList("neg.txt");//��ԭʼ������ͼƬ�ļ��б�



	
	int DescriptorDim;
	MySVM svm;
	svm.load("SVM_HOG.xml");
	DescriptorDim = svm.get_var_count();//����������ά������HOG�����ӵ�ά��
	cout<<"������ά����"<<DescriptorDim<<endl;
	int supportVectorNum = svm.get_support_vector_count();//֧�������ĸ���
	cout<<"֧������������"<<supportVectorNum<<endl;

	//Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);//alpha���������ȵ���֧����������
	Mat alphaMat = Mat::zeros(1, supportVectorNum, CV_32FC1);
	Mat supportVectorMat = Mat::zeros(supportVectorNum, DescriptorDim, CV_32FC1);//֧����������
	Mat resultMat = Mat::zeros(1, DescriptorDim, CV_32FC1);//alpha��������֧����������Ľ��

	//��֧�����������ݸ��Ƶ�supportVectorMat������,����supportVectorNum��֧��������ÿ��֧��������������DescriptorDimά(��)
	for(int i=0; i<supportVectorNum; i++)
	{
		const float * pSVData = svm.get_support_vector(i);//���ص�i��֧������������ָ��
		for(int j=0; j<DescriptorDim; j++)
			supportVectorMat.at<float>(i,j) = pSVData[j];//��i�������ĵ�jά����
	}

	//��alpha���������ݸ��Ƶ�alphaMat��
	//double * pAlphaData = svm.get_alpha_vector();//����SVM�ľ��ߺ����е�alpha����
	double * pAlphaData = svm.get_alpha_vector();
	for(int i=0; i<supportVectorNum; i++)
	{
		alphaMat.at<float>(0,i) = pAlphaData[i];//alpha���������ȵ���֧����������
	}

	//����-(alphaMat * supportVectorMat),����ŵ�resultMat��
	//gemm(alphaMat, supportVectorMat, -1, 0, 1, resultMat);//��֪��Ϊʲô�Ӹ��ţ�
	resultMat = -1 * alphaMat * supportVectorMat;

	//�õ����յ�setSVMDetector(const vector<float>& detector)�����п��õļ����
	vector<float> myDetector;
	//��resultMat�е����ݸ��Ƶ�����myDetector��
	for(int i=0; i<DescriptorDim; i++)
	{
		myDetector.push_back(resultMat.at<float>(0,i));
	}
	myDetector.push_back(svm.get_rho());//������ƫ����rho���õ������
	cout<<"�����ά����"<<myDetector.size()<<endl;
	//����HOGDescriptor�ļ����
	HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);
	hog.setSVMDetector(myDetector);



	//���ļ��ж����Լ�ѵ����SVM����
	//float temp;
	//vector<float> myDetector;//�Լ��ļ��������
	//while(!fin_detector.eof())
	//{
		//fin_detector >> temp;
		//myDetector.push_back(temp);
	//}
	//cout<<"�����ά����"<<myDetector.size()<<endl;
	//���ü��������Ϊ�Լ�ѵ����SVM����
	//HOGDescriptor hog;
	//hog.setSVMDetector(myDetector);



	while(getline(fin_imgList,ImgName))
	{
		cout<<"����"<<ImgName<<endl;
		string fullName = "E:\\INRIAPerson\\Negjpg_undesign\\" + ImgName;
		src = imread(fullName);
		Mat img = src.clone();

		vector<Rect> found;
		hog.detectMultiScale(src, found, 0, Size(8,8),Size(32,32), 1.05, 2);
		//����õ��ľ��ο�
		for(int i=0; i < found.size(); i++)
		{
			//�����ο������޶���ͼ���ڲ���r��x��y�����������Դͼ��src�������
			Rect r = found[i];
			if(r.x < 0)
				r.x = 0;
			if(r.y < 0)
				r.y = 0;
			if(r.x + r.width > src.cols)
				r.width = src.cols - r.x;
			if(r.y + r.height > src.rows)
				r.height = src.rows - r.y;

			//�����ο�����ͼƬ����Ϊ����
			Mat hardExampleImg = src(r);//��ԭͼ�Ͻ�ȡ���ο��С��ͼƬ
			resize(hardExampleImg,hardExampleImg,Size(64,128));//�����ó�����ͼƬ����Ϊ64*128��С
			sprintf(saveName,"hardexample%09d.jpg",hardExampleCount++);//����hard exampleͼƬ���ļ���
			imwrite(saveName, hardExampleImg);//�����ļ�

			//�����ο���Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,����������Ҫ��һЩ����
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