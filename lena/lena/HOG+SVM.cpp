#include<iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;
using namespace std;

#define PosSamNO 1126    //����������
#define NegSamNO 1210    //����������
#define TRAIN false    //�Ƿ����ѵ��,true��ʾ����ѵ����false��ʾ��ȡxml�ļ��е�SVMģ��
//#define CENTRAL_CROP false   //true:ѵ��ʱ����96*160��INRIA������ͼƬ���ó��м��64*128��С����

//HardExample�����������������HardExampleNO����0����ʾ�������ʼ���������󣬼�������HardExample����������
//��ʹ��HardExampleʱ��������Ϊ0����Ϊ������������������������ά����ʼ��ʱ�õ����ֵ
#define HardExampleNO 7715

//��ʹ�õĲ���opencv�Դ������˼�����������Լ�ѵ����svm+hog��������
//�̳���CvSVM���࣬��Ϊ����setSVMDetector()���õ��ļ���Ӳ���ʱ����Ҫ�õ�ѵ���õ�SVM��decision_func������
//��ͨ���鿴CvSVMԴ���֪decision_func������protected���ͱ������޷�ֱ�ӷ��ʵ���ֻ�ܼ̳�֮��ͨ����������
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
	//��ⴰ��,��ߴ�(16,16),�鲽��(8,8),cell�ߴ�(8,8),ֱ��ͼbin����9
	HOGDescriptor hog(Size(64,128),Size(16,16),Size(8,8),Size(8,8),9);//HOG���������������HOG������
	//HOGDescriptor hog(Size(64,128),Size(8,8),Size(4,4),Size(4,4),9);
	//HOG�����ӵ�ά������ͼƬ��С����ⴰ�ڴ�С�����С��ϸ����Ԫ��ֱ��ͼbin��������
	int DescriptorDim;
	//SVM������
	MySVM svm;

	//��TRAINΪtrue������ѵ��������
	if(TRAIN)
	{
		string ImgName;//ͼƬ��(����·��)
		ifstream finPos("pos.txt");//������ͼƬ���ļ����б�
		ifstream finNeg("neg.txt");//������ͼƬ���ļ����б�

		Mat sampleFeatureMat;//����ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��
		Mat sampleLabelMat;//ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����

		//���ζ�ȡ������ͼƬ������HOG������
		for(int num=0; num<PosSamNO && getline(finPos,ImgName); num++)
		{
			cout<<"����"<<ImgName<<endl;
			ImgName = "E:\\INRIAPerson\\Posjpg64_128\\" + ImgName;//������������·����
			Mat src = imread(ImgName);//��ȡͼƬ
			//resize(src,src,Size(64,128));

			vector<float> descriptors;//HOG����������
			hog.compute(src,descriptors,Size(8,8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
			//cout<<"������ά����"<<descriptors.size()<<endl;

			//�����һ������ʱ��ʼ�����������������������Ϊֻ��֪��������������ά�����ܳ�ʼ��������������
			if( 0 == num )
			{
				DescriptorDim = descriptors.size();//HOG�����ӵ�ά��
				//��ʼ������ѵ������������������ɵľ��������������������ĸ�������������HOG������ά��sampleFeatureMat
				sampleFeatureMat = Mat::zeros(PosSamNO+NegSamNO+HardExampleNO, DescriptorDim, CV_32FC1);
				//��ʼ��ѵ����������������������������������ĸ�������������1��1��ʾ���ˣ�-1��ʾ����
				sampleLabelMat = Mat::zeros(PosSamNO+NegSamNO+HardExampleNO, 1, CV_32FC1);
			}

			//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for(int i=0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num,i) = descriptors[i];//��num�����������������еĵ�i��Ԫ��

			sampleLabelMat.at<float>(num,0) = 1;//���������Ϊ1������
		}

		//���ζ�ȡ������ͼƬ������HOG������
		for(int num=0; num<NegSamNO && getline(finNeg,ImgName); num++)
		{
			cout<<"����"<<ImgName<<endl;
			ImgName = "E:\\INRIAPerson\\Negjpg_undesign\\" + ImgName;//���ϸ�������·����
			Mat src = imread(ImgName);//��ȡͼƬ
			//resize(src,src,Size(64,128));

			vector<float> descriptors;//HOG����������
			hog.compute(src,descriptors,Size(8,8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
			//cout<<"������ά����"<<descriptors.size()<<endl;

			//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
			for(int i=0; i<DescriptorDim; i++)
				sampleFeatureMat.at<float>(num+PosSamNO,i) = descriptors[i];//��PosSamNO+num�����������������еĵ�i��Ԫ��
			sampleLabelMat.at<float>(num+PosSamNO,0) = -1;//���������Ϊ-1������
		}

		
		//����HardExample������
		if(HardExampleNO > 0)
		{
			ifstream finHardExample("HardExample.txt");//HardExample���������ļ����б�
			//���ζ�ȡHardExample������ͼƬ������HOG������
			for(int num=0; num<HardExampleNO && getline(finHardExample,ImgName); num++)
			{
				cout<<"����"<<ImgName<<endl;
				ImgName = "E:\\INRIAPerson\\HardExample\\" + ImgName;//����HardExample��������·����
				Mat src = imread(ImgName);//��ȡͼƬ
				//resize(src,src,Size(64,128));

				vector<float> descriptors;//HOG����������
				hog.compute(src,descriptors,Size(8,8));//����HOG�����ӣ���ⴰ���ƶ�����(8,8)
				//cout<<"������ά����"<<descriptors.size()<<endl;

				//������õ�HOG�����Ӹ��Ƶ�������������sampleFeatureMat
				for(int i=0; i<DescriptorDim; i++)
					sampleFeatureMat.at<float>(num+PosSamNO+NegSamNO,i) = descriptors[i];
				sampleLabelMat.at<float>(num+PosSamNO+NegSamNO,0) = -1;//���������Ϊ-1������
			}
		}

		////���������HOG�������������ļ�
		//ofstream fout("SampleFeatureMat.txt");
		//for(int i=0; i<PosSamNO+NegSamNO; i++)
		//{
		//  fout<<i<<endl;
		//  for(int j=0; j<DescriptorDim; j++)
		//      fout<<sampleFeatureMat.at<float>(i,j)<<"  ";
		//  fout<<endl;
		//}

		//ѵ��SVM������
		//������ֹ��������������1000�λ����С��FLT_EPSILONʱֹͣ����
		CvTermCriteria criteria = cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
		//SVM������SVM����ΪC_SVC�����Ժ˺������ɳ�����C=0.01
		CvSVMParams param(CvSVM::C_SVC, CvSVM::LINEAR, 0, 1, 0, 0.01, 0, 0, 0, criteria);
		cout<<"��ʼѵ��SVM������"<<endl;
		svm.train(sampleFeatureMat, sampleLabelMat, Mat(), Mat(), param);
		cout<<"ѵ�����"<<endl;
		svm.save("SVM_HOG_�Ծ�ѵ��һ��.xml");//��ѵ���õ�SVMģ�ͱ���Ϊxml�ļ�
	}

	//��TRAINΪfalse����XML�ļ���ȡѵ���õķ�����
	else
	{
		svm.load("SVM_HOG_�Ծ�ѵ��һ��.xml");
	}

	/*����SVMѵ����ɺ�õ���XML�ļ����棬��һ�����飬����support vector������һ�����飬����alpha,��һ��������������rho; 
    ��alpha����ͬsupport vector��ˣ�ע�⣬alpha*supportVector,���õ�һ����������֮���ٸ���������������һ��Ԫ��rho�� 
    ��ˣ���õ���һ�������������ø÷�������ֱ���滻opencv�����˼��Ĭ�ϵ��Ǹ���������cv::HOGDescriptor::setSVMDetector()���� 
    �Ϳ����������ѵ������ѵ�������ķ������������˼���ˡ�*/

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
	//svm.predict�жϷ�ʽ�ǣ�alphaMat*supportVectorMat-rho���Ϊ����Ϊ����������rho-alphaMat*supportVectorMat���Ϊ����Ϊ������
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
	HOGDescriptor myHOG;
	myHOG.setSVMDetector(myDetector);
	//myHOG.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

	//�������Ӳ������ļ�
	//ofstream fout("HOGDetectorParagram_�Ծ�ѵ��һ��.txt");
	//for(int i=0; i<myDetector.size(); i++)
		//fout<<myDetector[i]<<endl;

	/**************����ͼƬ����HOG���˼��******************/
	Mat src = imread("test10.png");
	vector<Rect> found, found_filtered;//���ο�����
	cout<<"���ж�߶�HOG������"<<endl;
	myHOG.detectMultiScale(src, found, 0, Size(8,8), Size(32,32), 1.05, 2);//��ͼƬ���ж�߶����˼��
	cout<<"�ҵ��ľ��ο������"<<found.size()<<endl;

	//�ҳ�����û��Ƕ�׵ľ��ο�r,������found_filtered��,�����Ƕ�׵Ļ�,��ȡ���������Ǹ����ο����found_filtered��
	for(int i=0; i < found.size(); i++)
	{
		Rect r = found[i];
		int j=0;
		for(; j < found.size(); j++)
		{
			if(j != i && (r & found[j]) == r)//˵��r�Ǳ�Ƕ����found[j]����ģ�������ǰ��r
				break;
		}
		if( j == found.size())//rû�б�Ƕ���ڵ�0��1��2...found.size()-1�ŵľ��ο��ڣ���r�Ƿ���������
			found_filtered.push_back(r);
	}

	//�����ο���Ϊhog�����ľ��ο��ʵ�������Ҫ��΢��Щ,����������Ҫ��һЩ����
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
	waitKey();//ע�⣺imshow֮������waitKey�������޷���ʾͼ��

	system("pause");
}