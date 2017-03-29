package shmtu.wekautils;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import shmtu.extral.attributeSelection.CHIAttributeEval;
import shmtu.extral.attributeSelection.IGAttributeEval2;
import shmtu.extral.attributeSelection.MIAttributeEval;
import shmtu.util.CommonUtils;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.functions.SMO;
import weka.classifiers.meta.AttributeSelectedClassifier;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSink;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.supervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.StringToWordVector;

/**
 * weka 工具类
 * 1.在对过滤器操作之前 设置选线操作setOptions必须在 setInputformat之前
 * 2.转换词向量操作 需要指定类别 类属性的位置。在所有操作之前 都要确保设置好类属性。
 * 3.把数据集对象instances作为参数的时候要小心 不能来回修改使用
 * 4.libsvm 使用需要设置 -S 参数 =1
 * 5.经验证 元分类器 需要时间很多，并且元分类器 效果不好
 * 6.不知道为什么 如果在 MI特征选择和CHI特征选择扩展类中添加离散方法，那么结果和IG的结果是一样的，属性选择的结果一样
 * 7.在文本分类中 IG特征选择需要设置 binary=true，就不会进行离散化 并且节省大量时间。离散化是由于 IG特征选择针对连续值的处理方式，离散化后再处理 在MI和CHI中不需要
 * 并不是 与分类精度有关的 
 * @author HP_xiaochao
 *
 */
public class WekaUtil {
	public static int maxTextCopyNum = 15;//扩展的时候规定最长的扩展长度
	public static int minAttributeLen = 10;//过滤掉过短文本
	/**
	 * 枚举类 后来没有用
	 * 原计划 使用这个枚举类实现 对各种分类算法的区别
	 * @author HP_xiaochao
	 *2016年12月31日
	 *
	 */
	public enum CatagoryType{
		//ig:0,mi:1,chi:2
		//bayes:0,multinomial:1,svm:2
		//isMeta:0,1
		igNaiveBayesMeta(0,0,"igNaiveBayesMeta",1),igNaiveBayes(0,0,"igNaiveBayes",0),
		igNaiveBayesMutinomialMeta(0,1,"igNaiveBayesMutinomialMeta",1),igNaiveBayesMutinomial(0,1,"igNaiveBayesMutinomial",0),
		igSVMMeta(0,2,"igSVMMeta",1),igSVM(0,2,"igSVM",0),
		
		miNaiveBayesMeta(1,0,"miNaiveBayesMeta",1),miNaiveBayes(1,0,"miNaiveBayes",0),
		miNaiveBayesMutinomialMeta(1,1,"miNaiveBayesMutinomialMeta",1),miNaiveBayesMutinomial(1,1,"miNaiveBayesMutinomial",0),
		miSVMMeta(1,2,"miSVMMeta",1),miSVM(1,2,"miSVM",0),
		
		chiNaiveBayesMeta(2,0,"chiNaiveBayesMeta",1),ichiNaiveBayes(2,0,"chiNaiveBayes",0),
		chiNaiveBayesMutinomialMeta(2,1,"chiNaiveBayesMutinomialMeta",1),chiNaiveBayesMutinomial(2,1,"chiNaiveBayesMutinomial",0),
		chiSVMMeta(2,2,"chiSVMMeta",1),chiSVM(2,2,"chiSVM",0);
		
		public int attributeSelectionType;//0,1,2
		public int classfierType;//0,1
		public String str;
		public int isMeta;
		public int getIsMeta() {
			return isMeta;
		}
		public void setIsMeta(int isMeta) {
			this.isMeta = isMeta;
		}
		public int getAttributeSelectionType() {
			return attributeSelectionType;
		}
		public void setAttributeSelectionType(int attributeSelectionType) {
			this.attributeSelectionType = attributeSelectionType;
		}
		public int getClassfierType() {
			return classfierType;
		}
		public void setClassfierType(int classfierType) {
			this.classfierType = classfierType;
		}
		public String getStr() {
			return str;
		}
		public void setStr(String str) {
			this.str = str;
		}
		CatagoryType(int attributeSelectionType,int classfierType,String str,int isMeta){
			this.str = str;
			this.attributeSelectionType = attributeSelectionType;
			this.classfierType = classfierType;
			this.isMeta = isMeta;
		}
	}
	/**
	 * 加载方式 1.
	 * 通过加载器加载arff文件
	 * @param filePath
	 * @return
	 * @throws Exception
	 */
	public static Instances loadArffByLoader(String arffFilePath) throws Exception{
		ArffLoader arff = new ArffLoader();
		arff.setSource(new File(arffFilePath));
		Instances instances = arff.getDataSet();
		System.out.println("加载数据文件成功");
		return instances;
	}
	/**
	 * 加载方式 2.
	 * 通过 Data Source 类加载arff文件
	 * @param arffFilePath
	 * @return
	 * @throws Exception
	 */
	public static Instances loadArffByDataSource(String arffFilePath) throws Exception{
		Instances instances = DataSource.read(arffFilePath);
		System.out.println("加载数据文件成功");
		return instances;
	}
	/**
	 * 保存实例数据 方式1.
	 * 通过 DataSink 保存
	 * @param instances
	 * @param saveArffFilePath
	 * @throws Exception
	 */
	public static void saveArffByInstancesByDataSink(Instances instances,String saveArffFilePath) throws Exception{
		DataSink.write(saveArffFilePath, instances);
		System.out.println("保存实例正确： to " + saveArffFilePath);
	}
	/**
	 * 保存数据 方式2.
	 * 通过转换器
	 * @param instances
	 * @param saveArffFilePath
	 * @throws Exception
	 */
	public static void saveArffBySaver(Instances instances,String saveArffFilePath) throws Exception{
		ArffSaver saver = new ArffSaver();
		saver.setInstances(instances);
		saver.setFile(new File(saveArffFilePath));
		saver.writeBatch();
		System.out.println("保存实例数据正确：to " + saveArffFilePath);
	}
	/**
	 * 打印 数据实例头信息
	 * @param instances
	 */
	public static void printHeader(Instances instances){
		if(instances!=null){
			System.out.println(new Instances(instances,0));
		}
	}
	/**
	 * 返回实例数据集 的属性列表列表 没有类属性的
	 * @param instances
	 * @return
	 */
	public static List<Attribute> getAttributesByINstances(Instances instances){
		List<Attribute> attributes = new ArrayList<Attribute>();
		if(instances!=null){
			int attributeNum = instances.numAttributes();
			System.out.println("属性数目:"+attributeNum);
			for(int i=0;i<attributeNum;i++){
				if(i!=instances.classIndex())
					attributes.add(instances.attribute(i));
			}
		}
		return attributes;
	}
	/**
	 * 文档实例 转换成词向量实例 通过 StringToVctor
	 * 返回新数据实例
	 * @param instances
	 * @param wordToKeep
	 * @return
	 * @throws Exception
	 */
	public static Instances stringToVectorWeightBoolean(Instances instances,Integer wordToKeep) throws Exception{
		String methodType = "文档数据转换词向量 权值采用 布尔值 wordToKeep= " + wordToKeep;
		CommonUtils.print("----------- " + methodType + " -------------");
		CommonUtils.print("确保设置好 类属性位置");
		StringToWordVector vector = new StringToWordVector();
		if(wordToKeep!=null)
			vector.setWordsToKeep(wordToKeep);
		vector.setInputFormat(instances);
		Instances newArff = Filter.useFilter(instances, vector);
		CommonUtils.print("-------------- 文档转变词向量成功 特征数目:" + newArff.numAttributes() + "-------------------");
		CommonUtils.print("第一个属性是:" + newArff.attribute(0).name());
		CommonUtils.print("最后属性是:" + newArff.attribute(newArff.numAttributes()-1).name());
		return newArff;
	}
	/**
	 * 文档词向量转换 权重tf值
	 * @param instances
	 * @return
	 * @throws Exception
	 */
	public static Instances  stringToVerctorWeightIF(Instances instances,Integer newWordsToKeep) throws Exception{
		String methodType = "文档数据转换词向量 权值采用 tf值 numWordsToKeep = " + newWordsToKeep;
		CommonUtils.print("----------- " + methodType + " -------------");
		CommonUtils.print("确保设置好 类属性位置");
		StringToWordVector vector = new StringToWordVector();
		vector.setTFTransform(true);
		vector.setOutputWordCounts(true);
		if(newWordsToKeep!=null)
			vector.setWordsToKeep(newWordsToKeep);
		vector.setInputFormat(instances);
		Instances newArff = Filter.useFilter(instances, vector);
		CommonUtils.print("-------------- 文档转变词向量成功 特征数目:" + newArff.numAttributes() + "-------------------");
		CommonUtils.print("第一个属性是:" + newArff.attribute(0));
		CommonUtils.print("最后属性是:" + newArff.attribute(newArff.numAttributes()-1));
		return newArff;
	}
	/**
	 * 文档词向量转换 权重ifidf
	 * @param instances
	 * @return
	 * @throws Exception
	 */
	public static Instances  stringToVerctorWeightIFIDF(Instances instances,Integer numWordsTcoKeep) throws Exception{
		String methodType = "文档数据转换词向量 权值采用 tfidf值";
		CommonUtils.print("----------- " + methodType + " -------------");
		CommonUtils.print("确保设置好 类属性位置." + instances.attribute(instances.numAttributes()-1));
		StringToWordVector vector = new StringToWordVector();
		vector.setTFTransform(true);
		vector.setIDFTransform(true);
		vector.setOutputWordCounts(true);
		if(numWordsTcoKeep!=null)
			vector.setWordsToKeep(numWordsTcoKeep);
		vector.setInputFormat(instances);
		Instances newArff = Filter.useFilter(instances, vector);
		CommonUtils.print("-------------- 文档转变词向量成功 特征数目:" + newArff.numAttributes() + "-------------------");
		CommonUtils.print("第一个属性是:" + newArff.attribute(0));
		CommonUtils.print("最后属性是:" + newArff.attribute(newArff.numAttributes()-1));
		return newArff;
	}
	public static void catagoryByType(CatagoryType type,Instances instances,int maxNumAttribute){
//		int classType = type.getClassfierType()
	}
	/**
	 * IG 特征选择方法 Naivebayes分类方法
	 *  使用weka元分类器  AttributeSelectedClassifier
	 * @param instances
	 * @throws Exception
	 */
	public static void catagoryByIGNaiveBayesMeta(Instances instances,int maxNumAttribute) throws Exception{
		String methodType = "IG_特征选择方法_Naivebayes_meta分类方法";
		CommonUtils.print("--------- " + methodType +  ", maxNumAttribute is " + maxNumAttribute + " ----------");
		//1.初始化 特征评估函数和搜索策略
		InfoGainAttributeEval igAttributeEval = new InfoGainAttributeEval();
		igAttributeEval.setBinarizeNumericAttributes(true);
//		CHIAttributeEval chiAttributeEval = new CHIAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		//2.构造元分类器
		AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
		NaiveBayes nBayes = new NaiveBayes();
		classifier.setClassifier(nBayes);
		classifier.setEvaluator(igAttributeEval);
		classifier.setSearch(ranker);
		//3.评估元分类器
		Evaluation evaluation = new Evaluation(instances);
	    evaluation.crossValidateModel(classifier, instances, 10, new Random(1));
	    //4.打印并保存结果
	    printEvaluation(evaluation,"result/" + methodType + "_" + maxNumAttribute + ".txt");
	}
	/**
	 * 采用元分类器的模式 IG特征选择 贝叶斯多项式模型分类器
	 * @param instances
	 * @param maxNumAttribute
	 * @throws Exception
	 */
	public static void catagoryByIGNaiveBayesMultinomialMeta(Instances instances,int maxNumAttribute) throws Exception{
		String methodType = "IG_特征选择方法_NaivebayesMultinomial_meta分类方法";
		CommonUtils.print("--------- " + methodType +  ", maxNumAttribute is " + maxNumAttribute + " ----------");
		//1.构造特征选择评估函数 搜索策略
		InfoGainAttributeEval igAttributeEval = new InfoGainAttributeEval();
		igAttributeEval.setBinarizeNumericAttributes(true);
//		CHIAttributeEval chiAttributeEval = new CHIAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		//2.构造元分类器
		AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
		NaiveBayesMultinomial nBayesM = new NaiveBayesMultinomial();
		classifier.setClassifier(nBayesM);
		classifier.setEvaluator(igAttributeEval);
		classifier.setSearch(ranker);
		//3.评估元分类器
		Evaluation evaluation = new Evaluation(instances);
	    evaluation.crossValidateModel(classifier, instances, 10, new Random(1));
	    //4.打印并保存结果
	    printEvaluation(evaluation,"result/" + methodType + "_" + maxNumAttribute + ".txt");
	}
	/**
	 * 分类方法
	 * IG特征选择 贝叶斯多项式模型
	 * @param instances
	 * @param maxNumAttribute
	 * @throws Exception
	 */
	public static void catagoryByIGNaiveBayesMultinomial(Instances instances,int maxNumAttribute) throws Exception{
		String methodType = "IG_特征选择方法_NaiveBayesMultinomial分类方法";
		CommonUtils.print("--------- " + methodType +  ", maxNumAttribute is " + maxNumAttribute + " ----------");
		//
		CommonUtils.print("classfier method:NaiveBayesMultinomial ;attribute selection:IG;attributeNum:"+maxNumAttribute);
		//1.特征选择评估函数 搜索策略
		InfoGainAttributeEval igAttributeEval = new InfoGainAttributeEval();
		igAttributeEval.setBinarizeNumericAttributes(true);
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		//2.初始化 特征选择过滤器
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(igAttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(instances);
		//3.特征选择 更新数据集
		Instances afterAsArrff = Filter.useFilter(instances, asFilter);
		afterAsArrff.setClassIndex(afterAsArrff.numAttributes()-1);
		//4.分类 并评估
		Evaluation evaluation = new Evaluation(afterAsArrff);
		NaiveBayesMultinomial nBayesM = new NaiveBayesMultinomial();
		evaluation.crossValidateModel(nBayesM, afterAsArrff, 10, new Random(1));
		//5.打印 并保存结果
		printEvaluation(evaluation,"result/" + methodType + "_" + maxNumAttribute + ".txt");
	}
	/**
	 * 元分类器 分类方法
	 * IG特征选择方法 libsvm分类器
	 * @param instances
	 * @param maxNumAttribute
	 * @throws Exception
	 */
	public static void catagoryByIGLibSVMMeta(Instances instances,int maxNumAttribute) throws Exception{
		String methodType = "IG_特征选择方法_SVM_meta分类方法";
		CommonUtils.print("--------- " + methodType +  ", maxNumAttribute is " + maxNumAttribute + " ----------");
		//1.定义 特征选择评估函数 搜索策略
		InfoGainAttributeEval igAttributeEval = new InfoGainAttributeEval();
		igAttributeEval.setBinarizeNumericAttributes(true);
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		//2.初始化 特征选择元分类器
		AttributeSelectedClassifier classifier = new AttributeSelectedClassifier();
		LibSVM libsvm = new LibSVM();
		libsvm.setOptions(new String[]{
				"-S","1"
		});
		classifier.setClassifier(libsvm);
		classifier.setEvaluator(igAttributeEval);
		classifier.setSearch(ranker);
		//3.评估
		Evaluation evaluation = new Evaluation(instances);
	    evaluation.crossValidateModel(classifier, instances, 10, new Random(1));
	    //4.打印并保存结果
	    printEvaluation(evaluation,"result/" + methodType + "_" + maxNumAttribute + ".txt");
	}
	/**
	 * 分类器 分类方法
	 * IG特征选择 朴素贝叶斯分类器模型
	 * @param instances
	 * @param maxNumAttribute
	 * @throws Exception
	 */
	public static void catagoryByIGNaiveBayes(Instances instances,int maxNumAttribute) throws Exception{
		String methodType = "IG_特征选择方法_NaiveBayes分类方法";
		CommonUtils.print("--------- " + methodType +  ", maxNumAttribute is " + maxNumAttribute + " ----------");
		//
		CommonUtils.print("classfier method:NaiveBayes ;attribute selection:IG;attributeNum:"+maxNumAttribute);
		//1.初始化 特征选择评估函数 搜索策略
		InfoGainAttributeEval igAttributeEval = new InfoGainAttributeEval();
		igAttributeEval.setBinarizeNumericAttributes(true);
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		//2.初始化特征选择过滤器
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(igAttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(instances);
		//3.特征选择 生成新的数据集
		Instances afterAsArrff = Filter.useFilter(instances, asFilter);
		afterAsArrff.setClassIndex(afterAsArrff.numAttributes()-1);
		//4.分类 并对新的数据集评估
		Evaluation evaluation = new Evaluation(afterAsArrff);
		NaiveBayes nBayes = new NaiveBayes();
		evaluation.crossValidateModel(nBayes, afterAsArrff, 10, new Random(1));
		//5.打印并保存结果
		printEvaluation(evaluation,"result/" + methodType + "_" + maxNumAttribute + ".txt");
	}
	public static void catagoryByIG2NaiveBayes(Instances instances,int maxNumAttribute) throws Exception{
		String methodType = "IG2_特征选择方法_NaiveBayes分类方法";
		CommonUtils.print("--------- " + methodType +  ", maxNumAttribute is " + maxNumAttribute + " ----------");
		//
		CommonUtils.print("classfier method:NaiveBayes ;attribute selection:IG2;attributeNum:"+maxNumAttribute);
		//1.初始化 特征选择评估函数 搜索策略
		IGAttributeEval2 ig2AttributeEval = new IGAttributeEval2();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		//2.初始化特征选择过滤器
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(ig2AttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(instances);
		//3.特征选择 生成新的数据集
		Instances afterAsArrff = Filter.useFilter(instances, asFilter);
		afterAsArrff.setClassIndex(afterAsArrff.numAttributes()-1);
		//4.分类 并对新的数据集评估
		Evaluation evaluation = new Evaluation(afterAsArrff);
		NaiveBayes nBayes = new NaiveBayes();
		evaluation.crossValidateModel(nBayes, afterAsArrff, 10, new Random(1));
		//5.打印并保存结果
		printEvaluation(evaluation,"result/" + methodType + "_" + maxNumAttribute + ".txt");
	}
	
	/**
	 * 结合特征选择IG和LIBSVM分类器
	 * 
	 * @param instances
	 * @throws Exception
	 */
	public static void catagoryByIGLibSVM(Instances instances,int maxNumAttribute) throws Exception{
		String methodType = "IG_特征选择方法_libsvm分类方法";
		CommonUtils.print("--------- " + methodType +  ", maxNumAttribute is " + maxNumAttribute + " ----------");
		//
		CommonUtils.print("classfier method:libsvm ;attribute selection:IG;attributeNum:"+maxNumAttribute);
		//1.初始化 特征选择评估函数和搜索策略
		InfoGainAttributeEval igAttributeEval = new InfoGainAttributeEval();
		igAttributeEval.setBinarizeNumericAttributes(true);//二值化方式
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		//2.初始化 特征选择过滤器
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(igAttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(instances);
		//3.特征选择生成新的数据集
		Instances afterAsArrff = Filter.useFilter(instances, asFilter);
		afterAsArrff.setClassIndex(afterAsArrff.numAttributes()-1);
		//4.分类 并评估
		Evaluation evaluation = new Evaluation(afterAsArrff);
		LibSVM libsvm = new LibSVM();
		libsvm.setOptions(new String[]{
				"-S","1"
		});
		//
		evaluation.crossValidateModel(libsvm, afterAsArrff, 10, new Random(1));
		//5.打印 并保存结果
		printEvaluation(evaluation,"result/" + methodType + "_" + maxNumAttribute + ".txt");
	}
	/**
	 * 分类器 分类并评估
	 * CHI 特征选择 贝叶斯多项式模型
	 * @param instances
	 * @param maxNumAttribute
	 * @throws Exception
	 */
	public static void catagoryByCHINaiveBayesMultinomial(Instances instances,int maxNumAttribute) throws Exception{
		String methodType = "CHI_特征选择方法_NaivebayesMultinomial分类方法";
		CommonUtils.print("--------- " + methodType +  ", maxNumAttribute is " + maxNumAttribute + " ----------");
		//1.定义评估函数 搜索策略
		CHIAttributeEval chiAttributeEval = new CHIAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		//2.特征选择
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(chiAttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(instances);
		//3.特征选择生成新的实例集
		Instances afterAsArrff = Filter.useFilter(instances, asFilter);
		
		afterAsArrff.setClassIndex(afterAsArrff.numAttributes()-1);
		WekaUtil.writeAttributeToFile("result/attributes_" + methodType + "_" + maxNumAttribute + ".txt",WekaUtil.getAttributesByINstances(afterAsArrff));
		//4.分类器创建
		NaiveBayesMultinomial nBayesM = new NaiveBayesMultinomial();
		//5.对特征选择后的新实例集交叉验证
		Evaluation evaluation = new Evaluation(afterAsArrff);
	    evaluation.crossValidateModel(nBayesM, afterAsArrff, 10, new Random(1));
	    //6.打印并保存评估结果
	    printEvaluation(evaluation,"result/" + methodType + "_" + maxNumAttribute + ".txt");
	}
	/**
	 * 分类器 分类并评估
	 * IG2 特征选择算法  贝叶斯多项式模型
	 * IG2 特征选择方法 等价于 IG特征选择方法 设置二值化后的效果
	 * @param instances
	 * @param maxNumAttribute
	 * @throws Exception
	 */
	public static void catagoryByIG2NaiveBayesMultinomial(Instances instances,int maxNumAttribute) throws Exception{
		String methodType = "ig2_特征选择方法_NaivebayesMultinomial分类方法";
		CommonUtils.print("--------- " + methodType +  ", maxNumAttribute is " + maxNumAttribute + " ----------");
		//1.定义评估函数 搜索策略
		IGAttributeEval2 igAttributeEval = new IGAttributeEval2();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		//2.特征选择
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(igAttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(instances);
		//3.特征选择生成新的实例集
		Instances afterAsArrff = Filter.useFilter(instances, asFilter);
		afterAsArrff.setClassIndex(afterAsArrff.numAttributes()-1);
		WekaUtil.writeAttributeToFile("result/attributes_" + methodType + "_" + maxNumAttribute + ".txt",WekaUtil.getAttributesByINstances(afterAsArrff));
		//4.分类器创建
		NaiveBayesMultinomial nBayesM = new NaiveBayesMultinomial();
		//5.对特征选择后的新实例集交叉验证
		Evaluation evaluation = new Evaluation(afterAsArrff);
	    evaluation.crossValidateModel(nBayesM, afterAsArrff, 10, new Random(1));
	    //6.打印并保存评估结果
	    printEvaluation(evaluation,"result/" + methodType + "_" + maxNumAttribute + ".txt");
	}
//	/**
//	 * 训练分类器模型 并返回分类器
//	 * @param instances
//	 * @param maxNumAttribute
//	 * @return
//	 * @throws Exception
//	 */
//	public static NaiveBayesMultinomial getClassfierByCHINaiveBayesMultinomial(Instances instances,int maxNumAttribute) throws Exception{
//		String methodType = "CHI_特征选择方法_NaivebayesMultinomial分类方法";
//		CommonUtils.print("--------- " + methodType +  ", maxNumAttribute is " + maxNumAttribute + " ----------");
//		//1.定义评估函数 搜索策略
//		CHIAttributeEval chiAttributeEval = new CHIAttributeEval();
//		Ranker ranker = new Ranker();
//		ranker.setNumToSelect(maxNumAttribute);
//		//2.特征选择
//		AttributeSelection asFilter = new AttributeSelection();
//		asFilter.setEvaluator(chiAttributeEval);
//		asFilter.setSearch(ranker);
//		asFilter.setInputFormat(instances);
//		//3.特征选择生成新的实例集
//		Instances afterAsArrff = Filter.useFilter(instances, asFilter);
//		afterAsArrff.setClassIndex(afterAsArrff.numAttributes()-1);
//		WekaUtil.writeAttributeToFile("result/attributes_" + methodType + "_" + maxNumAttribute + ".txt",WekaUtil.getAttributesByINstances(afterAsArrff));
//		//4.分类器创建
//		NaiveBayesMultinomial nBayesM = new NaiveBayesMultinomial();
//		nBayesM.buildClassifier(afterAsArrff);
//		return nBayesM;
//	}
	/**
	 * 分类器 分类并评估
	 * CHI 特征选择  naivebayes 分类器
	 * @param instances
	 * @param maxNumAttribute
	 * @throws Exception
	 */
	public static void catagoryByCHINaiveBayes(Instances instances, int maxNumAttribute) throws Exception {
		String methodType = "CHI_特征选择方法_Naivebayes分类方法";
		CommonUtils.print("--------- " + methodType + ", maxNumAttribute is " + maxNumAttribute + " ----------");
		// 1.定义评估函数 搜索策略
		CHIAttributeEval chiAttributeEval = new CHIAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		// 2.特征选择
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(chiAttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(instances);
		// 3.特征选择生成新的实例集
		Instances afterAsArrff = Filter.useFilter(instances, asFilter);
		afterAsArrff.setClassIndex(afterAsArrff.numAttributes()-1);
		// 4.分类器创建
		NaiveBayes nBayes = new NaiveBayes();
		// 5.对特征选择后的新实例集交叉验证
		Evaluation evaluation = new Evaluation(afterAsArrff);
		evaluation.crossValidateModel(nBayes, afterAsArrff, 10, new Random(1));
		// 6.打印并保存评估结果
		printEvaluation(evaluation, "result/" + methodType + "_" + maxNumAttribute + ".txt");
	}
	/**
	 * 分类器 分类并评估
	 * CHI 特征选择 libsvm分类器 
	 * 同样采用weka 的元分类器
	 * @param instances
	 * @param maxNumAttribute
	 * @throws Exception
	 */
	public static void catagoryByCHILibSVM(Instances instances,int maxNumAttribute) throws Exception{
		String methodType = "CHI_特征选择方法_libsvm分类方法";
		CommonUtils.print("--------- " + methodType +  ", maxNumAttribute is " + maxNumAttribute + " ----------");
		// 1.定义评估函数 搜索策略
		CHIAttributeEval chiAttributeEval = new CHIAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		// 2.特征选择
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(chiAttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(instances);
		// 3.特征选择生成新的实例集
		Instances afterAsArrff = Filter.useFilter(instances, asFilter);
		afterAsArrff.setClassIndex(afterAsArrff.numAttributes()-1);
		// 4.分类器创建
		LibSVM libsvm = new LibSVM();
		libsvm.setOptions(new String[]{
				"-S","1"
		});//该参数很重要 
			// 5.对特征选择后的新实例集交叉验证
		Evaluation evaluation = new Evaluation(afterAsArrff);
		evaluation.crossValidateModel(libsvm, afterAsArrff, 10, new Random(1));
		// 6.打印并保存评估结果
		printEvaluation(evaluation, "result/" + methodType + "_" + maxNumAttribute + ".txt");
	}
	/**
	 * 分类并评估
	 * SMO 分类效果快weka自带的SVM分类器 
	 * 4折交叉验证
	 * @param instances
	 * @param maxNumAttribute
	 * @throws Exception
	 */
	public static void catagoryByCHISMO(Instances instances,int maxNumAttribute) throws Exception{
		String methodType = "CHI_特征选择方法_svm_SMO分类方法";
		CommonUtils.print("--------- " + methodType +  ", maxNumAttribute is " + maxNumAttribute + " ----------");
		// 1.定义评估函数 搜索策略
		CHIAttributeEval chiAttributeEval = new CHIAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		// 2.特征选择
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(chiAttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(instances);
		// 3.特征选择生成新的实例集
		Instances afterAsArrff = Filter.useFilter(instances, asFilter);
		afterAsArrff.setClassIndex(afterAsArrff.numAttributes()-1);
		// 4.分类器创建
		SMO smo = new SMO();
		CommonUtils.print(smo.toleranceParameterTipText());
			// 5.对特征选择后的新实例集交叉验证
		Evaluation evaluation = new Evaluation(afterAsArrff);
		evaluation.crossValidateModel(smo, afterAsArrff, 4, new Random(1));
		// 6.打印并保存评估结果
		printEvaluation(evaluation, "result/" + methodType + "_" + maxNumAttribute + ".txt");
	}
	/**
	 * 分类器  分类并评估 
	 * MI特征选择 多项式朴素贝叶斯模型 
	 * @param instances
	 * @param maxNumAttribute
	 * @throws Exception
	 */
	public static void catagoryByMINaiveBayesMultinomial(Instances instances,int maxNumAttribute) throws Exception{
		String methodType = "MI_特征选择方法_NaivebayesMultinomial分类方法";
		CommonUtils.print("--------- " + methodType +  ", maxNumAttribute is " + maxNumAttribute + " ----------");
		// 1.定义评估函数 搜索策略
//		InfoGainAttributeEval igAttributeEval = new InfoGainAttributeEval();
//		CHIAttributeEval chiAttributeEval = new CHIAttributeEval();
		MIAttributeEval miAttributeEval = new MIAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		// 2.特征选择
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(miAttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(instances);
		// 3.特征选择生成新的实例集
		Instances afterAsArrff = Filter.useFilter(instances, asFilter);
		afterAsArrff.setClassIndex(afterAsArrff.numAttributes()-1);
//		List<Attribute> attributes = WekaUtil.getAttributesByINstances(afterAsArrff);
//		WekaUtil.writeAttributeToFile("result/attributes_" + methodType + "_" + maxNumAttribute + ".txt", attributes);
		// 4.分类器创建
		NaiveBayesMultinomial nBayesM = new NaiveBayesMultinomial();
		// 5.对特征选择后的新实例集交叉验证
		Evaluation evaluation = new Evaluation(afterAsArrff);
		evaluation.crossValidateModel(nBayesM, afterAsArrff, 10, new Random(1));
		// 6.打印并保存评估结果
		printEvaluation(evaluation, "result/" + methodType + "_" + maxNumAttribute + ".txt");
	}
	
	/**
	 * 分类器  分类并十折交叉验证 
	 * 互信息特征选择 朴素贝叶斯模型分类器
	 * 
	 * @param instances
	 * @throws Exception
	 */
	public static void catagoryByMINavidateBayes(Instances instances,int maxNumAttribute) throws Exception{
//		InfoGainAttributeEval igAttributeEval = new InfoGainAttributeEval();
//		int maxAttributeNum = 5000;
		String methodType = "MI_特征选择方法_naivebayes分类方法";
		CommonUtils.print("--------- " + methodType +  ", maxNumAttribute is " + maxNumAttribute + " ----------");
		
		System.out.println("classfier method:naivebayes ;attribute selection:MI;attributeNum:"+maxNumAttribute);
		// 1.定义评估函数 搜索策略
		MIAttributeEval miAttributeEval = new MIAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		// 2.特征选择
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(miAttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(instances);
		// 3.特征选择生成新的实例集
		Instances afterAsArrff = Filter.useFilter(instances, asFilter);
		afterAsArrff.setClassIndex(afterAsArrff.numAttributes()-1);
		// 4.分类器创建
		NaiveBayes nBayes = new NaiveBayes();
		// 5.对特征选择后的新实例集交叉验证
		Evaluation evaluation = new Evaluation(afterAsArrff);
		evaluation.crossValidateModel(nBayes, afterAsArrff, 10, new Random(1));
		// 6.打印并保存评估结果
		printEvaluation(evaluation, "result/" + methodType + "_" + maxNumAttribute + ".txt");
	}
	/**
	 * 分类器  分类并评估 10折交叉验证
	 * MI特征选择 libsvm分类器
	 * @param instances
	 * @param maxNumAttribute
	 * @throws Exception
	 */
	public static void catagoryByMILibSVM(Instances instances,int maxNumAttribute) throws Exception{
		String methodType = "MI_特征选择方法_libsvm分类方法";
		CommonUtils.print("--------- " + methodType +  ", maxNumAttribute is " + maxNumAttribute + " ----------");
		CommonUtils.print("classfier method:libsvm ;attribute selection:MI;attributeNum:"+maxNumAttribute);
		//1.创建 初始化特征选择评估函数 搜索策略
		MIAttributeEval miAttributeEval = new MIAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		//2.创建特征选择过滤器 初始化
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(miAttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(instances);
		//3.特征选择 生成新实例集
		Instances afterAsArrff = Filter.useFilter(instances, asFilter);
		afterAsArrff.setClassIndex(afterAsArrff.numAttributes()-1);
		//4.创建分类器
		LibSVM libsvm = new LibSVM();
		libsvm.setOptions(new String[]{
				"-S","1"
		});
		//5.分类 并评估
		Evaluation evaluation = new Evaluation(instances);
		evaluation.crossValidateModel(libsvm, afterAsArrff, 10, new Random(1));
		//6.打印并保存
		printEvaluation(evaluation,"result/" + methodType + "_" + maxNumAttribute + ".txt");
	}
	/**
	 * 测试集 训练集 独立分类
	 * 验证测试集
	 * 批量过滤
	 * 根据训练集 验证测试集
	 * @param instances
	 * @param testInstances
	 * @param maxNumAttribute
	 * @throws Exception
	 */
	public static void testsetEvalByIGNaiveBayesMultinomial(Instances instances,Instances testInstances,int maxNumAttribute) throws Exception{
//		int maxAttributeNum = 5000;
		String methodType = "IG_特征选择方法_naivebayesMultinomial分类方法_批量过滤处理";
		CommonUtils.print("--------- " + methodType +  ", maxNumAttribute is " + maxNumAttribute + " ----------");
		System.out.println("classfier method:naivebayes ;attribute selection:IG;attributeNum:"+maxNumAttribute);
		//1.特征选择 评估函数和搜索策略
		InfoGainAttributeEval igAttributeEval = new InfoGainAttributeEval();
		igAttributeEval.setBinarizeNumericAttributes(true);
//		MIAttributeEval miAttributeEval = new MIAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		//2.进行特征选择生成 训练集 测试集
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(igAttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(instances);
		Instances afterTrainAsArrff = Filter.useFilter(instances, asFilter);
		Instances afterTestAsArff = Filter.useFilter(testInstances, asFilter);
		//3.初始化分类器
//		NaiveBayes nBayes = new NaiveBayes();
		NaiveBayesMultinomial nBayesM = new NaiveBayesMultinomial();
		nBayesM.buildClassifier(afterTrainAsArrff);//
		//4.评估
		Evaluation evaluation = new Evaluation(afterTrainAsArrff);
	    evaluation.evaluateModel(nBayesM, afterTestAsArff);
	    //5.评估 模型分类效果
		printEvaluation(evaluation,"result/" + methodType + "_" + maxNumAttribute + ".txt");
	}
	/**
	 * 
	 * @param instances
	 * @param testInstances
	 * @param maxNumAttribute
	 * @throws Exception
	 */
	public static void testsetEvalByCHINaiveBayesMultinomial(Instances instances,Instances testInstances,int maxNumAttribute) throws Exception{
//		int maxAttributeNum = 5000;
		String methodType = "CHI_特征选择方法_naivebayesNultinomial分类方法_批量过滤处理";
		CommonUtils.print("--------- " + methodType +  ", maxNumAttribute is " + maxNumAttribute + " ----------");
		CommonUtils.print("classfier method:naivebayesM ;attribute selection:CHI;attributeNum:"+maxNumAttribute);
		//1.初始化特征选择方法  评估函数和搜索策略
		CHIAttributeEval chiAttributeEval = new CHIAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		//2.初始化 特征选择过滤器 批量过滤 训练姐 测试集 以训练集一致为准
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(chiAttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(instances);
		//3.生成 新的测试集 训练集
		Instances afterTrainAsArrff = Filter.useFilter(instances, asFilter);
		Instances afterTestAsArff = Filter.useFilter(testInstances, asFilter);
		//4.分类评估
		NaiveBayesMultinomial nBayesM = new NaiveBayesMultinomial();
		nBayesM.buildClassifier(afterTrainAsArrff);//
		Evaluation evaluation = new Evaluation(afterTrainAsArrff);
	    evaluation.evaluateModel(nBayesM, afterTestAsArff);
	    //5.打印并保存评估结果
		printEvaluation(evaluation,"result/" + methodType + "_" + maxNumAttribute + ".txt");
	}
	/**
	 * 打印出相关 过滤器的参数列表
	 * @param filter
	 */
	public static void pintFilterOption(StringToWordVector filter){
		String options[] = filter.getOptions();
		String filterStr = filter.toString();
		String printStr = filterStr+"\n";
		for(String o:options){
			printStr += o + " ";
		}
		System.out.println(printStr);
	}
	/**
	 * 把属性列表写入文件
	 * @param fileName
	 * @param attributes
	 * @throws IOException
	 */
	public static void writeAttributeToFile(String fileName,List<Attribute> attributes) throws IOException{
		CommonUtils.print("------- write attributes list to files:" + fileName + "  ---------------");
		if(attributes!=null&&attributes.size()>0){
			String content = "";
			for(Attribute a:attributes){
				content += a.name()+"\n";
			}
			CommonUtils.writeFile(fileName, content );
		}
	}
	/**
	 * 打印评估器 函数
	 * @param evaluation
	 * @param fileName
	 * @throws Exception
	 */
	public static void printEvaluation(Evaluation evaluation,String fileName) throws Exception{
		 //
	    String content = "";
	    content += evaluation.toSummaryString() + "\n";
	    content += evaluation.toClassDetailsString() + "\n";
	    content += evaluation.toMatrixString() + "\n";
	    CommonUtils.print(content);
	    if(fileName!=null)
	    	CommonUtils.writeFile(fileName, content);

	}
	/**
	 * 根据数据集的属性列表 返回名称列表
	 * @param attributes
	 * @return
	 */
	public static List<String> attributesToNames(List<Attribute> attributes){
		List<String> names = new ArrayList<String>();
		if(!CommonUtils.listIsEmpty(attributes)){
			for(Attribute a:attributes){
				names.add(a.name());
			}
		}
		return names;
	}
	/**
	 * 根据相关度列表 替换文件 并保存
	 * @param oldfilename
	 * @param newfilename
	 * @param relationlists
	 */
	public static void exchangeDataInstancesByRelationwordList(String oldfilename,String newfilename,List<String> relationlists){
		CommonUtils.print("根据相关列表  修改源文件");
		int extendNum = 0;
		Map<String,String> relationlistMap = new HashMap<String,String>();
		List<String> atributenameList = new ArrayList<String>();
		if(CommonUtils.listIsEmpty(relationlists))
			return ;
		for(String a:relationlists){
			String []arr = a.split(" ");
			String key = arr[0];
			String relations = a.substring(a.indexOf(key)+key.length()+1,a.length());
			//
			atributenameList.add(key);
			if(!CommonUtils.stringIsEmpty(relations))
				relationlistMap.put(key, relations);
		}
		//开始扩展
		List<String> lines = CommonUtils.readFileLines(oldfilename);
		List<String> newlines = new ArrayList<String>();
		
		if(!CommonUtils.listIsEmpty(lines)){
			for(String line:lines){
				if(line.startsWith("'")){
					String text = line.split(",")[0];
					String preText = text;
					String catagory = line.split(",")[1];
					for(String key:atributenameList){
						if(!CommonUtils.stringIsEmpty(relationlistMap.get(key))){
							String relationArr[] = relationlistMap.get(key).split(" ");
							for(String str:relationArr){
								if(!"".equals(str)&&text.split(" ").length<maxTextCopyNum){
									if(WekaUtil.checkTextContainStr(text, key)){
//										System.out.println(text);
										text = WekaUtil.copySrtCombinToText(text, key, str);
//										System.out.println(text);
									}
								}
							}
						}
					}
					if(!preText.equals(text)){
//						System.out.println(preText+";"+catagory);
//						System.out.println(text+";"+catagory);
						extendNum++;
					}
					newlines.add(text + "," + catagory);
				}else{
					newlines.add(line);
				}
			}
		}
		CommonUtils.print("扩展文本数目："+extendNum);
		CommonUtils.writeFileByList(newfilename,newlines);
	}
	/**
	 * 根据相似度列表 重写数据文件 并保存
	 * @param oldfilename
	 * @param newfilename
	 * @param similaritylists
	 */
	public static void exchangeDataFileBySimilaritylists(String oldfilename,String newfilename,List<String> similaritylists){
		System.out.println("根据相似度列表 修改源文件");
		Map<String,String> similaritylistMap = new HashMap<String,String>();
		List<String> attribueList = new ArrayList<String>();
		if(CommonUtils.listIsEmpty(similaritylists))
			return ;
		for(String a:similaritylists){
			String[]arr = a.split(" ");
			String key = arr[0];
			String similaritys = a.substring(a.indexOf(key)+key.length()+1,a.length()); 
			//
			attribueList.add(key);
			similaritylistMap.put(key, similaritys);
		}
		//开始替换
		List<String> lines = CommonUtils.readFileLines(oldfilename);
		List<String> newlines = new ArrayList<String>();
		if(!CommonUtils.listIsEmpty(lines)){
			for(String line:lines){
//				String text = line;
				if(line.startsWith("'")){
					String text = line.split(",")[0];
					String preText = text;
					String catagory = line.split(",")[1];
					for(String key:attribueList){
						String similarityArr[] = similaritylistMap.get(key).split(" ");
						for(String str:similarityArr){
							if(!"".equals(str)&&(!str.equals(key)))
								text = copySrtReplaceToText(text,str,key);
//								text = text.replace(str, key);
						}
					}
					if(!preText.equals(text)){
//						System.out.println(preText);
//						System.out.println(text);
					}
					newlines.add(text + "," + catagory);
				}else{
					newlines.add(line);
				}
			}
		}
		//
//		StringBuffer content = new StringBuffer();
//		for(String l:newlines){
//			content.append(l+"\r\n");
//		}
//		CommonUtils.writeFile(newfilename, content.toString());
		CommonUtils.writeFileByList(newfilename,newlines);
//		Common
	}
	/**
	 * 检查 text的词语列表中 检查是否包含 词语 str
	 * @param text
	 * @param str
	 * @return
	 */
	public static boolean checkTextContainStr(String text,String str){
		String arr[] = text.split(" ");
		for(int i=0;i<arr.length;i++){
			if(arr[i].equals(str))
				return true;
		}
		return false;
	}
	/**
	 * 扩展文本 从text中检查 str 把str替换为 str addStr 仅仅替换一次 
	 * @param text
	 * @param str
	 * @param addStr
	 * @return
	 */
	public static String copySrtCombinToText(String text,String str,String addStr){
		//如果长度太长不扩展
		String newText = "";
		String arr[] = text.split(" ");
		int count = 0;//只扩展一次 即如果一个词在文本中出现多次 仅仅扩展一次，不会对出现多少次替换多少次
		for(int i=0;i<arr.length;i++){
			if(arr[i].equals(str)&&(!CommonUtils.stringIsEmpty(str))&&count==0){
				arr[i] = str + " " + addStr;
				count++;
//				break;
			}
			newText+=arr[i]+" ";
		}
		return newText;
	}
	/**
	 * 相似度计算中 替换近义词
	 * @param text
	 * @param oldStr
	 * @param newStr
	 * @return
	 */
	public static String copySrtReplaceToText(String text,String oldStr,String newStr){
		String newText = "";
		String arr[] = text.split(" ");
		for(int i=0;i<arr.length;i++){
			if(arr[i].equals(oldStr)){
				arr[i] = newStr;
			}
			newText+=arr[i]+" ";
		}
		return newText;
	}
	/**
	 * 过滤掉arff文件过短的记录
	 * @param oldfilename
	 * @param newfilename
	 */
	public static void preProcessARFF(String oldfilename,String newfilename,String shortfilename){
		List<String> lines = CommonUtils.readFileLines(oldfilename);
		List<String> newlines = new ArrayList<String>();
		List<String> shortlines = new ArrayList<String>();//把短文本收集起来测试用
		if(!CommonUtils.listIsEmpty(lines)){
			for(String line:lines){
				if(line.startsWith("'")){
					String text = line.split(",")[0];
//					if(text.split(" ").length<minAttributeLen){
					if(CommonUtils.textlen(text)<minAttributeLen){
						shortlines.add(line);
						continue;
					}
				}else{
					shortlines.add(line);
				}
				newlines.add(line);
			}
		}
		
		CommonUtils.writeFileByList(newfilename, newlines);
		if(shortfilename!=null)
			CommonUtils.writeFileByList(shortfilename,shortlines );
		CommonUtils.print("-------- 筛选掉："+(lines.size()-newlines.size())+"条记录 ----------");
	}
	/**
	 * 离散化的测试
	 * @param data
	 * @throws Exception
	 */
	public static void discretizeInstaces(Instances data) throws Exception {
		Discretize disTransform = new Discretize();
		disTransform.setUseBetterEncoding(true);
		disTransform.setInputFormat(data);
		data = Filter.useFilter(data, disTransform);
		WekaUtil.saveArffByInstancesByDataSink(data, "wekafiles/discretize_texts.arff");
	}
}
