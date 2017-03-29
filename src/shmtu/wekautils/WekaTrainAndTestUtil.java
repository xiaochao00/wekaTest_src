package shmtu.wekautils;

import shmtu.util.CommonUtils;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToWordVector;
/**
 * weka 测试集 训练集分开 计算评估的工具类
 * 批量过滤
 * @author HP_xiaochao
 *2016年12月31日
 *
 */
public class WekaTrainAndTestUtil {
	
	/**
	 * 训练集训练模型 
	 * 测试集 根据 训练集的 空间向量转化 IG特征选择 贝叶斯多项式模型分类器 分类评估测试集
	 * 空间转换的 wordToKeep 没指定 默认1000
	 * @param trainfile
	 * @param testfile
	 * @param maxAttributeNum
	 * @throws Exception
	 */
	public static void evalTrainTestByIGNaiveBayesMultinomial(String trainfile,String testfile,int maxAttributeNum) throws Exception{
		String methodType = "Ig_naiveBayesMultinomial_train_test";
		CommonUtils.print("------------ "+methodType+" ---------------");
		Instances srcTrainInstances = WekaUtil.loadArffByDataSource(trainfile);
		Instances srcTestInstances = WekaUtil.loadArffByDataSource(testfile);
		srcTrainInstances.setClassIndex(srcTrainInstances.numAttributes()-1);
		srcTestInstances.setClassIndex(srcTestInstances.numAttributes()-1);
		//
		StringToWordVector vector = new StringToWordVector();
		vector.setTFTransform(true);
		vector.setIDFTransform(true);
		vector.setOutputWordCounts(true);
		vector.setInputFormat(srcTrainInstances);
		Instances vsmTrainInstances = Filter.useFilter(srcTrainInstances, vector);
		Instances vsmTestInstances = Filter.useFilter(srcTestInstances, vector);
		//
		vsmTrainInstances.setClassIndex(0);
		vsmTestInstances.setClassIndex(0);
		CommonUtils.print("训练集 属性数目："+vsmTrainInstances.numAttributes());
		CommonUtils.print("测试集 属性数目："+vsmTestInstances.numAttributes());
		//
		WekaUtil.testsetEvalByIGNaiveBayesMultinomial(vsmTrainInstances, vsmTestInstances, maxAttributeNum);
	}
	/**
	 * 训练集训练模型 
	 * 测试集 根据训练集 空间转换 CHI特征选择
	 * 过滤器 的实现类 都可以批量处理，指定setInputFormat 是标准
	 * 空间向量转换的 wordTokeep没指定 需要制定
	 * @param trainfile
	 * @param testfile
	 * @param maxAttributeNum
	 * @throws Exception
	 */
	public static void evalTrainTestByCHINaiveBayesMultinomial(String trainfile,String testfile,int maxAttributeNum) throws Exception{
		String methodType = "CHI_naiveBayesMultinomial_train_test";
		CommonUtils.print("------------ "+methodType+" ---------------");
		Instances srcTrainInstances = WekaUtil.loadArffByDataSource(trainfile);
		Instances srcTestInstances = WekaUtil.loadArffByDataSource(testfile);
		srcTrainInstances.setClassIndex(srcTrainInstances.numAttributes()-1);
		srcTestInstances.setClassIndex(srcTestInstances.numAttributes()-1);
		//
		StringToWordVector vector = new StringToWordVector();
		vector.setTFTransform(true);
		vector.setIDFTransform(true);
		vector.setOutputWordCounts(true);
		vector.setInputFormat(srcTrainInstances);
		Instances vsmTrainInstances = Filter.useFilter(srcTrainInstances, vector);
		Instances vsmTestInstances = Filter.useFilter(srcTestInstances, vector);
		//
		vsmTrainInstances.setClassIndex(0);
		vsmTestInstances.setClassIndex(0);
		CommonUtils.print("训练集 属性数目："+vsmTrainInstances.numAttributes());
		CommonUtils.print("测试集 属性数目："+vsmTestInstances.numAttributes());
		//
		WekaUtil.testsetEvalByCHINaiveBayesMultinomial(vsmTrainInstances, vsmTestInstances, maxAttributeNum);
	}
}
