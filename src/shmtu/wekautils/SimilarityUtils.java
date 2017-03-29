package shmtu.wekautils;

import java.util.List;

import shmtu.util.CommonUtils;
import shmtu.wordsimilarity.WordSimilarityUtil;
import weka.core.Instances;

public class SimilarityUtils {
	
	public static void similarityInstanceByHownetAndCilin(String oldFilename,String newFilename) throws Exception {
		// 1.加载text文档
		CommonUtils.print("similarity compute load arff");
		Instances instances = WekaUtil.loadArffByDataSource(oldFilename);
		WekaUtil.printHeader(instances);
		instances.setClassIndex(instances.numAttributes() - 1);
		// 2.词向量
		CommonUtils.print("similarity compute stringToWord");
		Integer numWordsToKeey = 1000;
		Instances weightDate = WekaUtil.stringToVerctorWeightIFIDF(instances,numWordsToKeey);
		weightDate.setClassIndex(0);// 设置类别索引
		//3.特征选择
		CommonUtils.print("similarity compute feature selection");
		List<String> attributenames = WekaAttributeEvalUtil.igEvaluation(weightDate,2000);
//		List<String> attributenames = WekaAttributeEvalUtil.chiEvaluation(weightDate,2000);
//		miEvaluation
//		WekaUtil.catagoryByIGLibSVM(newArff.c,);
		CommonUtils.print("similarity compute compute similaritylist");
		List<String> attribtuteAfSimilarity = WordSimilarityUtil.computeSimilarityWordListByHownetAndCilin(attributenames);
		//4.重写文件
		CommonUtils.print("similarity compute rewrite arff");
		WekaUtil.exchangeDataFileBySimilaritylists(oldFilename,newFilename,attribtuteAfSimilarity);
		//
//		Instances newData = WekaUtil.loadArffByDataSource(newFilename)
	}
	public static void similarityInstanceByHownet(String oldFilename,String newFilename) throws Exception {
		// 1.加载text文档
		CommonUtils.print("similarity compute load arff");
		Instances instances = WekaUtil.loadArffByDataSource(oldFilename);
		WekaUtil.printHeader(instances);
		instances.setClassIndex(instances.numAttributes() - 1);
		// 2.词向量
		CommonUtils.print("similarity compute stringToWord");
		Integer numWordsToKeey = 1000;
		Instances weightDate = WekaUtil.stringToVerctorWeightIFIDF(instances,numWordsToKeey);
		weightDate.setClassIndex(0);// 设置类别索引
		//3.特征选择
		CommonUtils.print("similarity compute feature selection");
//		List<String> attributenames = WekaAttributeEvalUtil.igEvaluation(weightDate,2000);
		List<String> attributenames = WekaAttributeEvalUtil.chiEvaluation(weightDate,2000);
//		miEvaluation
//		WekaUtil.catagoryByIGLibSVM(newArff.c,);
		CommonUtils.print("similarity compute compute similaritylist");
		List<String> attribtuteAfSimilarity = WordSimilarityUtil.computeSimilarityWordListByHownet(attributenames);
		//4.重写文件
		CommonUtils.print("similarity compute rewrite arff");
		WekaUtil.exchangeDataFileBySimilaritylists(oldFilename,newFilename,attribtuteAfSimilarity);
		//
//		Instances newData = WekaUtil.loadArffByDataSource(newFilename)
	}
	public static void similarityInstance2(String oldFilename,String newFilename,int numComputeAttributes) throws Exception {
		CommonUtils.print("similarity compute start compute attributes num: " + numComputeAttributes);
		// 1.加载text文档
		CommonUtils.print("similarity compute load arff");
		Instances instances = WekaUtil.loadArffByDataSource(oldFilename);
		WekaUtil.printHeader(instances);
		instances.setClassIndex(instances.numAttributes() - 1);
		// 2.词向量
		CommonUtils.print("similarity compute stringToWord");
		Integer numWordsToKeey = 1000;
		Instances weightDate = WekaUtil.stringToVerctorWeightIFIDF(instances,numWordsToKeey);
		weightDate.setClassIndex(0);// 设置类别索引
		//3.特征选择
		CommonUtils.print("similarity compute feature selection");
		List<String> attributenames = WekaAttributeEvalUtil.igEvaluation(weightDate,numComputeAttributes);
//		List<String> attributenames = WekaAttributeEvalUtil.miEvaluation(weightDate,2000);
//		miEvaluation
//		WekaUtil.catagoryByIGLibSVM(newArff.c,);
		CommonUtils.print("similarity compute compute similaritylist");
		List<String> attribtuteAfSimilarity = WordSimilarityUtil.computeSimilarityWordListByHownet(attributenames);
		//4.重写文件
		CommonUtils.print("similarity compute rewrite arff");
		WekaUtil.exchangeDataFileBySimilaritylists(oldFilename,newFilename,attribtuteAfSimilarity);
		//
//		Instances newData = WekaUtil.loadArffByDataSource(newFilename)
	}
	/**
	 * 根据训练集 计算相似列表 替换测试集
	 * @param trainfilename
	 * @param oldTestfilename
	 * @param newTestfilename
	 * @param numComputeAttributes
	 * @throws Exception 
	 */
	public static void similarityTrainAndTest(String trainfilename,String oldTestfilename,String newTestfilename,int numComputeAttributes) throws Exception{
		CommonUtils.print("similarity compute start compute attributes num: " + numComputeAttributes);
		// 1.加载text文档
		CommonUtils.print("similarity compute load arff");
		Instances instances = WekaUtil.loadArffByDataSource(trainfilename);
		WekaUtil.printHeader(instances);
		instances.setClassIndex(instances.numAttributes() - 1);
		// 2.词向量
		CommonUtils.print("similarity compute stringToWord");
		Integer numWordsToKeey = 1000;
		Instances weightDate = WekaUtil.stringToVerctorWeightIFIDF(instances,numWordsToKeey);
		weightDate.setClassIndex(0);// 设置类别索引
		//3.特征选择
		CommonUtils.print("similarity compute feature selection");
		List<String> attributenames = WekaAttributeEvalUtil.igEvaluation(weightDate,numComputeAttributes);
//		List<String> attributenames = WekaAttributeEvalUtil.miEvaluation(weightDate,2000);
//		miEvaluation
//		WekaUtil.catagoryByIGLibSVM(newArff.c,);
		CommonUtils.print("similarity compute compute similaritylist");
		List<String> attribtuteAfSimilarity = WordSimilarityUtil.computeSimilarityWordListByHownet(attributenames);
		//4.重写文件 测试文件
		CommonUtils.print("similarity compute rewrite arff");
		WekaUtil.exchangeDataFileBySimilaritylists(oldTestfilename,newTestfilename,attribtuteAfSimilarity);
		//
//		
	}
}
