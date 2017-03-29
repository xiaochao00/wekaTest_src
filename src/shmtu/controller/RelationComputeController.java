package shmtu.controller;

import java.util.List;

import shmtu.wekautils.SimilarityUtils;
import shmtu.wekautils.WekaAttributeEvalUtil;
import shmtu.wekautils.WekaTrainAndTestUtil;
import shmtu.wekautils.WekaUtil;
import shmtu.wekautils.WordRelationUtil;
import weka.core.Instances;

/**
 * 相关度 计算
 * @author HP_xiaochao
 *2017年1月1日
 *
 */
public class RelationComputeController {
	//实验目的：
	//扩展短文本 期望达到，对短文本有效的分类效果，最终实验结果不明显
	//实验步骤：
	//对原语料 测试评估，并全部扩展源文本，发现效果不好
	//之后 抽取出短文本 将短文本最为测试集 其他作为训练集
	//首先 训练集训练模型，测试 测试集得到实验结果
	//然后 对训练集计算关联度 并仅仅扩展短文本
	//同样利用 原来训练集模型 对扩展后的短文本测试集 测试，得到实验结果
	//对比 两次短文本实验结果
	//试验1 对比扩展源文本
	public static void doRelation() throws Exception{
		String oldfilename = "wekafiles/texts1926_pre.arff";
		//1.评估源数据集分类效果
		loadAndEval(oldfilename,1500);
		//2.相似度替换
		String newFilename = "wekafiles/relationText.arff";
		relationInstance(oldfilename,newFilename);
		//3.评估 替换后分类效果
		loadAndEval(newFilename,1500);
	}
	public static void loadAndEval(String filename,int maxnum) throws Exception {
		// 1.加载text文档
		Instances instances = WekaUtil.loadArffByDataSource(filename);
		WekaUtil.printHeader(instances);
		instances.setClassIndex(instances.numAttributes()-1);
		//2.词向量
		Integer numWordsToKeey = 3000;
		Instances weightDate = WekaUtil.stringToVerctorWeightIF(instances,numWordsToKeey);
		weightDate.setClassIndex(0);//设置类别索引
		//3.分类评估
//		WekaUtil.catagoryByIGLibSVM(weightDate, maxnum);
		WekaUtil.catagoryByCHINaiveBayesMultinomial(weightDate, maxnum);
	}
	public static void relationInstance(String oldFilename,String newFilename)throws Exception{
		// 1.加载数据
//		String arffFilePath = "wekafiles/texts.arff";
		Instances data = WekaUtil.loadArffByDataSource(oldFilename);
		data.setClassIndex(data.numAttributes() - 1);
		WekaUtil.printHeader(data);
		// 2.词向量转换
		Integer numWordsToKeey = 1000;
		Instances dataStringToVector = WekaUtil.stringToVectorWeightBoolean(data,numWordsToKeey);
		dataStringToVector.setClassIndex(0);
		// 3.特征选择
		int maxNumAttribute = 1000;
		Instances dataAs = WekaAttributeEvalUtil.igEvaluationAndReturn(dataStringToVector, maxNumAttribute);
		// 4.计算 把两个词的组合作为一个特征
		List<String> relationList = WordRelationUtil.computeWordCombineListByIG(dataAs);
		// 5.重写文件
		WekaUtil.exchangeDataInstancesByRelationwordList(oldFilename,newFilename,relationList);
//				"wekafiles/texts.arff", "wekafiles/texts.arff", relationList);
	}
	//试验二 对比扩展 短文本 训练集 测试集
	public static void testsetRelationExtral() throws Exception{
//		String trainfile = "wekafiles/texts1926_pre.arff";
		String trainfile = "wekafiles/texts10w_pre.arff";
		
//		String testfile = "wekafiles/texts1926_shortText.arff";
		String testfile = "wekafiles/texts10w_shortText.arff";
		//String trainnewfile = "wekafiles/texts_pre_relation.arff";
//		String testnewfile = "wekafiles/shortText_relation.arff";
//		String testnewfile = "wekafiles/texts1926_shortText_relation.arff";
		String testnewfile = "wekafiles/texts10w_shortText_relation.arff";
		//1.先训练 未扩展的分类结果
//		WekaTrainAndTestUtil.evalTrainTestByIGNaiveBayesMultinomial(trainfile, testfile, 1000);
		WekaTrainAndTestUtil.evalTrainTestByCHINaiveBayesMultinomial(trainfile, testfile, 2000);
		
		// 2.计算 把两个词的组合作为一个特征 计算每个属性的相关相关
		List<String> relationList = WordRelationUtil.relationCompute(trainfile, 1000);
		//3.扩展测试文件，暂不需要扩展原数据文件
		WekaUtil.exchangeDataInstancesByRelationwordList(testfile, testnewfile, relationList);
//		4.再次训练 并测试 得到扩展的分类结果
//		WekaTrainAndTestUtil.evalTrainTestByIGNaiveBayesMultinomial(trainfile, testnewfile, 1000);
		WekaTrainAndTestUtil.evalTrainTestByCHINaiveBayesMultinomial(trainfile, testnewfile, 2000);
	}
	/**
	 * 处理语料集 成 短语料集和长语料
	 */
	public static void processShortData(){
//		String oldfilename = "wekafiles/texts1926.arff";
//		String newfilename = "wekafiles/texts1926_pre.arff";
//		String shortFilename = "wekafiles/texts1926_shortText.arff";
		String oldfilename = "wekafiles/texts10w.arff";
		String newfilename = "wekafiles/texts10w_pre.arff";
		String shortFilename = "wekafiles/texts10w_shortText.arff";
		WekaUtil.preProcessARFF(oldfilename, newfilename,shortFilename);
	}
	public static void removeShortText(){
		String oldfile = "wekafiles/texts10w_shortText.arff";
		String newfile = "wekafiles/texts10w_shortText_pre.arff";
		WekaUtil.preProcessARFF(oldfile, newfile,null);
	}
	public static void similarityAndRelation() throws Exception{
		//对短文本 近义词替换 然后关联词扩展
		String trainfile = "wekafiles/texts10w_pre.arff";
		
//		String testfile = "wekafiles/texts1926_shortText.arff";
		String testfile = "wekafiles/texts10w_shortText.arff";
		//String trainnewfile = "wekafiles/texts_pre_relation.arff";
//		String testnewfile = "wekafiles/shortText_relation.arff";
//		String testnewfile = "wekafiles/texts1926_shortText_relation.arff";
		String testSimilarityNewfile = "wekafiles/texts10w_shortText_similarity.arff";
		String testSimilarityRelationnewfile = "wekafiles/texts10w_shortText_similarity_relation.arff";
		//1.先训练 未扩展的分类结果
//		WekaTrainAndTestUtil.evalTrainTestByIGNaiveBayesMultinomial(trainfile, testfile, 1000);
		WekaTrainAndTestUtil.evalTrainTestByCHINaiveBayesMultinomial(trainfile, testfile, 2000);
		
		// 2.计算 把两个词的组合作为一个特征 计算每个属性的相关相关
		List<String> relationList = WordRelationUtil.relationCompute(trainfile, 500);
		// 3.近义词计算
		SimilarityUtils.similarityTrainAndTest(trainfile,testfile,testSimilarityNewfile,2000);
		//
		//3.扩展测试文件，暂不需要扩展原数据文件
		WekaUtil.exchangeDataInstancesByRelationwordList(testSimilarityNewfile, testSimilarityRelationnewfile, relationList);
//		4.再次训练 并测试 得到扩展的分类结果
//		WekaTrainAndTestUtil.evalTrainTestByIGNaiveBayesMultinomial(trainfile, testnewfile, 1000);
		WekaTrainAndTestUtil.evalTrainTestByCHINaiveBayesMultinomial(trainfile, testSimilarityRelationnewfile, 2000);
		
	}
	public static void main(String[] args) throws Exception {
//		processShortData();
		testsetRelationExtral();
//		similarityAndRelation();
//		removeShortText();
	}
}
