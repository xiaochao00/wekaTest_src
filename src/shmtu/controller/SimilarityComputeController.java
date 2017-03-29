package shmtu.controller;

import shmtu.wekautils.SimilarityUtils;
import shmtu.wekautils.WekaUtil;
import weka.core.Instances;

/**
 * 词语相关度计算
 * @author HP_xiaochao
 *2016年12月31日
 *
 */
public class SimilarityComputeController {
	//试验目的：
	//通过特征选择 选取特征，计算特征之间的相似度，根据Hownet和同义词词林，并且加入词性过滤，去重筛选
	//试验 得到 近义词表，通过近义词表 替换置换源语料，进行分类评估，期望减少特征获得更好的分类效果
	// 实验效果不好。并没有得到预期的结果。
	//由于 近义词混淆了类别信息，文本表示采用TFIDF，可以从前后的特征选择中对比特征
	//文本表示采用TFIDF权重，恰好同义词替换 增大了DF，降低了权重，可以采用TF试验，或者改进的TFCHI等实验。词频也不一定
	//鉴于此 首先实现替换和做比对，
	//然后对比前后特征
	//最后 通过TF权重做实验，观察结果是否提高
	//基本实验步骤：
	//首先评价源语料 分类评估
	//其次替换
	//最后 评价新的语料
	public static void doSimilarity() throws Exception{
		String oldfilename = "wekafiles/texts1926.arff";
//		String oldfilename = "wekafiles/wb10w-hasni.arff";
//		String oldfilename = "wekafiles/THUCNews.arff";
		
		
		int numAttribute = 1100;
		int reduceNumAttribute = 1000;
		//1.评估源数据集分类效果
//		loadAndEval(oldfilename,numAttribute);
		loadAndEval(oldfilename,reduceNumAttribute);
		//2.相似度替换
		String newFilename = "wekafiles/texts1926_hownet_similarity.arff";
//		String newFilename = "wekafiles/THUCNews_similarity.arff.arff";
//		String newFilename = "wekafiles/similarity_wb10w-hasni.arff";
//		String newFilename = "wekafiles/similarity_wb10w_14C.arff";
		SimilarityUtils.similarityInstanceByHownetAndCilin(oldfilename,newFilename);
		//3.评估 替换后分类效果
		loadAndEval(newFilename,reduceNumAttribute);
	}
	/**
	 * 对比 添加同义词词林效果前后对比
	 * @throws Exception
	 */
	public static void doSimilarity3() throws Exception{
		String oldfilename = "wekafiles/texts1926.arff";
//		String oldfilename = "wekafiles/wb10w-hasni.arff";
//		String oldfilename = "wekafiles/THUCNews.arff";
		
		
		int numAttribute = 1100;
		int reduceNumAttribute = 1000;
		//1.评估源数据集分类效果
//		loadAndEval(oldfilename,numAttribute);
		loadAndEval(oldfilename,reduceNumAttribute);
		//2.相似度替换
		String newFilename = "wekafiles/texts1926_hownet_similarity.arff";
		String newFilename2 = "wekafiles/texts1926_hownet_cilin_similarity.arff";
//		String newFilename = "wekafiles/THUCNews_similarity.arff.arff";
//		String newFilename = "wekafiles/similarity_wb10w-hasni.arff";
//		String newFilename = "wekafiles/similarity_wb10w_14C.arff";
		SimilarityUtils.similarityInstanceByHownet(oldfilename,newFilename);
		SimilarityUtils.similarityInstanceByHownetAndCilin(oldfilename,newFilename2);
		//3.评估 替换后分类效果
		loadAndEval(newFilename,reduceNumAttribute);
		loadAndEval(newFilename2,reduceNumAttribute);
	}
	public static void loadAndEval(String filename,int maxnum) throws Exception {
		// 1.加载text文档
		Instances instances = WekaUtil.loadArffByDataSource(filename);
		WekaUtil.printHeader(instances);
		instances.setClassIndex(instances.numAttributes()-1);
		//2.词向量
		Integer numWordsToKeey = 1000;
//		Instances weightDate = WekaUtil.stringToVerctorWeightIFIDF(instances,numWordsToKeey);
		Instances weightDate = WekaUtil.stringToVerctorWeightIF(instances,numWordsToKeey);
//		Instances weightDate = WekaUtil.stringToVectorWeightBoolean(instances,numWordsToKeey);/
		weightDate.setClassIndex(0);//设置类别索引
		//3.分类评估
//		WekaUtil.catagoryByIGLibSVM(weightDate, maxnum);
		WekaUtil.catagoryByCHINaiveBayesMultinomial(weightDate, maxnum);
	}
	/**
	 * 指定相似度生成 相关的相似数据集，
	 * 根据不同的计算特征数目 评估新语料集
	 * 阈值 0.8 0.9
	 * @throws Exception 
	 */
	public static void doSimility2() throws Exception{
		//
		int numComputeAttributes = 6000;
		String oldfilename = "wekafiles/texts1926.arff";
		String newFilename = "wekafiles/similarityText1926_num" + numComputeAttributes + ".arff";
		SimilarityUtils.similarityInstance2(oldfilename,newFilename,numComputeAttributes);
	}
	public static void main(String[] args) throws Exception {
//		doSimility2();
		doSimilarity();
//		doSimilarity3();
	}
}
