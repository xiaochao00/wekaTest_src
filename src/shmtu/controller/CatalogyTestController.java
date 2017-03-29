package shmtu.controller;

import java.util.Date;

import shmtu.util.CommonUtils;
import shmtu.wekautils.WekaUtil;
import weka.core.Instances;

/**
 * 分类测试试验 
 * @author HP_xiaochao
 *2017年1月1日
 *
 */
public class CatalogyTestController {
	public static void doSomeTest() throws Exception{
		Date startTime = new Date();
//		String arffFilePath = "wekafiles/texts1926.arff";
//		String arffFilePath = "wekafiles/texts10w.arff";
//		String arffFilePath = "wekafiles/similarityText1926_num4000.arff";
		String arffFilePath = "wekafiles/THUCNews_similarity.arff.arff";
		
		
		//1.加载数据集
		Instances instances = WekaUtil.loadArffByDataSource(arffFilePath);
		instances.setClassIndex(instances.numAttributes()-1);
		//2.空间向量转换
		Integer numWordsToKeey = 1000;
//		Integer numWordsToKeey = 3000;//10w语料库按照1000 总共得到6千多特征，如果特征数目多6000需要改回来
//		Instances newArff = WekaUtil.stringToVerctorWeightIFIDF(instances,numWordsToKeey);
		Instances newArff = WekaUtil.stringToVerctorWeightIF(instances,numWordsToKeey);
		newArff.setClassIndex(0);//设置类别索引
		//3.指定分类方法测试 试验
//		WekaUtil.catagoryByIGNaiveBayesMultinomial(newArff, 2000);
//		WekaUtil.catagoryByIG2NaiveBayes(newArff, 3000);
//		WekaUtil.catagoryByCHISMO(newArff, 2000);
		WekaUtil.catagoryByCHINaiveBayesMultinomial(newArff, 1900);
		Date endTime = new Date();
		CommonUtils.print("用时:"+(endTime.getTime()-startTime.getTime()));
	}
	public static void main(String[]args) throws Exception{
		doSomeTest();
	}
}
