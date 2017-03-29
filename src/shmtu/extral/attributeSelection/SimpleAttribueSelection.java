package shmtu.extral.attributeSelection;

import java.util.Random;

import shmtu.util.CommonUtils;
import shmtu.wekautils.SimpleAttributeSelectionUtil;
import shmtu.wekautils.WekaAttributeEvalUtil;
import shmtu.wekautils.WekaUtil;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.core.Instances;

/**
 * 局部特征选择方法
 * @author HP_xiaochao
 *2017年3月11日
 *
 */
public class SimpleAttribueSelection {
	public static void main(String[] args) throws Exception {
		doSth();
	}
	
	public static void doSth() throws Exception{
		//1.加载语料
		String filepath = "wekafiles/texts1926.arff";
		Instances datas = WekaUtil.loadArffByDataSource(filepath);
		datas.setClassIndex(datas.numAttributes()-1);
		//2.词向量转换
		Instances datas_vsm = WekaUtil.stringToVerctorWeightIFIDF(datas, 1000);
		datas_vsm.setClassIndex(0);
		//3.特征选择1
		int maxNumAttribute = 2000;
		Instances simpleChooseData = SimpleAttributeSelectionUtil.simpleChooseByCHI(datas_vsm, maxNumAttribute);
		//4.分别评估
		
		WekaUtil.catagoryByCHINaiveBayesMultinomial(datas_vsm, maxNumAttribute);
		eval(simpleChooseData);
	}
	
	public static void eval(Instances data) throws Exception{
		String methodType = "multinomial";
		// 4.分类器创建
		NaiveBayesMultinomial nBayesM = new NaiveBayesMultinomial();
		// 5.对特征选择后的新实例集交叉验证
		Evaluation evaluation = new Evaluation(data);
		evaluation.crossValidateModel(nBayesM, data, 10, new Random(1));
		// 6.打印并保存评估结果
		printEvaluation(evaluation, "result/" + methodType  + ".txt");
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
	 * 1.计算特征 列别 关联矩阵
	 * 2.根据 数据集的比例 选取特征
	 * 
	 * 
	 * 
	 * 
	 */
}
