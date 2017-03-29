package shmtu.wekautils;

import java.util.List;

import shmtu.extral.attributeSelection.CHIAttributeEval;
import shmtu.extral.attributeSelection.IGAttributeEval2;
import shmtu.extral.attributeSelection.MIAttributeEval;
import shmtu.util.CommonUtils;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.core.Attribute;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AttributeSelection;

/**
 * 特征选择验证
 * 在文本分类中 使用IG特征选择方法需要把IG的二值化设置为true，这样不会离散化 节约大部分时间
 * @author HP_xiaochao
 *
 */
public class WekaAttributeEvalUtil {
	/**
	 * 使用IG特征选择方法 并保存特征
	 * 使用IG特征选择方法需要 设置BinarizeNumeric值为真
	 * 这样不在离散化 
	 * @param data
	 * @throws Exception
	 */
	public static void igEval(Instances data,int maxNumAttribute) throws Exception {
		String methodAbstract = "attributes_selection_ig_" + maxNumAttribute;
		CommonUtils.print("--------------- " + methodAbstract + " ---------------");
		InfoGainAttributeEval igAttributeEval = new InfoGainAttributeEval();
		igAttributeEval.setBinarizeNumericAttributes(true);
//		CHIAttributeEval chiAttributeEval = new CHIAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(igAttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(data);
		Instances afterAsArrff = Filter.useFilter(data, asFilter);
		List<Attribute> afterAsAttributes = WekaUtil.getAttributesByINstances(afterAsArrff);
		WekaUtil.writeAttributeToFile("result/" + methodAbstract + ".txt", afterAsAttributes);
	}
	/**
	 * 使用根据wekaIG特征选择方法复制的 不带离散化的IG特征选择方法  并保存特征列表文件
	 * 实验效果和wekaIG带二值化参数效果一样
	 * @param data
	 * @param maxNumAttribute
	 * @throws Exception
	 */
	public static void ig2Eval(Instances data,int maxNumAttribute) throws Exception {
		String methodAbstract = "attributes_selection_ig2_" + maxNumAttribute;
		CommonUtils.print("--------------- " + methodAbstract + " ---------------");
		IGAttributeEval2 ig2AttributeEval = new IGAttributeEval2();
//		igAttributeEval.setBinarizeNumericAttributes(true);
//		CHIAttributeEval chiAttributeEval = new CHIAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(ig2AttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(data);
		Instances afterAsArrff = Filter.useFilter(data, asFilter);
		List<Attribute> afterAsAttributes = WekaUtil.getAttributesByINstances(afterAsArrff);
		WekaUtil.writeAttributeToFile("result/" + methodAbstract + ".txt", afterAsAttributes);
	}
	/**
	 * IG 特征选择并 返回选择的属性 并保存文件
	 * 接受数据集是 转换为空间向量后的
	 * @param data
	 * @param maxNumAttribute
	 * @return
	 * @throws Exception
	 */
	public static List<String> igEvaluation(Instances data,int maxNumAttribute) throws Exception {
		String methodAbstract = "attributes_selection_ig_" + maxNumAttribute;
		CommonUtils.print("--------------- " + methodAbstract + " ---------------");
		CommonUtils.print("class attribute : " + data.attribute(data.classIndex()));
		InfoGainAttributeEval igAttributeEval = new InfoGainAttributeEval();
		igAttributeEval.setBinarizeNumericAttributes(true);
//		CHIAttributeEval chiAttributeEval = new CHIAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(igAttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(data);
		Instances afterAsArrff = Filter.useFilter(data, asFilter);
		List<Attribute> afterAsAttributes = WekaUtil.getAttributesByINstances(afterAsArrff);
		WekaUtil.writeAttributeToFile("result/" + methodAbstract + ".txt", afterAsAttributes);
		List<String> attributesName = WekaUtil.attributesToNames(afterAsAttributes);
		return attributesName;
	}
	/**
	 * IG 特征选择方法 选择后属性名列表 并保存文件
	 * 接受数据集是 转换为空间向量后的
	 * @param data
	 * @param maxNumAttribute
	 * @return
	 * @throws Exception
	 */
	public static List<String> ig2Evaluation(Instances data,int maxNumAttribute) throws Exception {
		String methodAbstract = "attributes_selection_ig2_" + maxNumAttribute;
		CommonUtils.print("--------------- " + methodAbstract + " ---------------");
		CommonUtils.print("class attribute : " + data.attribute(data.classIndex()));
		IGAttributeEval2 ig2AttributeEval = new IGAttributeEval2();
//		CHIAttributeEval chiAttributeEval = new CHIAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(ig2AttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(data);
		Instances afterAsArrff = Filter.useFilter(data, asFilter);
		List<Attribute> afterAsAttributes = WekaUtil.getAttributesByINstances(afterAsArrff);
		WekaUtil.writeAttributeToFile("result/" + methodAbstract + ".txt", afterAsAttributes);
		List<String> attributesName = WekaUtil.attributesToNames(afterAsAttributes);
		return attributesName;
	}
	/**
	 * 互信息特征选择评估算法 返回 过滤后的属性列表
	 * @param data
	 * @param maxNumAttribute
	 * @return
	 * @throws Exception
	 */
	public static List<String> miEvaluation(Instances data,int maxNumAttribute) throws Exception {
		String methodAbstract = "attributes_selection_mi_" + maxNumAttribute;
		CommonUtils.print("--------------- " + methodAbstract + " ---------------");
		CommonUtils.print("class attribute : " + data.attribute(data.classIndex()));
		MIAttributeEval miAttributeEval = new MIAttributeEval();
//		CHIAttributeEval chiAttributeEval = new CHIAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(miAttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(data);
		Instances afterAsArrff = Filter.useFilter(data, asFilter);
		List<Attribute> afterAsAttributes = WekaUtil.getAttributesByINstances(afterAsArrff);
		WekaUtil.writeAttributeToFile("result/" + methodAbstract + ".txt", afterAsAttributes);
		List<String> attributesName = WekaUtil.attributesToNames(afterAsAttributes);
		return attributesName;
	}
	/**
	 * 卡方 评估特征选择 并保存文件
	 * @param data
	 * @param maxNumAttribute
	 * @return
	 * @throws Exception
	 */
	public static List<String> chiEvaluation(Instances data,int maxNumAttribute) throws Exception {
		String methodAbstract = "attributes_selection_chi_" + maxNumAttribute;
		CommonUtils.print("--------------- " + methodAbstract + " ---------------");
		CommonUtils.print("class attribute : " + data.attribute(data.classIndex()));
//		InfoGainAttributeEval igAttributeEval = new InfoGainAttributeEval();
		CHIAttributeEval chiAttributeEval = new CHIAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(chiAttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(data);
		Instances afterAsArrff = Filter.useFilter(data, asFilter);
		List<Attribute> afterAsAttributes = WekaUtil.getAttributesByINstances(afterAsArrff);
		WekaUtil.writeAttributeToFile("result/" + methodAbstract + ".txt", afterAsAttributes);
		List<String> attributesName = WekaUtil.attributesToNames(afterAsAttributes);
		return attributesName;
	}
	/**
	 * 卡方 特征选择 并返回实例
	 * @param data
	 * @param maxNumAttribute
	 * @return
	 * @throws Exception
	 */
	public static Instances chiEvaluationAndReturn(Instances data,int maxNumAttribute) throws Exception {
		String methodAbstract = "attributes_selection_chi_" + maxNumAttribute;
		CommonUtils.print("--------------- " + methodAbstract + " ---------------");
		CommonUtils.print("class attribute : " + data.attribute(data.classIndex()));
//		InfoGainAttributeEval igAttributeEval = new InfoGainAttributeEval();
		CHIAttributeEval chiAttributeEval = new CHIAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(chiAttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(data);
		Instances afterAsArrff = Filter.useFilter(data, asFilter);
		return afterAsArrff;
	}
	/**
	 * ig2 特征选择 并返回实例 
	 * @param data
	 * @param maxNumAttribute
	 * @return
	 * @throws Exception
	 */
	public static Instances ig2EvaluationAndReturn(Instances data,int maxNumAttribute) throws Exception {
		String methodAbstract = "attributes_selection_chi_" + maxNumAttribute;
		CommonUtils.print("--------------- " + methodAbstract + " ---------------");
		CommonUtils.print("class attribute : " + data.attribute(data.classIndex()));
//		InfoGainAttributeEval igAttributeEval = new InfoGainAttributeEval();
		IGAttributeEval2 igAttributeEval = new IGAttributeEval2();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(igAttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(data);
		Instances afterAsArrff = Filter.useFilter(data, asFilter);
		return afterAsArrff;
	}
	/**
	 * 对实例IG特征选择 并返回 选择后的实例
	 * @param data
	 * @param maxNumAttribute
	 * @return
	 * @throws Exception
	 */
	public static Instances igEvaluationAndReturn(Instances data,int maxNumAttribute) throws Exception{
		String methodAbstract = "attributes_selection_ig_" + maxNumAttribute;
		CommonUtils.print("--------------- " + methodAbstract + " ---------------");
		CommonUtils.print("class attribute : " + data.attribute(data.classIndex()));
		InfoGainAttributeEval igAttributeEval = new InfoGainAttributeEval();
		igAttributeEval.setBinarizeNumericAttributes(true);
//		CHIAttributeEval chiAttributeEval = new CHIAttributeEval();
		Ranker ranker = new Ranker();
		ranker.setNumToSelect(maxNumAttribute);
		AttributeSelection asFilter = new AttributeSelection();
		asFilter.setEvaluator(igAttributeEval);
		asFilter.setSearch(ranker);
		asFilter.setInputFormat(data);
		Instances afterAsArrff = Filter.useFilter(data, asFilter);
		return afterAsArrff;
	}

	
}
