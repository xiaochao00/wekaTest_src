package shmtu.test;

import java.io.File;
import java.util.List;
import java.util.Random;

import shmtu.util.ReadWriteFileWithEncode;
import shmtu.wekautils.WekaUtil;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.core.AbstractInstance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSink;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.ModifyInstanceWeights;
import weka.filters.unsupervised.instance.instanceweightsmodifiers.FromFile;

public class Test2 {
	public static void main(String[] args) throws Exception {
		doSth();
	}
	
	public static void doSth() throws Exception{
		String tranFilename = "wekafiles/20News_train_vsm_ig500.arff";
		String testFilename = "wekafiles/20News_test_vsm_ig500.arff";
		Instances tran = WekaUtil.loadArffByDataSource(tranFilename);
		Instances test = WekaUtil.loadArffByDataSource(testFilename);
		tran.setClassIndex(tran.numAttributes()-1);
		test.setClassIndex(test.numAttributes()-1);
		//
		LibSVM nBayesM = new LibSVM();
		nBayesM.buildClassifier(tran);//
		//4.评估
		Evaluation evaluation = new Evaluation(tran);
	    evaluation.evaluateModel(nBayesM, test);
	    //5.评估 模型分类效果
		WekaUtil.printEvaluation(evaluation,"result/" + "权重调整" + "_"  + ".txt");
		//
		String weightfilename = "result/weight.txt";
		double[] weights = new double[tran.numInstances()];
		List<String> wlist = ReadWriteFileWithEncode.readlinesByEncode(new File(weightfilename), "utf-8");
		if(wlist.size()!=tran.numInstances()){
			System.out.println("数目不一致"+wlist.size()+";"+tran.numInstances());
		}else{
			for(int i=0;i<wlist.size();i++){
//				System.out.println("pro weight:"+tran.instance(i).weight());
				double weight = Double.parseDouble(wlist.get(i));
//				System.out.println(weight);
				weights[i] = weight;
				tran.instance(i).setWeight(weight);
			}
		}
		//
//		WekaUtil.saveArffBySaver(tran, "wekafiles/20News_train_vsm_ig500_weight.arff");
//		XRFFSaver saver = new XRFFSaver();
//	     saver.setFile(new File("wekafiles/20News_train_vsm_ig500_weight.arff"));
//	     saver.setInstances(tran);
//	     saver.writeBatch();
//		ModifyInstanceWeights modifyI = new ModifyInstanceWeights();
//		FromFile wfiler = new FromFile();
//		wfiler.setWeightsFile(new File(weightfilename));
//		modifyI.setModifier(wfiler);
//		modifyI.setInputFormat(tran);
//		Instances newtran = Filter.useFilter(tran, modifyI);
//		//
////	     Instances newtran = WekaUtil.loadArffByLoader("wekafiles/20News_train_vsm_ig500_weight.arff");
//	     newtran.setClassIndex(newtran.numAttributes()-1);
	     Instances newtran2 = tran.resampleWithWeights(new Random(1234), weights);
	     WekaUtil.saveArffBySaver(newtran2, "wekafiles/20News_train_vsm_ig500_weight.arff");
	     LibSVM nBayesM2 = new LibSVM();
		nBayesM2.buildClassifier(newtran2);//
		//4.评估
		Evaluation evaluation2 = new Evaluation(newtran2);
	    evaluation2.evaluateModel(nBayesM2, test);
	    //5.评估 模型分类效果
		WekaUtil.printEvaluation(evaluation2,"result/" + "权重调整" + "_"  + ".txt");
	}
	public static Instances weightModifyByModifyInstances(Instances data,double[]weights){
		Instances newData = 
		if(weights.length!=data.numInstances()){
			System.out.println("数目不一致"+weights.length+";"+data.numInstances());
		}else{
			for(int i=0;i<weights.length;i++){
				double weight = weights[i];
				weights[i] = weight;
				data.instance(i).setWeight(weight);
			}
		}
	}
	public static Instances weightModifyByUnofficial(Instances data,String weightfilename) throws Exception{
		ModifyInstanceWeights modifyI = new ModifyInstanceWeights();
		FromFile wfiler = new FromFile();
		wfiler.setWeightsFile(new File(weightfilename));
		modifyI.setModifier(wfiler);
		modifyI.setInputFormat(data);
		Instances newtran = Filter.useFilter(data, modifyI);
		return newtran;
	}
	public static Instances weightModifyByResampleWithWeights(Instances data,double[]weights){
		return data.resampleWithWeights(new Random(1234), weights);
	}
}
