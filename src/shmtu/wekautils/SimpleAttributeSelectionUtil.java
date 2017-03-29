package shmtu.wekautils;

import java.util.HashSet;
import java.util.Set;

import shmtu.extral.attributeSelection.CHIAttributeEval;
import weka.core.Instances;
import weka.core.Utils;

public class SimpleAttributeSelectionUtil {
	
	public static void doChooseByCHI(Instances data,int maxNum) throws Exception{
		CHIAttributeEval chiEval = new CHIAttributeEval();
		chiEval.buildEvaluator(data);
		Set<Integer> termSet = new HashSet<Integer>();
		//
		//
		int classnum = data.classAttribute().numValues();
		int preNum = maxNum/classnum;
		for(int i=0;i<classnum;i++){
			//
			String classname = data.classAttribute().value(i);
			System.out.println(classname+":-------");
			//
			double []termeval = chiEval.termsEvaluateOfClass(i);
			int[]sortIndex = Utils.sort(termeval);
			int count = 0;
			int[]choseIndexArr = new int[preNum];
			for(int j=(sortIndex.length-1);count<preNum&&j>=0;count++,j--){
				int termindex = sortIndex[j];
				choseIndexArr[count] = termindex;
				String termname = data.attribute(termindex).name();
				System.out.printf(termname+";");
				termSet.add(termindex);
			}
			System.out.println();
		}
	}
	
	public static Instances simpleChooseByCHI(Instances data,int maxNum) throws Exception{
		CHIAttributeEval chiEval = new CHIAttributeEval();
		chiEval.buildEvaluator(data);
		Set<Integer> termSet = new HashSet<Integer>();
		//
		//
		int classnum = data.classAttribute().numValues();
		int preNum = maxNum/classnum;
		for(int i=0;i<classnum;i++){
			//
			String classname = data.classAttribute().value(i);
			System.out.println(classname+":-------");
			//
			double []termeval = chiEval.termsEvaluateOfClass(i);
			int[]sortIndex = Utils.sort(termeval);
			int count = 0;
			int[]choseIndexArr = new int[preNum];
			for(int j=(sortIndex.length-1);count<preNum&&j>=0;count++,j--){
				int termindex = sortIndex[j];
				choseIndexArr[count] = termindex;
				String termname = data.attribute(termindex).name();
				System.out.printf(termname+";");
				termSet.add(termindex);
			}
			System.out.println();
		}
		System.out.println("choose attribute num:"+termSet.size());
		for(int i=0;i<data.numAttributes()-1;i++){
			if(!termSet.contains(i)&&(i!=data.classIndex())){
				data.deleteAttributeAt(i);
			}
		}
		//
		return data;
	}
	
}
