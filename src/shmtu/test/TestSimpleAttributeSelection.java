package shmtu.test;

import shmtu.util.CommonUtils;
import shmtu.wekautils.WekaUtil;
import weka.core.Instances;
import weka.core.Utils;

public class TestSimpleAttributeSelection {
	public static void main(String[] args) {
//		testUtils();
		testRemoveAttribute();
	}
	
	public static void testRemoveAttribute(){
		try {
			//1.
			Instances datas = WekaUtil.loadArffByDataSource("wekafiles/simple.arff");
			CommonUtils.print(datas.toString());
			//2.
			datas.deleteAttributeAt(1);
			CommonUtils.print(datas.toString());
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
	/**
	 * 使用 weka排序的工具类
	 */
	public static void testUtils(){
		double [] d = {0,1.2,1,2.9,2};
		int[]index = Utils.sort(d);
		for(int i=0;i<index.length;i++){
			double value = d[index[i]];
			System.out.println(value+" is in order index "+i);
		}
	}
}
