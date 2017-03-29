package shmtu.util;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 公共操作工具类
 * @author HP_xiaochao
 *2016年12月19日
 *
 */
public class CommonUtils {
	
	/**
	 * 泛型 数组转列表
	 * @param arr
	 * @return
	 */
	public static <T> List<T> arrayToList(T[]arr){
		if(arr==null)
			return null;
		List<T> list = new ArrayList<T>();
		for(T t:arr){
			list.add(t);
		}
		return list;
	}
	/**
	 * 字符串列表转字符串数组
	 * @param list
	 * @return
	 */
	public static String[] stringlistToArr(List<String>list){
		if(list==null)
			return null;
		String[] arr = new String[list.size()] ;
		for(int i=0;i<list.size();i++){
			arr[i] = list.get(i);
		}
		return arr;
	}
	/**
	 * 判断字符串是否是空 
	 * 	空返回 真；非空返回假
	 * @param str
	 * @return
	 */
	public static boolean stringIsEmpty(String str){
		if(str==null||"".equals(str))
			return true;
		return false;
	}
	/**
	 * 打印内容
	 * @param content
	 */
	public static void print(String content){
		System.out.println(content);
	}
	/**
	 * 内容写入指定文件编码UTF-8
	 * @param file
	 * @param content
	 */
	public static void writeFile(String file,String content){
		try {
			ReadWriteFileWithEncode.writeByEncodie(file,content,ReadWriteFileWithEncode.DEFAULT_ENCODE);
			System.out.println("写入文件成功。" + file);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	/**
	 * 列表写入指定文件
	 * @param filename
	 * @param strList
	 */
	public static void writeFileByList(String filename ,List<String> strList){
		StringBuffer content = new StringBuffer();
		for(String l:strList){
			content.append(l+"\r\n");
		}
		writeFile(filename, content.toString());
	}
	/**
	 * 列表为空的判断
	 * @param list
	 * @return
	 */
	public static <T> boolean listIsEmpty(List<T> list){
		if(list==null || list.size()==0)
			return true;
		return false;
	}
	/**
	 * 按行读取文件 返回行列表
	 * @param filename
	 * @return
	 */
	public static List<String> readFileLines(String filename){
		return ReadWriteFileWithEncode.readlinesByEncode(new File(filename), ReadWriteFileWithEncode.DEFAULT_ENCODE);
	}
	/**
	 * 对int数组排序 升序
	 * @param arr
	 * @return
	 */
	public static int[] sortIntArr(int[]arr){
		Arrays.sort(arr);
		return arr;
	}
	/**
	 * 获取矩阵的稀疏数组的索引 ，
	 * @param indexI
	 * @param indexJ
	 * @param n
	 * @return
	 */
	public static int computeSparesIndex(int indexI,int indexJ,int n){
		int sparesIndex = 0;
		int i = indexI+1;
		int j = indexJ+1;
		//公式 (2n-i)(i-1)/2+(j-1)-1
		sparesIndex = (2*n-i)*(i-1)/2+(j-i)-1;
//		System.out.println("sparesIndex:"+i+"-"+j+"="+sparesIndex);
		return sparesIndex;
	}
	/**
	 * 逆置列表
	 * @param tArr
	 * @return
	 */
	public static <T> T[]reverseArr(T[] tArr){
		T[] tArr2 = tArr;
		int len = tArr.length;
		if(len==0)
			return null;
		for(int i=0;i<len;i++){
			tArr2[len-i-1] = tArr[i];
		}
		return tArr2;
	}
	/**
	 * 分割相似列表字符串 返回长度
	 * @param similaritylist
	 * @return
	 */
	public static int getSimilitylistLen(String similaritylist){
		int len = 0;
		if(!CommonUtils.stringIsEmpty(similaritylist)){
			String similarityArr[] = similaritylist.split(" ");
			for(String str:similarityArr){
				if(!CommonUtils.stringIsEmpty(str))
					len++;
			}
		}
		return len;
	}
	/**
	 * 判断arff中text字段的长度
	 * @param text
	 * @return
	 */
	public static int textlen(String text){
		int len = 0;
		if(text!=null&&(!text.equals(" "))&&text.startsWith("'")&&text.endsWith("'")){
			text = text.substring(1,text.length()-1);
			String arr[]=text.split(" ");
			for(String a:arr){
				if(!stringIsEmpty(a))
					len++;
			}
		}
		return len;
	}
}
