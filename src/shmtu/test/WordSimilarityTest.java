package shmtu.test;

import java.io.File;
import java.io.IOException;
import java.util.List;

import edu.buaa.edu.wordsimilarity.WordSimilarity;
import shmtu.util.ReadWriteFileWithEncode;
import shmtu.wordsimilarity.WordSimilarityUtil;

public class WordSimilarityTest {
	public static void main(String[]args) throws IOException{
		String word1 = "大众";
		String word2 = "百姓";
		double s = WordSimilarityUtil.wordSimilarityCilin(word1, word2);
		double s2 = WordSimilarity.simWord(word1, word2);
		System.out.println("cilin similaity is :" + s);
		System.out.println("hownet similaity is :" + s2);
		doSome();
//		doSmoeCilin();
	}
	/**
	 * @throws IOException 
	 * 
	 */
	public static void doSome() throws IOException{
		List<String> atributenames = ReadWriteFileWithEncode.readlinesByEncode(new File("attributes_ig_1000.txt"), ReadWriteFileWithEncode.DEFAULT_ENCODE);
//		WordSimilarityUtil.attributeSimilarity(atributenames);
//		WordSimilarityUtil.attributeSimilarityAndPOS(atributenames);
		List<String> simialrityList = WordSimilarityUtil.computeSimilarityWordListByHownetAndCilin(atributenames);
		for(int i=0;i<simialrityList.size();i++){
			System.out.println( " similarity list is:" + simialrityList.get(i));
		}
	}
	
	public static void doSmoeCilin(){
		List<String> atributenames = ReadWriteFileWithEncode.readlinesByEncode(new File("result/attributes_selection_chi_1000.txt"), ReadWriteFileWithEncode.DEFAULT_ENCODE);
//		
		if(atributenames!=null){
			for(int i=0;i<atributenames.size();i++){
				String word1 = atributenames.get(i);
				for(int j=i+1;j<atributenames.size();j++){
					String  word2 = atributenames.get(j);
					double cilinSimilarity = WordSimilarityUtil.wordSimilarityCilin(word1, word2);
					double hownetSimilarity = WordSimilarity.simWord(word1, word2);
					if(cilinSimilarity>0.8&&hownetSimilarity>0.95&&(!(word1.length()==1||word2.length()==1))){
						System.out.println(word1+" and " +word2+" similarity is "+cilinSimilarity+";"+hownetSimilarity);
					}
				}
			}
		}
	}
}
