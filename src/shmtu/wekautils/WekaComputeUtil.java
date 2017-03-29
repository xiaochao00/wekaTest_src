package shmtu.wekautils;
/**
 * 计算 工具类
 * @author HP_xiaochao
 *在计算的时候一定要注意 除号两边必须是小数 如果是整除的话，虽然小数点没有，但对精度有一定的影响
 */
public class WekaComputeUtil {
	/**
	 * 根据属性的 每类分布和各类总分布 计算卡方值
	 * 属性对于整个集合的卡方值是对于所有类的卡方值的最大值
	 * @param rowArr
	 * @param columnSum
	 * @return
	 */
	public static double computeMaxCHI(double[]rowArr,double[]columnSum){
		double maxChi = 0.0;
		double rowSum = 0.0;
		double N = 0.0;
		
		for(int i=0;i<rowArr.length;i++){
			rowSum+=rowArr[i];
			N+=columnSum[i];
		}
		//N(AD-BC)/(A+C)(A+B)(B+D)(D+C)
		for(int i=0;i<rowArr.length;i++){
			double A = rowArr[i];
			double B = rowSum - A;
			double C = columnSum[i] - A;
			double D = N-A-B-C;
			double chi = N*(A*D-B*C)*(A*D-B*C)/((A+C)*(A+B)*(B+D)*(C+D));
			if(chi>maxChi)
				maxChi = chi;
		}
		return maxChi;
	}
	/**
	 * 返回 某特征的 相对于所有类别的 chi值
	 * @param rowArr
	 * @param columnSum
	 * @return
	 */
	public static double []computeCHI(double[]rowArr,double[]columnSum){
		double chiarr[] = new double[rowArr.length];
		//
		double rowSum = 0.0;
		double N = 0.0;
		
		for(int i=0;i<rowArr.length;i++){
			rowSum+=rowArr[i];
			N+=columnSum[i];
		}
		//N(AD-BC)/(A+C)(A+B)(B+D)(D+C)
		for(int i=0;i<rowArr.length;i++){
			double A = rowArr[i];
			double B = rowSum - A;
			double C = columnSum[i] - A;
			double D = N-A-B-C;
			double chi = N*(A*D-B*C)*(A*D-B*C)/((A+C)*(A+B)*(B+D)*(C+D));
			chiarr[i] = chi;
		}
		return chiarr;
	} 
	/**
	 * 根据两个词在各种情况下的类分布 计算两个词的联合熵
	 * 该方法在计算两个词语关联度的时候用到的公式
	 * 后来效果不好 抛弃了
	 * @param N11
	 * @param N01
	 * @param N10
	 * @param N00
	 * @param classSum
	 * @return
	 */
	public static double computeUnionEntropy(int[]N11,int[]N01,int[]N10,int[]N00,int []classSum){
		int N = sumIntArr(classSum);
		int n11 = sumIntArr(N11);
		int n01 = sumIntArr(N01);
		int n10 = sumIntArr(N10);
		int n00 = sumIntArr(N00);
		double unionEntropy = 0.0;
		unionEntropy = (n11*entropy(N11)+n01*entropy(N01)+n10*entropy(N10)+n00*entropy(N00))/N;
		return unionEntropy;
	}
	/**
	 * 计算两个词的联合熵算法2 两个词的情况仅仅分为 都出现 和其他 计算熵
	 * 后来观察该方法在之前存在 即没用
	 * @param N11
	 * @param classSum
	 * @return
	 */
	public static double computeUnionEntropy2(int[]N11,int []classSum){
		int N = sumIntArr(classSum);
		int[]N00 = substractIntArr(classSum,N11);
		int n11 = sumIntArr(N11);
		int n00 = sumIntArr(N00);
		double unionEntropy = 0.0;
		unionEntropy = (n11*entropy(N11)+n00*entropy(N00))/N;
		return unionEntropy;
	}
	/**
	 * 根据属性的类分布 和每类的分布和 计算 互信息
	 * 返回最大的
	 * @param rowArr
	 * @param columnSum
	 * @return
	 */
	public static double computeMaxMI(double[]rowArr,double[]columnSum){
		double maxMI = 0.0;
		double rowSum = 0.0;
		double N = 0.0;
		
		for(int i=0;i<rowArr.length;i++){
			rowSum+=rowArr[i];
			N+=columnSum[i];
		}
		//max logNA/(A+C)(A+B)
		for(int i=0;i<rowArr.length;i++){
			double A = rowArr[i];
			double B = rowSum - A;
			double C = columnSum[i] - A;
//			double D = N-A-B-C;
			double mi = Math.log(A*N*1.0/(A+C)*(A+B));
			if(mi>maxMI)
				maxMI = mi;
		}
		return maxMI;
	}
	/**
	 * MI计算
	 * 返回平均值
	 * 效果不好 使用最大值好
	 * @param rowArr
	 * @param columnSum
	 * @return
	 */
	public static double computeAVGMI(double[]rowArr,double[]columnSum){
		double maxMI = 0.0;
		double rowSum = 0.0;
		double N = 0.0;
		
		for(int i=0;i<rowArr.length;i++){
			rowSum+=rowArr[i];
			N+=columnSum[i];
		}
		//(A+C)/N*logNA/(A+C)(A+B)
		for(int i=0;i<rowArr.length;i++){
			double A = rowArr[i];
			double B = rowSum - A;
			double C = columnSum[i] - A;
//			double D = N-A-B-C;
			double mi = Math.log(A*N*1.0/(A+C)*(A+B));
			maxMI+=(A+C)*1.0/N*mi;
		}
		return maxMI;
	}
	/**
	 * 求整形数组和公式
	 * @param intArr
	 * @return
	 */
	public static int sumIntArr(int[]intArr){
		if(intArr==null)
			return 0;
		int sum = 0;
		for(int i:intArr){
			sum+=i;
		}
		return sum;
	}
	/**
	 * 根据 某词的类分布数组和总的类分布数组 计算熵
	 * @param termClass
	 * @param classSum
	 * @return
	 */
	public static double computeEntropy(int[]termClass,int []classSum){
		double entropy = 0.0;
		int[]noTermClass = new int[termClass.length];
		for(int i=0;i<termClass.length;i++){
			noTermClass[i] = classSum[i] - termClass[i];
		}
		int sumHasTerm = sumIntArr(termClass);
		int sumNoTerm = sumIntArr(noTermClass);
		int N = sumIntArr(classSum);
		if(sumHasTerm+sumNoTerm!=N){
			System.out.println("熵计算出错 请检查");
			System.exit(0);
		}
		double entropyHasTerm = entropy(termClass);
		double entropyNoTerm = entropy(noTermClass);
		entropy = sumHasTerm*entropyHasTerm/N+sumNoTerm*entropyNoTerm/N;
		return entropy;
	}
	/**
	 * 根据两个词的类分布计算交叉熵
	 * 根据词的出现与否计算
	 * 交叉熵越大 两个词越关联
	 * ece(w1,w2) 
	 * ece(w2,w1)值不同
	 * 			w2	w2非
	 * w1		A	B
	 * w1非		C	D
	 * ece(w1,w2)=A/(A+B)*log(AN/((A+B)*(A+C)))+C/(C+D)*log(C*N/((C+D)(A+B)))
	 * @param n11
	 * @param n10
	 * @param n01
	 * @param n00
	 * @return
	 */
	public static double exceptCrossEntropyByWordClassDistribution(int n11[],int n10[],int n01[],int n00[]){
		double ece = 0.0;
		double A = sumIntArr(n11)*1.0;
		double B = sumIntArr(n10)*1.0;
		double C = sumIntArr(n01)*1.0;
		double D = sumIntArr(n00)*1.0;
		ece = eceCompute(A,B,C,D)*1.0;
		if(Double.isNaN(ece)){
			System.out.println(ece);
		}
		return ece;
	}
	/**
	 * 计算交叉熵
	 * ece(w1,w2)=A/(A+B)*log(AN/((A+B)*(A+C)))+C/(C+D)*log(C*N/((C+D)(A+B)))
	 * ece(w1,w2)=A/(A+B)*log(AN/((A+B)*(A+C)))  仅仅考虑w1存在条件下w2的影响，不考虑不出现的情况
	 * @param A
	 * @param B
	 * @param C
	 * @param D
	 * @return
	 */
	public static double eceCompute(double A,double B,double C,double D){
		double N = A+B+C+D;
//		double p = (A+B)/N;
		double ece = 0.0;
		double ece1 = 0.0;
//		double ece2 = 0.0;
		ece1 = A/(A+B)*computeLog(A*N/((A+B)*(A+C)));
//		ece2 = C/(C+D)*computeLog(C*N/((C+D)*(A+B)));
		if(Double.isNaN(ece1)||Double.isInfinite(ece1))
			ece1 = 0.0;
//		if(Double.isNaN(ece2)||Double.isInfinite(ece2))
//			ece2 = 0.0;
//		ece = p*(ece1+ece2);
		ece = ece1;
		if(Double.isNaN(ece)){
			System.out.println("A:"+A+"B:"+B+"C:"+C+"D:"+D+"N"+N);
		}
		return ece;
	}
	private static double computeLog(double d){
		if(d==0.0)
			return 0.0;
		else
			return Math.log(d);
	}
	public static double sumDouble(double[]darr){
		double sum = 0.0;
		if(darr!=null&&darr.length>0){
			for(double d:darr){
				sum+=d;
			}
		}
		return sum;
	}
	/**
	 * 熵公式计算
	 * 未添加平滑处理
	 * @param arr
	 * @return
	 */
	public static double entropy(int[]arr){
		int sum = sumIntArr(arr);
		if(sum==0)
			return 0;
		double entropy = 0.0;
		if(arr==null)
			return 0.0;
		for(int i:arr){
			double p = i*1.0/sum;
			if(p!=0)
				entropy+=p*Math.log(p);
		}
		return -entropy;
	}
	/**
	 * 
	 * @param n11
	 * @param n01
	 * @param n10
	 * @param n00
	 * @param N
	 * @return
	 */
	public static double computeEntropyN(int n11,int n01,int n10,int n00,int N){
		int sum = n11+n01+n10+n00;
		double entropyN11 = entropyByN(n11,N);
		double entropyN01 = entropyByN(n01,N);
		double entropyN10 = entropyByN(n10,N);
		double entropyN00 = entropyByN(n00,N);
		double entropy = n11*1.0/N*entropyN11+n10*1.0/N*entropyN10+n01*1.0/N*entropyN01+n00*1.0/N*entropyN00;
		return entropy;
	}
	public static double entropyByN(int ashNum,int N){
		double p1 = ashNum*1.0/N;
		double p2 =	(N-ashNum)*1.0/N;
		double entropy = (p1==0?0:p1*Math.log(p1))+(p2==0?0:p2*Math.log(p2));
		return -entropy;
	}
	/**
	 * 两个整形数组的差值
	 * @param a1
	 * @param a2
	 * @return
	 */
	public static int[] substractIntArr(int[]a1,int[]a2){
		int[]a3 = new int[a1.length];
		for(int i=0;i<a1.length;i++){
			a3[i] = a1[i] - a2[i];
		}
		return a3;
	}
	/**
	 * 两个整形数组的和
	 * @param a1
	 * @param a2
	 * @return
	 */
	public static int[]plusIntArr(int[]a1,int[]a2){
		int[]a3 = new int[a1.length];
		for(int i=0;i<a1.length;i++){
			a3[i] = a1[i] + a2[i];
		}
		return a3;
	}
}
