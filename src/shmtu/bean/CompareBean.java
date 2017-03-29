package shmtu.bean;
/**
 * 比较实体 得父类
 * @author HP_xiaochao
 *2016年12月21日
 *
 */
public class CompareBean {
	private double value;
	private int index;
	public double getValue() {
		if(Double.isNaN(value))
			return 0.0;
		return value;
	}
	public void setValue(double value) {
		this.value = value;
	}
	public int getIndex() {
		return index;
	}
	public void setIndex(int index) {
		this.index = index;
	}
	@Override
	public String toString(){
		return "index:"+index+";value:"+value;
	}
}
