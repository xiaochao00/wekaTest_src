package shmtu.test;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.converters.XRFFSaver;
import weka.core.Instances;

import java.io.File;

import shmtu.wekautils.WekaUtil;

/**
 * Loads file "args[0]", sets class if necessary (in that case the last 
 * attribute), adds some test weights and saves it as XRFF file
 * under "args[1]". E.g.: <br/>
 *   AddWeights anneal.arff anneal.xrff.gz
 *
 * @author FracPete (fracpete at waikato dot ac dot nz)
 */
public class AddWeights {
  public static void main(String[] args) throws Exception {
    // load data
	
    DataSource source = new DataSource("wekafiles/simple.arff");
    Instances data = source.getDataSet();
    if (data.classIndex() == -1)
      data.setClassIndex(data.numAttributes() - 1);

    // set weights
    double factor = 0.5  / (double) data.numInstances();
    for (int i = 0; i < data.numInstances(); i++) {
    	System.out.println(0.5 + factor*i);
      data.instance(i).setWeight(0.5 + factor*i);
    }

    // save data
//    XRFFSaver saver = new XRFFSaver();
//    saver.setFile(new File("wekafiles/simple_weight.arff"));
//    saver.setInstances(data);
//    saver.writeBatch();
    WekaUtil.saveArffByInstancesByDataSink(data, "wekafiles/simple_weight.arff");
  }
}