package shmtu.extral.attributeSelection;

import java.util.Enumeration;

import weka.attributeSelection.ASEvaluation;
import weka.attributeSelection.AttributeEvaluator;
import weka.core.Capabilities;
import weka.core.ContingencyTables;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.Capabilities.Capability;

/**
 * 卡方检验 特征评估
 * 
 * @author HP_xiaochao
 *
 */
public class IGAttributeEval2 extends ASEvaluation implements AttributeEvaluator, OptionHandler {
	private double[] m_InfoGains;
	/** Treat missing values as a seperate value */
	private boolean m_missing_merge;

	public IGAttributeEval2() {
		resetOptions();
	}

	/** Just binarize numeric attributes */
	private boolean m_Binarize;

	@Override
	public Enumeration<Option> listOptions() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();

		// attributes
		result.enable(Capability.NOMINAL_ATTRIBUTES);
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.DATE_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);

		// class
		result.enable(Capability.NOMINAL_CLASS);
		result.enable(Capability.MISSING_CLASS_VALUES);

		return result;
	}

	@Override
	public void setOptions(String[] options) throws Exception {
		// TODO Auto-generated method stub

	}

	/**
	 * get whether missing values are being distributed or not
	 * 
	 * @return true if missing values are being distributed.
	 */
	public boolean getMissingMerge() {
		return m_missing_merge;
	}

	/**
	 * get whether numeric attributes are just being binarized.
	 * 
	 * @return true if missing values are being distributed.
	 */
	public boolean getBinarizeNumericAttributes() {
		return m_Binarize;
	}

	@Override
	public String[] getOptions() {
		// TODO Auto-generated method stub
		// return null;
		String[] options = new String[2];
		int current = 0;

		if (!getMissingMerge()) {
			options[current++] = "-M";
		}
		if (getBinarizeNumericAttributes()) {
			options[current++] = "-B";
		}

		while (current < options.length) {
			options[current++] = "";
		}

		return options;

	}

	/**
	 * Reset options to their default values
	 */
	protected void resetOptions() {
		m_InfoGains = null;
		m_missing_merge = true;
		m_Binarize = false;
	}

	@Override
	public double evaluateAttribute(int attribute) throws Exception {
		// TODO Auto-generated method stub
		return m_InfoGains[attribute];
	}

	@Override
	public void buildEvaluator(Instances data) throws Exception {
		// TODO Auto-generated method stub
		// can evaluator handle data?
		getCapabilities().testWithFail(data);

		int classIndex = data.classIndex();
		int numInstances = data.numInstances();
		int numClasses = data.attribute(classIndex).numValues();
		// Reserve space and initialize counters
	    double[][][] counts = new double[data.numAttributes()][][];
	    for (int k = 0; k < data.numAttributes(); k++) {
	      if (k != classIndex) {
	        int numValues = data.attribute(k).numValues();
	        counts[k] = new double[2][numClasses + 1];
	      }
	    }

	    // Initialize counters
	    double[] temp = new double[numClasses + 1];
	    for (int k = 0; k < numInstances; k++) {
	      Instance inst = data.instance(k);
	      if (inst.classIsMissing()) {
	        temp[numClasses] += inst.weight();
	      } else {
	        temp[(int) inst.classValue()] += inst.weight();
	      }
	    }
	    for (int k = 0; k < counts.length; k++) {
	      if (k != classIndex) {
	        for (int i = 0; i < temp.length; i++) {
	          counts[k][0][i] = temp[i];
	        }
	      }
	    }
		// compute counts
		int classN = data.classAttribute().numValues();
		for (int k = 0; k < numInstances; k++) {
			Instance inst = data.instance(k);
			double classValue = inst.classValue();
			int class_index = (int) classValue;
			for (int i = 0; i < inst.numValues(); i++) {
				int wordIndex = inst.index(i);
				if (wordIndex != classIndex) {
					counts[wordIndex][1][class_index] += 1;
					counts[wordIndex][0][class_index] -= 1;
				}
			}
		}
		 // Compute info gains
	    m_InfoGains = new double[data.numAttributes()];
	    for (int i = 0; i < data.numAttributes(); i++) {
	      if (i != classIndex) {
	        m_InfoGains[i] = (ContingencyTables.entropyOverColumns(counts[i]) - ContingencyTables
	          .entropyConditionedOnRows(counts[i]));
	      }
	    }
	}

}
