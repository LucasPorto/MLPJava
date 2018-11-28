package layers;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;


public class SigmoidLayer extends Layer {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public final boolean bipolar;
	
	/**
	 * @param bipolar
	 */
	public SigmoidLayer(boolean bipolar) {
		this.bipolar = bipolar;
	}
	
	@Override
	public int connect(int input_dim) {
		return input_dim;
	}
	
	@Override
	public DoubleMatrix forward(DoubleMatrix input) {
		DoubleMatrix sigmoid = this.sigmoidFunction(input);
		if (this.bipolar) {
			//DoubleMatrix plus = sigmoid.add(1);
			//DoubleMatrix minus = sigmoid.neg().add(1);
			this.d_inputs = MatrixFunctions.pow(sigmoid, 2).neg().add(1).mul(0.5);
		} else {
			this.d_inputs = sigmoid.mul(sigmoid.neg().add(1));
		}
		return sigmoid;
	}

	@Override
	public DoubleMatrix backward(DoubleMatrix grad_output) {
		return grad_output.mul(this.d_inputs);
	}
	
	/**
	 * Sigmoid function
	 * @param x
	 * @return 1/(1-exp(-x)) if binary or
	 * 		   1-2/(1-exp(-x)) if bipolar
	 */
	private DoubleMatrix sigmoidFunction(DoubleMatrix x) {
		DoubleMatrix denominator = MatrixFunctions.exp(x.neg()).add(1);
		DoubleMatrix sigmoid = MatrixFunctions.pow(denominator, -1);
		
		if (this.bipolar) {
			sigmoid.muli(2).subi(1);
			return sigmoid;
		} else {
			return sigmoid;
		}
	}

	
}
