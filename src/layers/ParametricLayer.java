package layers;
import java.util.Random;

import org.jblas.DoubleMatrix;

public abstract class ParametricLayer extends Layer { 
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public DoubleMatrix weights;
	transient public DoubleMatrix d_weights;
	transient private DoubleMatrix delta_weights;
	
	public boolean regL2 = false;
	public double l2_lambda = 0.0;
	public double reg = 0.0;
	transient public DoubleMatrix d_reg;
	
	/**
	 * Initializes weights with randomly sampled values on an interval.
	 * @param bound - interval bounds [-bound, bound]
	 */
	public void initializeWeights(double bound) {
		Random generator = new Random();
		for (int r=0; r < this.weights.rows; r++) {
			for (int c=0; c < this.weights.columns; c++) {
				double random_number = generator.doubles(-bound, bound).findAny().getAsDouble();
				this.weights.put(r, c, random_number);
			}
		}
		
		this.delta_weights = DoubleMatrix.zeros(this.weights.rows, this.weights.columns);
	}
	
	/**
	 * Sets weights to zero.
	 */
	public void zeroWeights() {
		this.weights.fill(0);
	}
	
	public void regularization(double l2_lambda) {
		this.regL2 = true;
		this.l2_lambda = l2_lambda;
	}
	
	/**
	 * Single weight update based on gradient descent.
	 * Sets gradients to 0 after execution.
	 * @param learning_rate - scalar multiple of gradient step
	 * @param momentum - scalar multiple of momentum term
	 */
	public void updateWeights(double learning_rate, double momentum) {
		DoubleMatrix new_weights = this.weights.sub(this.d_weights.mul(learning_rate));
		new_weights.addi(this.delta_weights.mul(momentum));
		this.delta_weights = new_weights.sub(this.weights);
		this.weights = new_weights;
		this.d_weights.fill(0);
		if (this.regL2) {
			this.d_reg.fill(0);
		}
	}
}
