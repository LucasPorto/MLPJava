package layers;
import org.jblas.DoubleMatrix;

public class AffineLayer extends ParametricLayer {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public final int units;
	public int[] weights_rows;
	
	
	/**
	 * Affine/Dense layer
	 * @param units - number of hidden units
	 */
	public AffineLayer(int units) {
		this.units = units;
	}
	
	public AffineLayer(int units, double regL2Lambda) {
		this.units = units;
		this.regularization(regL2Lambda);
	}

	public int connect(int input_dim) {
		// Weights matrix is [weights; biases]
		this.weights = DoubleMatrix.zeros(input_dim + 1, this.units);
		this.weights_rows = new int[input_dim];
		for (int c = 0; c < input_dim; c++) { 
			this.weights_rows[c] = c;
		}
		return this.units;
	}
	
	@Override
	public DoubleMatrix forward(DoubleMatrix input) {
		DoubleMatrix biased_input = DoubleMatrix.concatHorizontally(input, DoubleMatrix.ones(input.rows, 1));
		
		// Discard bias terms from weights
		this.d_inputs = this.weights.getRows(this.weights_rows).transpose();
		this.d_weights = biased_input.transpose();
		
		if (this.regL2) {
			this.reg = (this.l2_lambda/2.0)*this.weights.norm2();
			this.d_reg = this.weights.mul(this.l2_lambda);
		}
		return biased_input.mmul(this.weights);
	}
	
	@Override
	public DoubleMatrix backward(DoubleMatrix grad_output) {
		this.d_weights = this.d_weights.mmul(grad_output);
		if (this.regL2) {
			this.d_weights.addi(this.d_reg);
		}
		return grad_output.mmul(this.d_inputs);
	}

	@Override
	public void initializeWeights(double bound) {
		super.initializeWeights(bound);
		this.weights.putRow(this.weights.rows - 1, DoubleMatrix.zeros(1, this.weights.columns));
	}
}
 
