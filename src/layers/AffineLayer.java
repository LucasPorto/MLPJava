package layers;
import org.jblas.DoubleMatrix;

public class AffineLayer extends ParametricLayer {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public final int units;	
	
	/**
	 * Affine/Dense layer
	 * @param units - number of hidden units
	 */
	public AffineLayer(int units) {
		this.units = units;
	}
	
	public AffineLayer(int units, boolean biased) {
		this.units = units;
		this.biased = biased;
	}

	public int connect(int input_dim) {
		// Weights matrix is [weights; biases]
		this.weights = DoubleMatrix.zeros(input_dim, this.units);
		if (this.biased) {
			this.intercepts = DoubleMatrix.zeros(1, this.units);
		}
		
		return this.units;
	}
	
	@Override
	public DoubleMatrix forward(DoubleMatrix input) {		
		// Discard bias terms from weights
		this.d_inputs = this.weights.transpose();
		this.d_weights = input.transpose();
		
		if (this.regL2) {
			this.reg = (this.l2_lambda/2.0)*this.weights.norm2();
			this.d_reg = this.weights.mul(this.l2_lambda);
		}
		
		DoubleMatrix output = input.mmul(this.weights);
		if (this.biased) {
			this.d_intercepts = DoubleMatrix.ones(1, input.rows);
			output.addiRowVector(this.intercepts);
		}
		
		return output;
	}
	
	@Override
	public DoubleMatrix backward(DoubleMatrix grad_output) {
		//Gradient w.r.t weights
		this.d_weights = this.d_weights.mmul(grad_output);
		if (this.regL2) {
			this.d_weights.addi(this.d_reg);
		}
		
		//Gradient w.r.t intercepts
		if (this.biased) {
			this.d_intercepts = this.d_intercepts.mmul(grad_output);
		}
		return grad_output.mmul(this.d_inputs);
	}

	@Override
	public void initializeWeights(double bound) {
		super.initializeWeights(bound);
	}
}
 
