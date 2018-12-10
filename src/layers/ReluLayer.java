package layers;
import org.jblas.DoubleMatrix;

public class ReluLayer extends Layer {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public ReluLayer() {
	}

	@Override
	public DoubleMatrix forward(DoubleMatrix input) {
		this.d_inputs = input.gt(0);
		return input.mul(input.gt(0));
	}

	@Override
	public DoubleMatrix backward(DoubleMatrix grad_output) {
		return grad_output.mul(this.d_inputs);
	}

	@Override
	public int connect(int input_dim) {
		return input_dim;
	}
	
}
