package layers;
import org.jblas.DoubleMatrix;

public abstract class Layer implements LayerInterface, java.io.Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public DoubleMatrix d_inputs;
	
	@Override
	public abstract DoubleMatrix forward(DoubleMatrix input);
	@Override
	public abstract DoubleMatrix backward(DoubleMatrix grad_output);
	@Override
	public abstract int connect(int input_dim);
}
