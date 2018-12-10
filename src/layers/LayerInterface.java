package layers;
import org.jblas.DoubleMatrix;

public interface LayerInterface {
	
	/**
	 * Forward pass on layer
	 * @param input - DoubleMatrix of size (number of examples, input_dim)
	 * @return Layer output
	 */
	public DoubleMatrix forward(DoubleMatrix input);
	
	/**
	 * Backward pass on layer
	 * @param grad_output - upstream gradient with respect to layer output
	 * @return Upstream gradient with respect to layer input
	 */
	public DoubleMatrix backward(DoubleMatrix grad_output);
	
	/**
	 * Connects layer to network based on 
	 * output dimensions of previous layer.
	 * @param input_dim - layer input dimensions
	 * @return Layer output dimensions
	 */
	public int connect(int input_dim);

}
