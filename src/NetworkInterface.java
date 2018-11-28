import java.io.Serializable;
import org.jblas.DoubleMatrix;

import layers.Layer;
import loss.Loss;

public interface NetworkInterface extends Serializable {
	public Loss train(DoubleMatrix X, DoubleMatrix y, double learning_rate, double momentum);
	public void addLayer(Layer layer);
	public void initializeWeights(double bound);
	public void zeroWeights();
	public void updateWeights(double learning_rate, double momentum);
}
