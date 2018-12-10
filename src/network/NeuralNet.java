package network;
import java.util.LinkedList;
import org.jblas.DoubleMatrix;
import layers.Layer;
import layers.ParametricLayer;
import loss.LossFunctions;
import loss.Loss;

import java.util.Iterator;

public class NeuralNet implements NetworkInterface {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	public final int input_dim;
	public int output_dim;
	public DoubleMatrix inputTransform;
	public double outputTransform;
	LinkedList<Layer> layers = new LinkedList<Layer>();
	
	
	public NeuralNet(int input_dim)
	{
		this.input_dim = input_dim;
		this.output_dim = input_dim;
	}
	
	@Override
	public void addLayer(Layer layer) {
		this.output_dim = layer.connect(this.output_dim);
		this.layers.add(layer);
	}

	public DoubleMatrix forward(DoubleMatrix input) {
		DoubleMatrix output = input;
		for (Layer layer: this.layers) {
			output = layer.forward(output);
		}
		return output;
	}
	
	public DoubleMatrix backward(DoubleMatrix grad_loss)
	{
		DoubleMatrix grad_output = grad_loss;
		Iterator<Layer> descit = this.layers.descendingIterator();
		while(descit.hasNext()) {
			grad_output = descit.next().backward(grad_output);
		}
		return grad_output;
	}
	
	@Override
	public Loss train(DoubleMatrix X, DoubleMatrix y, double learning_rate, double momentum) {
		DoubleMatrix output = this.forward(X);
		Loss cost = LossFunctions.meanSquaredError(output, y);
		this.backward(cost.gradient);
		this.updateWeights(learning_rate, momentum);
		return cost;
	}

	@Override
	public void initializeWeights(double bound) {
		for (Layer layer: this.layers) {
			if (layer instanceof ParametricLayer ) {
				((ParametricLayer) layer).initializeWeights(bound);
			}
		}
	}

	@Override
	public void zeroWeights() {
		for (Layer layer: this.layers) {
			if (layer instanceof ParametricLayer ) {
				((ParametricLayer) layer).zeroWeights();
			}
		}		
	}

	@Override
	public void updateWeights(double learning_rate, double momentum) {
		for (Layer layer: this.layers) {
			if (layer instanceof ParametricLayer ) {
				((ParametricLayer) layer).updateWeights(learning_rate, momentum);
			}
		}	
	}
	
	public double getRegularizationTerm() {
		double regTerm = 0.0;
		for (Layer layer: this.layers) {
			if (layer instanceof ParametricLayer ) {
				if (((ParametricLayer) layer).regL2) {
					regTerm += ((ParametricLayer) layer).reg;
				}
			}
		}
		return regTerm;
	}
}
