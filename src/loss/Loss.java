package loss;


import org.jblas.DoubleMatrix;

public class Loss {
	public final double value;
	public final DoubleMatrix gradient;
	
	public Loss(double value, DoubleMatrix gradient) {
		this.value = value;
		this.gradient = gradient;
	}	
}
