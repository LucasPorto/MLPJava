package loss;
import org.jblas.DoubleMatrix;
public class LossFunctions {
	public static Loss meanSquaredError(DoubleMatrix output, DoubleMatrix true_values) {
		DoubleMatrix gradient =  output.sub(true_values);
		double error = Math.pow(gradient.norm2(), 2)*(1.0/(double) true_values.length);
		Loss cost = new Loss(error, gradient);
		return cost;
	}
}
