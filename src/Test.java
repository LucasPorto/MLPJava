
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import org.jblas.DoubleMatrix;
import layers.AffineLayer;
import layers.SigmoidLayer;
import loss.Loss;

public class Test {
	static NeuralNet createNetwork(boolean bipolar) {
		NeuralNet net = new NeuralNet(2);
		net.addLayer(new AffineLayer(4));
		net.addLayer(new SigmoidLayer(bipolar));
		net.addLayer(new AffineLayer(1));
		net.addLayer(new SigmoidLayer(bipolar));
		net.initializeWeights(0.5);
		return net;
	}
	
	static void saveLoss(DoubleMatrix loss) {
		PrintStream w = null;
		try {
			w = new PrintStream(new FileOutputStream(new File("loss.csv")));
			for (int r = 0; r < loss.rows; r++) {
				for (int c = 0; c < loss.columns; c++) {
					if (c + 1 < loss.columns) {
						w.print(loss.get(r, c));
						w.print(", ");
					} else {
						w.println(loss.get(r,c));
					}
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			w.flush();
			w.close();
		}
	}
	
	public static void main(String[] args) throws IOException {
		
		DoubleMatrix X;
		DoubleMatrix y;
		boolean bipolar = true;

		// Rows: number of examples, Cols: Example dimensions
		if (bipolar) {
			X = DoubleMatrix.valueOf("-1 -1; -1 1; 1 -1; 1 1");
			y = DoubleMatrix.valueOf("-1; 1; 1; -1");
		} else {
			X = DoubleMatrix.valueOf("0 0; 0 1; 1 0; 1 1");
			y = DoubleMatrix.valueOf("0; 1; 1; 0");
		}
		
		int trials = 1;
		for (int i = 0; i < trials; i++) {
			NeuralNet net = createNetwork(bipolar);
			int epochs = 500;
			double learning_rate =  0.2;
			double momentum = 0.9;
			DoubleMatrix training_data = DoubleMatrix.zeros(epochs, 1);
			for (int j = 0; j < epochs; j++) {
				Loss loss = net.train(X, y, learning_rate, momentum);
				training_data.put(j, 0, loss.value);
				System.out.println(loss.value);
				if (loss.value < 0.001) {
					saveLoss(training_data);
					break;
				}
			}
			System.out.println(net.forward(X));
		}
	}
}
