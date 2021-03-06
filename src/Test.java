
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import org.jblas.DoubleMatrix;
import layers.AffineLayer;
import layers.ReluLayer;
import loss.Loss;
import network.NeuralNet;
import network.Utils;

public class Test {
	
	// Example two-layer architecture
	static NeuralNet createNetwork(int input_dim) {
		NeuralNet net = new NeuralNet(input_dim);
		net.addLayer(new AffineLayer(4, true));
		net.addLayer(new ReluLayer());
		net.addLayer(new AffineLayer(1, true));
		net.initializeWeights(0.5);
		return net;
	}
	
	// Saves loss to a csv file
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
		
		// Example dataset: XOR
		// Rows: number of examples, Cols: Example dimensions
		if (bipolar) {
			X = DoubleMatrix.valueOf("-1 -1; -1 1; 1 -1; 1 1");
			y = DoubleMatrix.valueOf("-1; 1; 1; -1");
		} else {
			X = DoubleMatrix.valueOf("0 0; 0 1; 1 0; 1 1");
			y = DoubleMatrix.valueOf("0; 1; 1; 0");
		}
		
		// Instantiate neural network
		NeuralNet net = createNetwork(X.columns);
		
		// Example learning parameters
		int epochs = 5000;
		double learning_rate =  0.01;
		double momentum = 0.9;
		
		// Loss data, to be saved later
		DoubleMatrix loss_data = DoubleMatrix.zeros(epochs, 1);
		
		// Training loop
		for (int j = 0; j < epochs; j++) {
			Loss loss = net.train(X, y, learning_rate, momentum);
			loss_data.put(j, 0, loss.value);
			System.out.println(loss.value);
			if (loss.value < 0.001) {
				saveLoss(loss_data);
				break;
			}
		}
		
		// Evaluation
		System.out.println(net.forward(X));
		
		// Saves network and weights to a readable file
		Utils.saveNetwork(net, "myNetwork");
	}
}
