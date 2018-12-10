package network;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

public class Utils {
	
	public static NeuralNet loadNetwork(String filename) {
		FileInputStream fis = null;
        ObjectInputStream in = null;
        NeuralNet network = null;
        try {
            fis = new FileInputStream(filename);
            in = new ObjectInputStream(fis);
            network = (NeuralNet) in.readObject();
            in.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
        return network;
	}
	
	public static void saveNetwork(NeuralNet network, String filename) {
		FileOutputStream fos = null;
        ObjectOutputStream out = null;
        try {
            fos = new FileOutputStream(filename);
            out = new ObjectOutputStream(fos);
            out.writeObject(network);
            out.close();
        } catch (Exception ex) {
            ex.printStackTrace();
        }
	}
}
