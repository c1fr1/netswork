import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        int period = 500;
        Network network = new Network(784, 10, 150, 150);
        while (true) {
            try {
                Scanner s = new Scanner(new File("res/data.txt"));
                float costTotal = 0f;
                for (int i = 0; i < 42000; ++i) {
                    int target = s.nextInt();
                    float[] netIn = new float[784];
                    for (int n = 0; n < 784; n++) {
                        float brightness = (float) s.nextInt();
                        float britness = brightness / 255f;
                        netIn[n] = britness;
                    }
                    network.run(netIn);
                    float[] expectedOutput = new float[10];
                    Arrays.fill(expectedOutput, 0f);
                    expectedOutput[target] = 1f;
                    costTotal += network.getCost(expectedOutput);
                    network.backPropogate(expectedOutput);
                    if (i % period == 0 && i > 0) {
                        System.out.println("total cost round #" + i / period + ": " + costTotal);
                        System.out.println("average cost: " + (costTotal / (float) period));
                        costTotal = 0;
                        network.updateFlags();
                    }
                }
                s.close();
            } catch (FileNotFoundException e) {
                e.printStackTrace();
            }
        }
    }
    public static float sigmoid(float val) {
        return (float) (1/(1+Math.pow(Math.E, -val)));
    }
    public static float sigmoidPrime(float val) {
        if (Math.abs(val) > 500) {
            return  0f;
        }
        return (float) (Math.pow(Math.E, -val)/((1+Math.pow(Math.E, -val))*(1+Math.pow(Math.E, -val))));
    }
    public static float invSquash(float val) {
        return (float) -Math.log(1/val-1);
    }
}
