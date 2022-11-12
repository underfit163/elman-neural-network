package lab2;

import java.io.*;
import java.util.Arrays;

public class Main {
    public static void main(String[] args) {
        String fileName = "sigma.data.txt";
        int countNum = 301;
        double[] allData = generateData(fileName, countNum);

        double[] trainData = new double[300];
        double[] testData = new double[6];

        System.arraycopy(allData, 0, trainData, 0, 300);

        System.arraycopy(allData, 295, testData, 0, 6);

        int hiddenLen = 10;
        double alpha = 0.1;
        double koefMoment = 0.05;
        int epoch = 20;
        int countOut = 1;

        NeuralNetwork neuralNetwork = new NeuralNetwork
                (5, hiddenLen, countOut, trainData, epoch, alpha, koefMoment);

        neuralNetwork.elmanTrain();

        neuralNetwork.setEntersZero();
        //neuralNetwork.clearHidden();
        System.out.println("Выходной слой:");
        double[] arr = neuralNetwork.elmanOuter(testData);
        System.out.println(Arrays.toString(arr));
    }

    private static double[] generateData(String fileName, int countNum) {
        double[] allData = new double[countNum];
        try (PrintWriter printWriter = new PrintWriter(new FileWriter(fileName))) {
            for (int i = 0; i < countNum; i++) {
                allData[i] = Math.sin((double) i / 2);
                printWriter.println(allData[i]);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return allData;
    }
}
