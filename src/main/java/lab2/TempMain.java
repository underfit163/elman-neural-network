package lab2;

import java.io.*;
import java.util.Arrays;

public class TempMain {
    public static void main(String[] args) {
        String fileName = "temp.data.txt";
        int countNum = 301;
        double[] allData = readData(fileName, countNum);
        System.out.println(Arrays.toString(allData));

        double[] trainData = new double[300];
        double[] testData = new double[6];

        System.arraycopy(allData, 0, trainData, 0, 300);

        System.arraycopy(allData, 295, testData, 0, 6);

        int hiddenLen = 10;
        double alpha = 0.1;
        double koefMoment = 0.05;
        int epoch = 50;
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

    private static double[] readData(String fileName, int countNum) {
        double[] allData = new double[countNum];
        try (BufferedReader bufferedReader = new BufferedReader(new FileReader(fileName))) {
            for (int i = 0; i < countNum; i++) {
                allData[i] = Double.parseDouble(bufferedReader.readLine().split(",")[1]);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return allData;
    }

    public static double[][] normalizeData(double[][] data) {
        for (int i = 0; i < data.length; i++) {
            double divider = calcDividerForNormalization(data[i]);
            for (int j = 0; j < data[i].length - 1; j++) {
                data[i][j] = data[i][j] / divider;
            }
        }
        return data;
    }

    private static double calcDividerForNormalization(double[] vector) {
        double result = 0;
        for (int i = 0; i < vector.length - 1; i++) {
            result += Math.pow(vector[i], 2);
        }
        return Math.sqrt(result);
    }
}
