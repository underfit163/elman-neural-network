package lab2;

import java.io.*;
import java.util.Arrays;

public class TempMain {
    public static void main(String[] args) {
        String fileName = "temp.data.txt";
        int countNum = 301;
        double[] allData = readData(fileName, countNum);
        System.out.println(Arrays.toString(allData));
        allData = normalizeData(allData);
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
        System.out.println(Arrays.toString(deNormalizeData(arr)));
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
    private static final double d2 = 1;
    private static final double d1 = -1;
    private static double maxEl;
    private static double minEl;

    public static double[] normalizeData(double[] data) {
        maxEl = Arrays.stream(data).max().orElse(0);
        minEl = Arrays.stream(data).min().orElse(0);
        double[] normData = new double[data.length];
        for (int i = 0; i < data.length; i++) {
            normData[i] = ((data[i] - minEl) * (d2 - d1) / (maxEl - minEl)) + d1;
        }
        return normData;
    }

    public static double[] deNormalizeData(double[] normData) {
        double[] data = new double[normData.length];
        for (int i = 0; i < data.length; i++) {
            data[i] = ((normData[i] - d1) * (maxEl - minEl)) / (d2 - d1) + minEl;
        }
        return data;
    }
}
