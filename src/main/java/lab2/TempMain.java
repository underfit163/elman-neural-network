package lab2;

import java.io.*;
import java.util.Arrays;

public class TempMain {
    public static void main(String[] args) {
        String fileName = "temp.data.txt";
        int countNum = 299;
        double[] allData = readData(fileName, countNum);
        System.out.println(Arrays.toString(allData));
        allData = normalizeData(allData);
        System.out.println(Arrays.toString(allData));

        double[] trainData = new double[289];
        double[] testData = new double[15];

        System.arraycopy(allData, 0, trainData, 0, 289);
        System.arraycopy(allData, 284, testData, 0, 15);

        int hiddenLen = 10;
        double alpha = 0.2;
        double koefMoment = 0.05;
        int epoch = 100;
        int countOut = 1;
        int window = 3;

        NeuralNetwork neuralNetwork = new NeuralNetwork
                (window, hiddenLen, countOut, trainData, epoch, alpha, koefMoment);

        neuralNetwork.elmanTrain();
        neuralNetwork.setEntersZero();
        //neuralNetwork.clearHidden();
        System.out.println("Выходной слой:");
        int offset = 0;
        double gError = 0;
        double[] currentTestData;
        while (offset <= (testData.length - (window + 1))) {
            currentTestData = Arrays.copyOfRange(testData, offset, offset + window + countOut);
            double[] output = neuralNetwork.elmanOuter(testData);
            double[] e = new double[output.length];
            for (int j = 0; j < output.length; j++) {
                e[j] = output[j] - currentTestData[currentTestData.length - 1 - (output.length - 1) + j];
            }
            //Средняя квадратичная ошибка MSE
            double lErr = 0;
            for (int j = 0; j < output.length; j++) {
                lErr += Math.pow(e[j], 2);
            }
            gError += lErr / output.length;

            System.out.println(Arrays.toString(deNormalizeData(output)));
            if (offset == (testData.length - 1 - window)) {
                System.out.println("Средняя квадратичная ошибка MSE: " + (gError) / (offset+1));
            }
            offset++;
        }


    }

    private static double[] readData(String fileName, int countNum) {
        double[] allData = new double[countNum];
        try (BufferedReader bufferedReader = new BufferedReader(new FileReader(fileName));
        PrintWriter printWriter = new PrintWriter("temp.data.refactor")) {
            for (int i = 0; i < countNum; i++) {
                String[] s = bufferedReader.readLine().split(",");
                allData[i] = Double.parseDouble(s[1]);
                printWriter.println(s[0].replaceAll("\"",""));
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        return allData;
    }
    private static final double d2 = 1;
    private static final double d1 = 0;
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
