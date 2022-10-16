import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class Main {

    public static void main(String[] args) {
        try {
            String fileName = "iris.data";
            double[][] allData = covertFileToMatrix(fileName, 150, 5);
            System.out.println(Arrays.deepToString(allData));
            double[][] trainData = new double[120][5];
            double[][] testData = new double[30][5];
            int countClass = 3;

            int partLen = allData.length / countClass;
            int testPartLen = testData.length / countClass;
            for (int k = 0; k < countClass; k++) {
                for (int i = k * partLen; i < k * partLen + partLen; i++) {
                    for (int j = 0; j < allData[i].length; j++) {
                        if (i < partLen * (k + 1) - testPartLen)
                            trainData[i - testPartLen * k][j] = allData[i][j];
                        else testData[i - (partLen - testPartLen) * (k + 1)][j] = allData[i][j];
                    }
                }
            }

            System.out.println(Arrays.deepToString(trainData));
            System.out.println(Arrays.deepToString(testData));

            int hiddenLen = (testData[0].length - 1) * 2;
            //double alpha = (double) 1 / (testData.length - 1 + hiddenLen);
            double alpha = 0.5;

            NeuralNetwork neuralNetwork = new NeuralNetwork(4, hiddenLen, 3, trainData, testData, 100, alpha);

            neuralNetwork.elmanTrain();

            for (int i = 0; i < testData.length; i++) {
                System.out.println(Arrays.toString(neuralNetwork.elmanOuter(testData[i])));
            }

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static double[][] covertFileToMatrix(String fileName, int count, int entersLen) throws IOException {
        double[][] allData = new double[count][entersLen];
        ArrayList<String> result = new ArrayList<>();

        BufferedReader bufferedReader = new BufferedReader(new FileReader(fileName));
        String line = bufferedReader.readLine();
        String[] mas;
        int i = 0;
        while (line != null && !line.equals("")) {
            mas = line.replaceAll(" ", "").split(",");
            for (int j = 0; j < mas.length - 1; j++) {
                allData[i][j] = Double.parseDouble(mas[j]);
            }
            if (!result.contains(mas[mas.length - 1])) {
                result.add(mas[mas.length - 1]);
            }
            allData[i][allData[0].length - 1] = (result.indexOf(mas[mas.length - 1]) + 1);
            line = bufferedReader.readLine();
            i++;
        }
        return allData;
    }
}
