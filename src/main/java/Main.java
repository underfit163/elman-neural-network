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
            int partLen = trainData.length / 3;
            for (int i = 0; i < partLen; i++) {
                for (int j = 0; j < trainData[i].length; j++) {
                    trainData[i][j] = allData[i][j];
                    trainData[i + partLen][j] = allData[i + (partLen + testData.length / 3)][j];
                    trainData[i + 2 * partLen][j] = allData[i + 2 * (partLen + testData.length/3)][j];
                }
            }
            System.out.println(Arrays.deepToString(trainData));
            for (int i = 0; i < testData.length / 3; i++) {
                for (int j = 0; j < trainData[i].length; j++) {
                    testData[i][j] = allData[i + partLen][j];
                    testData[i + testData.length / 3][j] = allData[i + (testData.length / 3 + 2 * partLen)][j];
                    testData[i + 2 * testData.length / 3][j] = allData[i + (2 * testData.length / 3 + 3 * partLen)][j];
                }
            }
            System.out.println(Arrays.deepToString(testData));

            int hiddenLen = testData.length * 2;
            double alpha = (double) 1/(testData.length-1 + hiddenLen);


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
