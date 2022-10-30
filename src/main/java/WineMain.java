import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class WineMain {
        public static void main(String[] args) {
            try {
                String fileName = "wine.data";
                double[][] allData = covertFileToMatrix(fileName, 178, 14);

                //System.out.println("Все данные до нормализации:");
                //System.out.println(Arrays.deepToString(allData));
                //normalizeData(allData);
                //System.out.println("Все данные после нормализации:");
                //System.out.println(Arrays.deepToString(allData));

                double[][] trainData = new double[148][14];
                double[][] testData = new double[30][14];
                int countClass = 3;

//                System.arraycopy(allData, 0, trainData, 0, 49);
//                System.arraycopy(allData, 59, trainData, 49, 61);
//                System.arraycopy(allData, 130, trainData, 110, 38);

                System.arraycopy(allData, 0, trainData, 0, 38);
                System.arraycopy(allData, 59, trainData, 49, 38);
                System.arraycopy(allData, 130, trainData, 110, 38);

                System.arraycopy(allData, 49, testData, 0, 10);
                System.arraycopy(allData, 120, testData, 10, 10);
                System.arraycopy(allData, 168, testData, 20, 10);


                //System.out.println("Тренировочные данные:");
                //System.out.println(Arrays.deepToString(trainData));

                shuffleMatrix(trainData);
                //System.out.println("Тренировочные данные после шафла:");
                //System.out.println(Arrays.deepToString(trainData));

                //System.out.println("Тестовые данные:");
                //System.out.println(Arrays.deepToString(testData));

                //int hiddenLen = (testData[0].length - 1) * 2;
                int hiddenLen = 30;
                //double alpha = (double) 1 / (testData.length - 1 + hiddenLen);
                double alpha = 0.1;
                double koefMoment = 0.05;
                int epoch = 100;

                NeuralNetwork neuralNetwork = new NeuralNetwork
                        (13, hiddenLen, countClass, trainData, epoch, alpha, koefMoment);

                neuralNetwork.elmanTrain();

                neuralNetwork.setEntersZero();
                System.out.println("Выходной слой:");
                int cl = 1;
                int countTrue = 0;
                int maxI = 0;
                for (int i = 0; i < testData.length; i++) {
                    if (i % 10 == 0) {
                        System.out.println("Данные класса " + cl + ":");
                        cl++;
                    }
                    double[] arr = neuralNetwork.elmanOuter(testData[i]);
                    double max = Double.MIN_VALUE;
                    for (int j = 0; j < arr.length; j++) {
                        if (arr[j] > max) {
                            max = arr[j];
                            maxI = j;
                        }
                    }
                    if (maxI + 1 == testData[i][testData[0].length - 1]) {
                        countTrue++;
                    }
                    System.out.println(Arrays.toString(arr));
                }
                System.out.println("Правильно предсказаных ответов: " + countTrue + " из " + testData.length);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        public static double[][] covertFileToMatrix(String fileName, int count, int entersLen) throws IOException {
            double[][] allData = new double[count][entersLen];

            BufferedReader bufferedReader = new BufferedReader(new FileReader(fileName));
            String line = bufferedReader.readLine();
            String[] mas;
            int i = 0;
            while (line != null && !line.equals("")) {
                mas = line.replaceAll(" ", "").split(",");
                for (int j = 0; j < mas.length - 1; j++) {
                    allData[i][j] = Double.parseDouble(mas[j+1]);
                }
                allData[i][mas.length-1] = Double.parseDouble(mas[0]);
                line = bufferedReader.readLine();
                i++;
            }
            return allData;
        }

        public static void shuffleMatrix(double[][] matrix) {
            List<double[]> rows = new ArrayList<>(Arrays.asList(matrix));
            Collections.shuffle(rows);
            for (int i = 0; i < matrix.length; i++) {
                matrix[i] = rows.get(i);
            }
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
