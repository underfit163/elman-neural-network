package lab2;

import java.util.Arrays;

public class NeuralNetwork {
    private final int entersLen;//N+K
    private final int hiddenLen;//K
    private final int outputLen; //M
    private int epoch;
    private final double alpha;
    private final double koefMoment;
    private final double[] enters; //x
    private final double[] hidden;
    private final double[] output;//y
    private final double[] gS;
    private final double[] uI;
    private double[][] wij; // веса первого слоя
    private final double[][] wijPrev;
    private double[][] wsi; // веса второго слоя
    private final double[][] wsiPrev;
    private final double[] trainData;

    public NeuralNetwork(int N, int K, int M, double[] trainData, int epoch, double alpha, double koefMoment) {
        this.entersLen = N;
        this.hiddenLen = K;
        this.outputLen = M;
        this.trainData = trainData;
        this.epoch = epoch;
        this.alpha = alpha;
        this.koefMoment = koefMoment;

        enters = new double[N + K + 1]; //+1 это пороговый элемент на вход
        enters[0] = 1; //пороговый элемент

        hidden = new double[K + 1];
        hidden[0] = 1;
        output = new double[M];
        uI = new double[K + 1];
        gS = new double[M];
        uI[0] = 1;
        wij = new double[N + K + 1][K + 1];
        wsi = new double[K + 1][M];

        wijPrev = new double[N + K + 1][K + 1];
        wsiPrev = new double[K + 1][M];
        //1. Присвоить весам случайные начальные значения, имеющие, как правило,
        //равномерное распределение в определенном интервале (например, между -1 и 1).
        wij = randomVals(wij, -0.5, 0.5);
        wsi = randomVals(wsi, -0.5, 0.5);
    }

    /**
     * Прямое распространение.
     * 2. Для очередного момента определить состояние всех нейронов
     * сети (сигналы vl и yl). На этой основе можно сформировать входной вектор для произвольного момента t.
     */
    public double[] elmanOuter(double[] testData) {
        System.arraycopy(testData, 0, enters, 1, enters.length - hidden.length);
        saveHidden();
        //рассчитали скрытый слой
        for (int i = 1; i < hidden.length; i++) {//значение порогового элемента меняется при дальнейшем обучении или всегда остается 1?
            uI[i] = 0;
            for (int j = 0; j < enters.length; j++) {
                uI[i] += enters[j] * wij[j][i];
            }
            hidden[i] = activationFunction(uI[i]);
        }
        //расчитали выходной слой
        for (int s = 0; s < output.length; s++) {
            gS[s] = 0;
            for (int i = 0; i < hidden.length; i++) {
                gS[s] += hidden[i] * wsi[i][s];
            }
            output[s] = activationFunction(gS[s]);
        }
        return output;
    }

    /**
     * Алгоритм наискорейшего спуска
     */
    public void elmanTrain() {
        while (--epoch >= 0) {
            double gError = 0;
            setEntersZero();
            //clearHidden();
            int offset = 0;
            double[] currentTrainData;
            //После уточнения значений весов перейти к пункту 2 алгоритма для расчета в очередной момент времени.
            while (offset <= (trainData.length - (entersLen + 1))) {
                currentTrainData = Arrays.copyOfRange(trainData, offset, offset + entersLen + output.length);
                elmanOuter(currentTrainData);
                //3. Определить вектор погрешности обучения для нейронов
                //выходного слоя как разность между фактическим и ожидаемым
                //значениями сигналов выходных нейронов.
                double[] e = new double[output.length];
                for (int j = 0; j < output.length; j++) {
                    e[j] = output[j] - currentTrainData[currentTrainData.length - 1 - (output.length - 1) + j];
                }
                //Средняя квадратичная ошибка MSE
                double lErr = 0;
                for (int j = 0; j < output.length; j++) {
                    lErr += Math.pow(e[j], 2);
                }
                gError += lErr / output.length;

                if (offset == (trainData.length - 1 - entersLen)) {
                    System.out.println("Средняя квадратичная ошибка MSE: " + (gError) / (offset+1));
                }
                //4. Сформировать вектор градиента целевой функции относительно
                //весов выходного и скрытого слоя с использованием формул (137), (140) и (141).
                double[][] gdv2 = new double[hidden.length][output.length];// какая размерность вектора градиента? (K+1)*M
                for (int i = 0; i < hidden.length; i++) {
                    for (int s = 0; s < output.length; s++) {
                        gdv2[i][s] += e[s] * derivativeActivationFunction(gS[s]) * hidden[i];
                    }
                }
                // Вопрос, почему когда мы когда рассчитываем градиент для скрытого слоя,
                // мы не используем веса между скрытым и входным слоем? Ведь они вносят вклад в ошибку
                double[] dv = new double[hidden.length];// мы от добавочного нейрона берем производную? да //xb - вместе с предыдущими значениями контекстного слоя? да
                for (int i = 1; i < hidden.length; i++) {
                    if (offset != 0) {
                        for (int k = 1; k < hidden.length; k++) {
                            dv[i] += derivativeActivationFunction(enters[k + (enters.length - hidden.length)])
                                    * wij[k + (enters.length - hidden.length)][i];
                        }
                    } else {
                        dv[i] = 0;
                    }
                }

                for (int i = (enters.length - hidden.length + 1); i < enters.length; i++) {
                    dv[i - (enters.length - hidden.length)] =
                            derivativeActivationFunction(uI[i - (enters.length - hidden.length)])
                                    * (enters[i] + dv[i - (enters.length - hidden.length)]);
                }

                double[] gdv1 = new double[hidden.length];
                for (int i = 1; i < hidden.length; i++) {
                    for (int s = 0; s < output.length; s++) {
                        gdv1[i] += dv[i] * wsi[i][s];
                    }
                }
                double[][] gdv1End = new double[enters.length][hidden.length];
                for (int j = 0; j < enters.length; j++) {
                    for (int i = 1; i < hidden.length; i++) {
                        for (int s = 0; s < output.length; s++) {
                            gdv1End[j][i] += e[s] * derivativeActivationFunction(gS[s]) * gdv1[i];
                        }
                    }
                }
                //5. Уточнить значения весов сети согласно правилам метода наискорейшего спуска:
                // для нейронов выходного слоя сети по формуле (144)
                double wDelta;
                //koefMoment = Math.random();
                for (int i = 0; i < hidden.length; i++) {
                    for (int s = 0; s < output.length; s++) {
//                        if (t != 0) {
                        wDelta = (-1 * alpha * gdv2[i][s]) + koefMoment * wsiPrev[i][s];
//                        } else {
//                            wDelta = (-1 * alpha * gdv2[i][s]);
//                        }
                        wsi[i][s] = wsi[i][s] + wDelta;
                        wsiPrev[i][s] = wDelta;
                    }
                }
                //5. Уточнить значения весов сети согласно правилам метода наискорейшего спуска:
                // для нейронов скрытого слоя сети по формуле (145)
                //koefMoment = Math.random();
                for (int j = 0; j < enters.length; j++) {
                    for (int i = 1; i < hidden.length; i++) {
//                        if (t != 0) {
                        wDelta = (-1 * alpha * gdv1End[j][i]) + koefMoment * wijPrev[j][i];
//                        } else {
//                            wDelta = (-1 * alpha * gdv1End[j][i]);
//                        }
                        wij[j][i] = wij[j][i] + wDelta;//коэф рандомим? не желательно
                        wijPrev[j][i] = wDelta;
                    }
                }
                offset++;
            }
        }
    }

    public void saveHidden() {
        for (int i = 1; i < hidden.length; i++) {
            enters[enters.length - hidden.length + i] = hidden[i];
        }
    }

    public void clearHidden() {
        for (int i = 1; i < hidden.length; i++) {
            hidden[i] = 0;
        }
    }

    //Логистическая функция активации
    private double activationFunction(double x) {
        return (1 / (1 + Math.pow(Math.E, (-1 * x))));
    }

    private double derivativeActivationFunction(double x) {
        double f = (1 / (1 + Math.pow(Math.E, (-1 * x))));
        return f * (1 - f);
    }

    //Гипорболический тангенс
//    private double activationFunction(double x) {
//        return (Math.pow(Math.E, x) - Math.pow(Math.E, (-1 * x))) / (Math.pow(Math.E, x) + Math.pow(Math.E, (-1 * x)));
//    }
//
//    private double derivativeActivationFunction(double x) {
//        double f = (Math.pow(Math.E, x) - Math.pow(Math.E, (-1 * x))) / (Math.pow(Math.E, x) + Math.pow(Math.E, (-1 * x)));
//        return 1 - Math.pow(f, 2);
//    }

//    private double activationFunction(double x) {
//        return Math.atan(x);
//    }
//
//    private double derivativeActivationFunction(double x) {
//        return 1.0 / (1.0 + x * x);
//    }

    public void setEntersZero() {
        Arrays.fill(enters, 0);
        enters[0] = 1;
    }

    public double[][] randomVals(double[][] weights, double min, double max) {
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[i].length; j++) {
                weights[i][j] = (Math.random() * (max - min) + min);// от [min, max)
            }
        }
        return weights;
    }
}
