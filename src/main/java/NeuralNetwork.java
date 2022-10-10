
public class NeuralNetwork {

    private int entersLen;//N+K
    private int hiddenLen;//K
    private int outputLen; //M

    private int epoch;

    private double alpha;

    private double[] enters; //x
    private double[] hidden;

    private double[] output;//y
    private double[] gS;
    private double[] uI;

    private double[][] wij; // веса первого слоя
    private double[][] wsi; // веса второго слоя
    private double[][] trainData;
    private double[][] testData;

    public NeuralNetwork(int N, int K, int M, double[][] trainData, double[][] testData, int epoch, double alpha) {
        this.entersLen = N;
        this.hiddenLen = K;
        this.outputLen = M;
        this.trainData = trainData;
        this.testData = testData;
        this.epoch = epoch;
        this.alpha = alpha;

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

        //1. Присвоить весам случайные начальные значения, имеющие, как правило,
        //равномерное распределение в определенном интервале (например, между -1 и 1).
        wij = randomVals(wij, 0.1, 0.4);
        wsi = randomVals(wsi, 0.1, 0.4);
    }

    /**
     * Прямое распространение.
     * 2. Для очередного момента определить состояние всех нейронов
     * сети (сигналы vl и yl). На этой основе можно сформировать входной вектор для произвольного момента t.
     */
    public void elmanOuter() {
        //рассчитали первый слой
        saveHidden();
        for (int i = 1; i < hidden.length; i++) {//значение порогового элемента меняется при дальнейшем обучении или всегда остается 1?
            uI[i] = 0;
            for (int j = 0; j < enters.length; j++) {
                uI[i] += enters[j] * wij[j][i];
            }
            hidden[i] = activationFunction(uI[i]);
        }

        //расчитали выходной слой
        for (int i = 0; i < output.length; i++) {
            gS[i] = 0;
            for (int j = 0; j < hidden.length; j++) {
                gS[i] += hidden[j] * wsi[j][i];
            }
            output[i] = activationFunction(gS[i]);
        }
    }

    /**
     * Алгоритм наискорейшего спуска
     */
    public void elmanTrain() {
        double gError = 0;
        while (--epoch > 0) {
            //После уточнения значений весов перейти к пункту 2 алгоритма для расчета в очередной момент времени.
            for (int t = 0; t < trainData.length; t++) {
                System.arraycopy(trainData[t], 0, enters, 1, enters.length - hidden.length);
                elmanOuter();

                //3. Определить вектор погрешности обучения для нейронов
                //выходного слоя как разность между фактическим и ожидаемым
                //значениями сигналов выходных нейронов.

                double[] e = new double[output.length];
                for (int j = 0; j < output.length; j++) {
                    if (j + 1 == trainData[t][trainData[0].length - 1])
                        e[j] = output[j] - 1;
                    else {
                        e[j] = output[j];
                    }
                }
                //целевая функция
                double lErr = 0;
                for (int j = 0; j < output.length; j++) {
                    lErr += Math.pow(e[j], 2);
                }
                lErr *= 0.5;
                gError += Math.abs(lErr);
                System.out.println("Глобальная ошибка: " + gError);

                //4. Сформировать вектор градиента целевой функции относительно
                //весов выходного и скрытого слоя с использованием формул (137), (140) и (141).
                //5. Уточнить значения весов сети согласно правилам метода наискорейшего спуска:
                // для нейронов выходного слоя сети по формуле (144)
                /*double gradSI = 0;
                for (int s = 0; s < output.length; s++) {
                    for (int i = 0; i < hidden.length; i++) {
                        gradSI += e[s] * derivativeActivationFunction(gS[s]) * hidden[i];
                    }
                }
                for (int s = 0; s < output.length; s++) {
                    for (int i = 0; i < hidden.length; i++) {
                        wsi[i][s] = wsi[i][s] - alpha * gradSI;
                    }
                }*/

                double[] gdv2 = new double[hidden.length];
                for (int i = 0; i < hidden.length; i++) {
                    for (int s = 0; s < output.length; s++) {
                        gdv2[i] += e[s] * derivativeActivationFunction(gS[s]) * hidden[i];
                    }
                }


                for (int i = 0; i < hidden.length; i++) {
                    for (int s = 0; s < output.length; s++) {
                        wsi[i][s] = wsi[i][s] - alpha * gdv2[i];
                    }
                }

                //5. Уточнить значения весов сети согласно правилам метода наискорейшего спуска:
                // для нейронов скрытого слоя сети по формуле (145)
                double[] dv = new double[hidden.length];// мы от добавочного нейрона берем производную? да //xb - вместе с предыдущими значениями контекстного слоя? да
                for (int k = 0; k < hidden.length; k++) {
                    if (t != 0) {
                        for (int i = 1; i < hidden.length; i++) {
                            dv[k] += derivativeActivationFunction(enters[k + (enters.length - hidden.length)])
                                    * wij[k + (enters.length - hidden.length)][i];
                        }
                    }
                }

                for (int i = (enters.length - hidden.length+1); i < enters.length; i++) {
                    dv[i - (enters.length - hidden.length)] =
                            derivativeActivationFunction(uI[i - (enters.length - hidden.length)])
                                    * (enters[i] + dv[i - (enters.length - hidden.length)]);
                }


//                int delta;
//                for (int i = 0; i < hidden.length; i++) {
//                    for (int b = 0; b < enters.length; b++) {
//                        if (b == i + 1 + (enters.length - hidden.length)) //  i - это индекс нейрона скрытого слоя [1, K]
//                            delta = 1;
//                        else {
//                            delta = 0;
//                        }
//                        dv[i] = delta * enters[b];
//                    }
//                }
                double[] gdv1 = new double[hidden.length];
                for (int i = 0; i < hidden.length; i++) {
                    for (int s = 0; s < output.length; s++) {
                        gdv1[i] += e[s] * dv[i] * wsi[i][s];
                    }
                }

                for (int j = 0; j < enters.length; j++) {
                    for (int i = 1; i < hidden.length; i++) {
                        wij[j][i] = wij[j][i] - alpha * gdv1[i];
                    }
                }
            }
        }
    }


    public void saveHidden() {
        for (int i = 1; i < hidden.length; i++) {
          enters[enters.length - hidden.length + i] = hidden[i];
        }
    }

    private double activationFunction(double x) {
        return (1.0 / (1.0 + Math.exp(-1.0 * (x))));
    }

    private double derivativeActivationFunction(double x) {
        double f = 1.0 / (1.0 + Math.exp(-1.0 * (x)));
        return f * (1 - f);
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
