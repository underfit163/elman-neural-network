public class NeuralNetwork {

    int entersLen;//N+K
    int hiddenLen;//K
    int outputLen; //M

    int epoch;

    double alpha;

    double[] enters; //x
    double[] hidden;
    double[] output;//y

    double[][] wij; // веса первого слоя
    double[][] wsi; // веса второго слоя
    double[][] trainData;
    double[][] testData;

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

        wij = new double[N + K + 1][K + 1];
        wsi = new double[K + 1][M];

        //1. Присвоить весам случайные начальные значения, имеющие, как правило,
        //равномерное распределение в определенном интервале (например, между -1 и 1).
        wij = randomVals(wij, 0.1, 0.4);
        wsi = randomVals(wsi, 0.1, 0.4);

        //2. Для очередного момента определить состояние всех нейронов
        //сети (сигналы vl и yl). На этой основе можно сформировать входной вектор для произвольного момента t.

        //3. Определить вектор погрешности обучения для нейронов
        //выходного слоя как разность между фактическим и ожидаемым
        //значениями сигналов выходных нейронов.

        //4. Сформировать вектор градиента целевой функции относительно
        //весов выходного и скрытого слоя с использованием формул (137), (140) и (141).

        //5. Уточнить значения весов сети согласно правилам метода наискорейшего спуска:
        // для нейронов выходного слоя сети по формуле (144)
        // для нейронов скрытого слоя сети по формуле (145)

        //После уточнения значений весов перейти к пункту 2 алгоритма для расчета в очередной момент времени.
    }

    /**
     * Прямое распространение
     */
    public void elmanOuter() {
        //рассчитали первый слой
        saveHidden();
        for (int i = 1; i < hidden.length; i++) {//значение порогового элемента меняется при дальнейшем обучении или всегда остается 1?
            hidden[i] = 0;
            for (int j = 0; j < enters.length; j++) {
                hidden[i] += enters[j] * wij[j][i];
            }
            hidden[i] = activationFunction(hidden[i]);
        }

        //расчитали выходной слой
        for (int i = 0; i < output.length; i++) {
            output[i] = 0;
            for (int j = 0; j < hidden.length; j++) {
                output[i] += hidden[j] * wsi[j][i];
            }
            output[i] = activationFunction(output[i]);
        }
    }

    /**
     * Алгоритм наискорейшего спуска
     */
    public void elmanTrain() {
        double[] err = new double[hidden.length];
        double gError = 0;
        while (--epoch > 0) {
            for (int i = 0; i < trainData.length; i++) {
                if (enters.length - hidden.length + 1 - 1 >= 0)
                    System.arraycopy(trainData[i], 0, enters, 1, enters.length - hidden.length + 1 - 1);
                elmanOuter();

                //целевая функция
                double lErr = 0;
                for (int j = 1; j <= output.length; j++) {
                    if (j == trainData[i][trainData[0].length - 1])
                        lErr += Math.pow(output[j - 1] - 1, 2);
                    else {
                        lErr += Math.pow(output[j - 1], 2);
                    }
                }
                lErr *= 0.5;
            }
        }
    }


    public void saveHidden() {
        for (int i = enters.length - hidden.length; i < enters.length; i++) {
            enters[i] = hidden[i - (enters.length - hidden.length)];
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
