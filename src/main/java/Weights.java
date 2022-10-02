public class Weights {
    private int rows;
    private int cols;
    private double[][] matrix;


    public Weights(int Rows, int Cols) {
        rows = Rows;
        cols = Cols;
        matrix = new double[rows][cols];
        randomVals(matrix,0, 0.9);
    }

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public double getValue(int Row, int Col) {
        return matrix[Row][Col];
    }

    public void setValue(int Row, int Col, double Value) {
        matrix[Row][Col] = Value;
    }

    public double[][] getMatrix() {
        return matrix;
    }

    public void setMatrix(double[][] matrix) {
        this.matrix = matrix;
    }


    public void randomVals(double[][] weights, double min, double max) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                weights[i][j] = (Math.random() * (max - min) + min)/10;
            }
        }
    }
}
