public class Layer {
    private double[] arr;

    public Layer(int size) {
        arr = new double[size];
        for (int i = 0; i < size; i++) {
            arr[i] = Math.random();
        }
    }

    public int getSize() {
        return arr.length;
    }

    public double getVal(int i) {
        return arr[i];
    }

    public void setVal(int i, double val) {
        arr[i] = val;
    }

    public double[] getArr() {
        return arr;
    }

    public void setArr(double[] arr) {
        this.arr = arr;
    }
}
