public class Neuron {
    private float value;
    private float[] weights;
    private float bias;
    private int layerNum;
    private float[] weightFlags;
    private float biasFlag = 0;
    public Neuron(int previousLayerWidth, boolean random, int l) {
        value = 0;
        weights = new float[previousLayerWidth];
        for (int i = 0;i<previousLayerWidth;++i) {
            weights[i] = (float) Math.random()*2 - 1;
        }
        bias = (float) Math.random()*10-5;
        layerNum = l;
        weightFlags = new float[previousLayerWidth];
    }
    public float setValue() {
        Neuron[] previousLayer = Network.mainNetwork.getPrevLayer(layerNum);
        float ret = bias;
        for (int i = 0;i<previousLayer.length;i++) {
            ret += weights[i]*previousLayer[i].getVal();
        }
        ret = Main.sigmoid(ret);
        value = ret;
        return value;
    }
    public void setValue(float in) {
        value = in;
    }
    public float getVal() {
        return value;
    }
    public float getWeight(int num) {
        return weights[num];
    }
    public void changeWeightFlag(int num, float change) {
        weightFlags[num] += change;
    }
    public void setWeight(int num, float nVal) {
        weights[num] = nVal;
    }
    public float getBias() {
        return bias;
    }
    public void setBias(float newValue) {
        bias = newValue;
    }
    public void changeBiasFlag(float change) {
        biasFlag += change;
    }
    public void updateFlags() {
        for (int w=0;w<weights.length;++w) {
            weights[w] += weightFlags[w];
        }
        bias += biasFlag;
    }
}