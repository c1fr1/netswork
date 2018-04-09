public class Network {
    public Neuron[] inputLayer;
    public Neuron[][] middleLayers;
    public Neuron[] outputLayer;
    public static Network mainNetwork;
    public Network(int inNeurons, int outNeurons, int numLayers, int layerSize) {
        inputLayer = new Neuron[inNeurons];
        middleLayers = new Neuron[numLayers][layerSize];
        for (int i = 0;i<inNeurons;++i) {
            inputLayer[i] = new Neuron(0, true, -1);
        }
        int layerCount = inNeurons;
        for (int l = 0;l<numLayers;++l) {
            middleLayers[l] = new Neuron[layerSize];
            for (int i = 0; i < layerSize; ++i) {
                middleLayers[l][i] = new Neuron(layerCount, true, l);
            }
            layerCount = layerSize;
        }
        outputLayer = new Neuron[outNeurons];
        for (int i = 0;i<outNeurons;++i) {
            outputLayer[i] = new Neuron(layerSize, true, numLayers);
        }
        mainNetwork = this;
    }
    public float[] run(float[] in) {
        for (int i = 0;i<in.length;++i) {
            inputLayer[i].setValue(in[i]);
        }
        for (Neuron[] na:middleLayers) {
            for (Neuron n:na) {
                n.setValue();
            }
        }
        float[] out = new float[outputLayer.length];
        for (int n=0;n<outputLayer.length;++n) {
            out[n] = outputLayer[n].setValue();
        }
        return out;
    }
    public static Network getMainNetwork() {
        return mainNetwork;
    }
    public float getCost(float[] expected) {
        float ret = 0;
        for (int i = 0;i<expected.length;++i) {
            ret += (outputLayer[i].getVal()-expected[i])*(outputLayer[i].getVal()-expected[i]);
        }
        return ret;
    }
    public Neuron[] getPrevLayer(int forNum) {
        if (forNum == 0) {
            return inputLayer;
        }else {
            return middleLayers[forNum - 1];
        }
    }
    public Neuron[] getLayer(int forNum) {
        if (forNum == -1) {
            return inputLayer;
        }else {
            return middleLayers[forNum];
        }
    }
    public void backPropogate(float[] targetValues) {
        float[] biasConstants = new float[outputLayer.length];
        Neuron[] pl;
        for (int i = 0;i<outputLayer.length;i++) {
            float interior = 0;
            for (int n=0; n < middleLayers[middleLayers.length-1].length;n++) {
                interior += middleLayers[middleLayers.length-1][n].getVal()*outputLayer[i].getWeight(n);
            }
            interior += outputLayer[i].getBias();
            float biasConstant = 2*(Main.sigmoid(interior) - targetValues[i])*(Main.sigmoidPrime(interior));
            biasConstants[i] = biasConstant;
            outputLayer[i].changeBiasFlag(-biasConstant);
            for (int n=0; n < middleLayers[middleLayers.length-1].length;n++) {
                outputLayer[i].changeWeightFlag(n, -biasConstant*middleLayers[middleLayers.length-1][n].getVal());
            }
        }
        pl = outputLayer;
        for (int lNum = middleLayers.length-1;lNum>0;--lNum){
            float[] tbiasConstants = new float[middleLayers[lNum-1].length];
            for (int n = 0; n<middleLayers[lNum].length;n++) {
                float interior = 0;
                for (int nn = 0; nn<middleLayers[lNum-1].length;++nn) {
                    interior += middleLayers[lNum-1][nn].getVal()*middleLayers[lNum][n].getWeight(nn);
                }
                interior += middleLayers[lNum][n].getBias();
                float biasConstant = 0;
                for (int bc = 0;bc < biasConstants.length;++bc) {
                    biasConstant += biasConstants[bc] * pl[bc].getWeight(n);
                }
                biasConstant *= Main.sigmoidPrime(interior);
                tbiasConstants[n] = biasConstant;
                middleLayers[lNum][n].changeBiasFlag(-biasConstant);
                for (int nn = 0; nn < middleLayers[lNum - 1].length; ++nn) {//-1
                    middleLayers[lNum][n].changeWeightFlag(nn, -biasConstant * middleLayers[lNum - 1][nn].getVal());//-1
                }
            }
            pl = middleLayers[lNum];
            biasConstants = tbiasConstants;
        }
        for (int n = 0; n<middleLayers[0].length;n++) {
            float interior = 0;
            for (int nn = 0; nn<inputLayer.length;++nn) {
                interior += inputLayer[nn].getVal()*middleLayers[0][n].getWeight(nn);
            }
            interior += middleLayers[0][n].getBias();
            float biasConstant = 0;
            for (int bc = 0;bc < biasConstants.length;++bc) {
                biasConstant += biasConstants[bc] * pl[bc].getWeight(n);
            }
            biasConstant *= Main.sigmoidPrime(interior);
            middleLayers[0][n].changeBiasFlag(-biasConstant);
            for (int nn = 0;nn < inputLayer.length;++nn) {
                middleLayers[0][n].changeWeightFlag(nn, -biasConstant*inputLayer[nn].getVal());
            }
        }
    }
    public void updateFlags() {
        for (Neuron n:inputLayer) {
            n.updateFlags();
        }
        for (Neuron[] ns:middleLayers) {
            for (Neuron n:ns) {
                n.updateFlags();
            }
        }
        for (Neuron n: outputLayer) {
            n.updateFlags();
        }
    }
}
