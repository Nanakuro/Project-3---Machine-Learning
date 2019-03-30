//
//  FFNN.h
//  
//
//  Created by Minh on 03/27/19.
//

#ifndef FFNN_h
#define FFNN_h

#include "Layer.h"
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FFNN {
    double learn_rate;
    int num_lay;
    vector<Layer> Layers;
    VectorXd input, output;
    vector<pair<VectorXd, VectorXd>> DataSet;
    vector<VectorXd> InputSet;
    vector<VectorXd> LabelSet;
public:
    
    FFNN(int);
    FFNN(vector<int>);
    FFNN(vector<int>, vector<pair<VectorXd, VectorXd>>);

    void setInput(VectorXd v)       { input = v; }
    int getNumLayers()              { return num_lay; }
    VectorXd getInput()             { return input; }
    VectorXd getOutput()            { return output; }
    vector<Layer> getLayers()       { return Layers; }
    double getLearnRate()           { return learn_rate; }
    void setLearnRate(double nu)    { learn_rate = nu; }
    
    void setF(double (*func)(double)) {
        for (int i=0; i<Layers.size(); ++i) {
            Layers[i].setFunc(func);
        }
    }
    
    void printWeights() {
        for (int i=0; i<num_lay; ++i) {
            cout << "LAYER " << i+1 << ":" << endl
                 << Layers[i].Weights << endl << endl;
        }
    }
    
    void printSetup() {
        cout << endl;
        for (int i=0; i<num_lay; ++i) {
            cout << "LAYER "     << Layers[i].layer() << ":" << endl
                 << "Weights:\n" << Layers[i].Weights << endl
                 << "\nBias:\n"  << Layers[i].bias << endl << endl;
        }
    }
    
    void print() {
        assert(input.size() != 0);
        cout << endl;
        for (int i=0; i<num_lay; ++i) {
            Layers[i].print();
        }
        cout << "\nFINAL OUTPUT:\n" << getOutput() << endl;
    }
    
    void insertLayer(int l) {
        assert(l > 0 && l < num_lay);
        num_lay++;
        int n_in  = Layers[l-1].outSize(),
            n_out = Layers[l].inSize();
        Layer L(n_in, n_out, Layers[l].f);
        Layers.insert(Layers.begin()+l, L);
        for (int i=l; i<num_lay; ++i) {
            Layers[i].setLayer(i);
        }
    }
    
    VectorXd evaluate() {
        assert(input.size() > 0);
        output = input;
        for (int i=0; i<Layers.size(); ++i) {
            output = Layers[i].evaluate(output);
        }
        return output;
    }
    
    VectorXd evaluate(VectorXd inV) {
        input = inV;
        evaluate();
        return output;
    }
    
    double Cost(double(*f)(VectorXd&,VectorXd&)) {
        double C = 0.0;
        for (int i=0; i<InputSet.size(); ++i) {
            evaluate(InputSet[i]);
            C += f(output, LabelSet[i]) / (int)InputSet.size();
        }
        return C;
    }
    
    double Cost() { return Cost(CostCrossEntropy); }
    
    void updateLayers() {
        for (Layer& L : Layers) {
            L.Weights -= (learn_rate * L.dCdW);
            L.bias    -= (learn_rate * L.dCdb);
        }
    }
    
    
    
    double finDiffWeights(int l, int i, int j) {
        assert(input.size() > 0);
        double C_i = Cost();
        Layers[l].Weights(i,j) += delta;
        double C_f = Cost();
        Layers[l].Weights(i,j) -= delta;
        return (C_f - C_i) / delta;
    }
    
    double finDiffBias(int l, int i) {
        assert(input.size() > 0);
        double C_i = Cost();
        Layers[l].bias(i) += delta;
        double C_f = Cost();
        Layers[l].bias(i) -= delta;
        return (C_f - C_i) / delta;
    }
    
    void gradDescfinDiff() {
        for (int l=0; l<num_lay; ++l) {
            VectorXd dCdb_Lay(Layers[l].bias.size());
            MatrixXd dCdW_Lay(Layers[l].Weights.rows(), Layers[l].Weights.cols());
            
            assert(dCdb_Lay.size() == dCdW_Lay.rows());
            
            for (int i=0; i<Layers[l].Weights.rows(); ++i) {
                dCdb_Lay(i) = finDiffBias(l,i);
                for (int j=0; j<Layers[l].Weights.cols(); ++j) {
                    dCdW_Lay(i,j) = finDiffWeights(l, i, j);
                }
            }
            Layers[l].dCdW = dCdW_Lay;
            Layers[l].dCdb = dCdb_Lay;
        }
        updateLayers();
    }
};

FFNN::FFNN(int num_layers) {
    num_lay = num_layers;
    Layers.resize(num_lay);
    for (int i=0; i<num_lay; ++i) {
        Layers[i] = Layer();
    }
}

FFNN::FFNN(vector<int> layer_neurons) {
    assert(layer_neurons.size() > 1);
    num_lay = (int)layer_neurons.size() - 1;
    Layers.resize(num_lay);
    for (int i=0; i<num_lay; ++i) {
        int n_in  = layer_neurons[i],
            n_out = layer_neurons[i+1];
        Layers[i] = Layer(n_in, n_out);
    }
}

FFNN::FFNN(vector<int> layer_neurons, vector<pair<VectorXd, VectorXd>> Data) {
    assert(layer_neurons.size() > 1);
    DataSet = Data;
    for (int i=0; i<DataSet.size(); ++i) {
        InputSet.push_back(DataSet[i].first);
        LabelSet.push_back(DataSet[i].second);
    }
    num_lay = (int)layer_neurons.size()-1;
    Layers.resize(num_lay);
    for (int i=0; i<num_lay; ++i) {
        int n_in  = layer_neurons[i],
            n_out = layer_neurons[i+1];
        Layers[i] = Layer(n_in, n_out);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif /* FFNN_h */
