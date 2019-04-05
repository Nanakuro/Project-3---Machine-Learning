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
    int num_lay;
    vector<Layer> Layers;
    VectorXd input, output;
    vector<pair<VectorXd, VectorXd>> DataSet;
    vector<VectorXd> InputSet, LabelSet;
    double learn_rate;
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
    void setLearnRate(double eta)   { learn_rate = eta; }
    
    void clearGrad() { for (Layer& L : Layers) { L.clearGrad(); } }
    
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
        assert(input.size() > 0);
        cout << endl;
        for (int i=0; i<num_lay; ++i) {
            Layers[i].print();
        }
        cout << "\nFINAL OUTPUT:\n" << getOutput() << endl;
    }
    
    void printdC() {
        assert(input.size() > 0);
        //cout << endl;
        for (int l=0; l<num_lay; ++l) {
            cout << "LAYER " << l+1 << ":" << endl;
            cout << Layers[l].dCdb << endl << endl;
            cout << Layers[l].dCdW << endl << endl;
        }
        //cout << "\nFINAL OUTPUT:\n" << getOutput() << endl;
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
    
    VectorXd feedForward() {
        assert(input.size() > 0);
        output = input;
        for (int i=0; i<Layers.size(); ++i) {
            output = Layers[i].feedForward(output);
        }
        return output;
    }
    
    VectorXd feedForward(VectorXd& inV) {
        input = inV;
        return feedForward();
    }
    
    double Cost(double(*f)(VectorXd&,VectorXd&)) {
        double C = 0.0;
        for (int i=0; i<InputSet.size(); ++i) {
            feedForward(InputSet[i]);
            C += f(output, LabelSet[i]) / (int)InputSet.size();
        }
        return C;
    }
    
    double Cost() { return Cost(CostCrossEntropy); }
    
    void resetdC() { for (Layer& L : Layers) {L.resetdC();} }
    
    void updateLayers() {
        for (Layer& L : Layers) {
            L.Weights -= (learn_rate * L.dCdW);
            L.bias    -= (learn_rate * L.dCdb);
        }
    }
    
    void backPropagate(int idx_beg, int M, bool update=false) {
        resetdC();
        int max_idx = idx_beg+M;
        while (idx_beg < max_idx) {
            feedForward(InputSet[idx_beg]);
            for (int l=num_lay-1; l>=0; --l) {
                if (l==num_lay-1) {
                    Layers[l].dCdb_temp = dCE_dz(output,LabelSet[idx_beg]).cwiseProduct(Layers[l].getdfdy());
                } else {
                    Layers[l].dCdb_temp = Layers[l].getdfdy().cwiseProduct(
                                            Layers[l+1].Weights.transpose()*Layers[l+1].dCdb_temp);
                }
                Layers[l].dCdb += Layers[l].dCdb_temp / M;
                
                Layers[l].dCdW_temp = Layers[l].dCdb_temp * Layers[l].input.transpose();
                Layers[l].dCdW += Layers[l].dCdW_temp / M;
            }
            ++idx_beg;
        }
        if (update) { updateLayers(); }
    }
    
    void gradDescBP(int epoch, int batch_size) {
        assert(InputSet.size() % batch_size == 0);
        cout << "Epoch: ";
        for (int e=0; e<epoch; ++e) {
            for (int idx=0; idx<InputSet.size(); idx+=batch_size) {
                backPropagate(idx, batch_size, true);
            }
            cout << e+1 << " " << flush;
        }
        cout << endl;
    }
    
    double test(vector<pair<VectorXd,VectorXd>> TestDataSet) {
        int count, success = 0;
        for (count=0; count<TestDataSet.size(); ++count) {
            VectorXd test_input = TestDataSet[count].first,
                     test_label = TestDataSet[count].second;
            feedForward(test_input);
            VectorXd diff = (output - test_label).cwiseAbs();
            for (int i=0; i<diff.size(); ++i) {
                if (diff(i) < 0.5) { continue; }
                --success;
                break;
            }
            ++success;
        }
        double success_rate = 1.0*success/count;
        return success_rate;
    }
    
    
    
    double finDiffWeights(int l, int i, int j) {
        //assert(input.size() > 0);
        double C_i = Cost();
        Layers[l].Weights(i,j) += fin_Delta;
        double C_f = Cost();
        Layers[l].Weights(i,j) -= fin_Delta;
        return (C_f - C_i) / fin_Delta;
    }
    
    double finDiffBias(int l, int i) {
        //assert(input.size() > 0);
        double C_i = Cost();
        Layers[l].bias(i) += fin_Delta;
        double C_f = Cost();
        Layers[l].bias(i) -= fin_Delta;
        return (C_f - C_i) / fin_Delta;
    }
    
    void gradDescfinDiff(bool update=false) {
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
        if (update) { updateLayers(); }
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
