//
//  Layer.h
//  
//
//  Created by Minh on 03/26/19.
//

#ifndef Layer_h
#define Layer_h

#include "misc.h"
#include <Eigen/Dense>
using namespace Eigen;

double sigmoid(double x) {
    return 1.0/(1.0 + exp(-x));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Layer {
    static int num_layer;
    int lay, num_input, num_output;
    MatrixXd Weights;
    VectorXd bias;
public:
    VectorXd input, output;
    
    Layer();
    Layer(int, int);
    
    int layer()             { return lay; }
    int inSize()            { return num_input; }
    int outSize()           { return num_output; }
    MatrixXd getWeights()   { return Weights; }
    VectorXd getBias()      { return bias; }
    
    VectorXd evaluate() {
        output = Weights * input;
        output += bias;
        output = output.unaryExpr(&sigmoid);
        return output;
    }
    
    VectorXd evaluate(VectorXd inV) {
        output = Weights * inV;
        output += bias;
        output = output.unaryExpr(&sigmoid);
        return output;
    }
};

int Layer::num_layer = 0;

Layer::Layer() {
    lay = ++num_layer;
    num_input  = 3;
    num_output = 3;
    uniform_real_distribution<double> rand(-1,1);
    Weights = MatrixXd::NullaryExpr(num_output, num_input, [&]() { return rand(mt); });
    bias    = VectorXd::NullaryExpr(num_output, [&]() { return rand(mt); });
}

Layer::Layer(int n_in, int n_out) {
    lay = ++num_layer;
    num_input = n_in;
    num_output = n_out;
    uniform_real_distribution<double> rand(-1,1);
    Weights = MatrixXd::NullaryExpr(num_output, num_input, [&]() { return rand(mt); });
    bias    = VectorXd::NullaryExpr(num_output, [&]() { return rand(mt); });
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class FFNN {
    int num_lay;
    vector<Layer> Layers;
    VectorXd input, output;
public:
    FFNN(int);
    FFNN(vector<int>);
    
    VectorXd evaluate() {
        output = input.replicate<1,1>();
        for (int i=0; i<Layers.size(); ++i) {
        }
        return output;
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
    num_lay = (int)layer_neurons.size() - 1;
    Layers.resize(num_lay);
    for (int i=0; i<num_lay; ++i) {
        int n_in  = layer_neurons[i],
            n_out = layer_neurons[i+1];
        Layers[i] = Layer(n_in, n_out);
    }
}

#endif /* Layer_h */
