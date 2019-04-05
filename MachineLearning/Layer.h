//
//  Layer.h
//  
//
//  Created by Minh on 03/26/19.
//

#ifndef Layer_h
#define Layer_h

#include "misc.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Layer {
    static int num_layer;
    int lay, num_input, num_output;
    VectorXd dfdy;
public:
    MatrixXd Weights, dCdW, dCdW_temp;
    VectorXd bias, dCdb, dCdb_temp;
    VectorXd input, output;
    double (*f)(double);
    
    Layer();
    Layer(int, int);
    Layer(int, int, double (*F)(double));
    
    void setFunc(double (*func)(double))        { f = func; }
    void setLayer(int l)                        { lay = l; }
    void setdfdy()                              { dfdy = output.cwiseProduct(VectorXd::Ones(output.size())-output);}
    VectorXd getdfdy()                          { return dfdy; }
    int layer()                                 { return lay; }
    int inSize()                                { return num_input; }
    int outSize()                               { return num_output; }
    
    void print() {
        cout << "LAYER " << lay << ":" << endl
             << "Input:\n" << input << endl
             << "\nWeights:\n" << Weights << endl
             << "\nBias:\n" << bias << endl << endl;
    }
    
    VectorXd feedForward() {
        assert(input.size() != 0);
        output = Weights*input + bias;
        output = output.unaryExpr(f);
        setdfdy();
        return output;
    }
    
    VectorXd feedForward(VectorXd inV) {
        assert(inV.size() != 0);
        input = inV;
        return feedForward();
    }
    
    void resetdC() {
        dCdW = MatrixXd::Zero(Weights.rows(), Weights.cols());
        dCdb = VectorXd::Zero(bias.size());
    }
    
    void clearGrad() {
        dCdW *= 0; dCdW_temp *= 0;
        dCdb *= 0; dCdb_temp *= 0;
        input *= 0; output *= 0;
        dfdy *= 0;
    }
};

int Layer::num_layer = 0;

Layer::Layer() {
    lay = num_layer++;
    f = &sigmoid;
    num_input  = 2;
    num_output = num_input;
    uniform_real_distribution<double> rand(-0.01,0.01);
    Weights = MatrixXd::NullaryExpr(num_output, num_input, [&]() { return rand(mt); });
    bias    = VectorXd::NullaryExpr(num_output,            [&]() { return rand(mt); });
}

Layer::Layer(int n_in, int n_out) {
    lay = num_layer++;
    f = &sigmoid;
    num_input = n_in;
    num_output = n_out;
    uniform_real_distribution<double> rand(-0.01,0.01);
    Weights = MatrixXd::NullaryExpr(num_output, num_input, [&]() { return rand(mt); });
    bias    = VectorXd::NullaryExpr(num_output,            [&]() { return rand(mt); });
}

Layer::Layer(int n_in, int n_out, double (*func)(double)) {
    lay = num_layer++;
    f = func;
    num_input = n_in;
    num_output = n_out;
    uniform_real_distribution<double> rand(-0.01,0.01);
    Weights = MatrixXd::NullaryExpr(num_output, num_input, [&]() { return rand(mt); });
    bias    = VectorXd::NullaryExpr(num_output,            [&]() { return rand(mt); });
}


#endif /* Layer_h */
