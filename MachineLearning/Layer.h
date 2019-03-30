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
    MatrixXd Weights, dCdW;
    VectorXd bias, dCdb;
    VectorXd input, output;
    double (*f)(double);
    
    Layer();
    Layer(int, int);
    Layer(int, int, double (*F)(double));
    
    void setFunc(double (*func)(double))        { f = func; }
    void setLayer(int l)                        { lay = l; }
    void setdfdy()                              { dfdy = output.cwiseProduct(VectorXd::Ones(output.size())-output);}
    int layer()                                 { return lay; }
    int inSize()                                { return num_input; }
    int outSize()                               { return num_output; }
    
    void print() {
        cout << "LAYER " << lay << ":" << endl
             << "Input:\n" << input << endl
             << "\nWeights:\n" << Weights << endl
             << "\nBias:\n" << bias << endl << endl;
    }
    
    VectorXd evaluate() {
        assert(input.size() != 0);
        output = Weights*input + bias;
        output = output.unaryExpr(f);
        setdfdy();
        return output;
    }
    
    VectorXd evaluate(VectorXd inV) {
        assert(inV.size() != 0);
        input = inV;
        evaluate();
    }
    
};

int Layer::num_layer = 0;

Layer::Layer() {
    lay = num_layer++;
    f = &sigmoid;
    num_input  = 2;
    num_output = num_input;
    uniform_real_distribution<double> rand(-1,1);
    Weights = MatrixXd::NullaryExpr(num_output, num_input, [&]() { return rand(mt); });
    bias    = VectorXd::NullaryExpr(num_output,            [&]() { return rand(mt); });
}

Layer::Layer(int n_in, int n_out) {
    lay = num_layer++;
    f = &sigmoid;
    num_input = n_in;
    num_output = n_out;
    uniform_real_distribution<double> rand(-1,1);
    Weights = MatrixXd::NullaryExpr(num_output, num_input, [&]() { return rand(mt); });
    bias    = VectorXd::NullaryExpr(num_output,            [&]() { return rand(mt); });
}

Layer::Layer(int n_in, int n_out, double (*func)(double)) {
    lay = num_layer++;
    f = func;
    num_input = n_in;
    num_output = n_out;
    uniform_real_distribution<double> rand(-1,1);
    Weights = MatrixXd::NullaryExpr(num_output, num_input, [&]() { return rand(mt); });
    bias    = VectorXd::NullaryExpr(num_output,            [&]() { return rand(mt); });
}


#endif /* Layer_h */
