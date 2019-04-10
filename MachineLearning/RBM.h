//
//  RBM.h
//  
//
//  Created by Minh on 04/04/19.
//

#ifndef RBM_h
#define RBM_h

#include "misc.h"

class RBM {
    VectorXd vis_temp, hid_temp;
    int totVisState, totHidState;
public:
    VectorXd visible, vis_bias;
    VectorXd hidden, hid_bias;
    MatrixXd Weights;
    
    RBM(int,int);
    
    VectorXd hid_eff_field() { return Weights.transpose()*visible + hid_bias; }
    VectorXd vis_eff_field() { return Weights*hidden + vis_bias; }
    
    VectorXd ph_v_vec() { return hid_eff_field().unaryExpr( [](double x){return 1.0/(1.0+exp(-2*x));} ); }
    VectorXd pv_h_vec() { return vis_eff_field().unaryExpr( [](double x){return 1.0/(1.0+exp(-2*x));} ); }
    
    void setNewTemp(VectorXd vis_new, VectorXd hid_new) { vis_temp = vis_new; hid_temp = hid_new; }
    void save()     { setNewTemp(visible, hidden); }
    void resetV()   { visible = vis_temp; }
    void resetH()   { hidden = hid_temp; }
    void reset()    { resetV(); resetH(); }
    
    void binToNeurons(string binary, string layer) {
        if (layer=="v") {
            assert(binary.size() == visible.size());
            for (string::size_type i=0; i<binary.size(); ++i) { visible(i) = (binary[i]=='1') ? 1.0 : -1.0; }
        } else if (layer=="h") {
            assert(binary.size() == hidden.size());
            for (string::size_type i=0; i<binary.size(); ++i) { hidden(i) = (binary[i]=='1') ? 1.0 : -1.0; }
        } else {
            throw invalid_argument("Invalid mode (\"v\" or \"h\" only)");
        }
    }
    
    void setVisState(int v) { binToNeurons(zfill(intToBin(v),(int)visible.size()), "v"); }
    void setHidState(int h) { binToNeurons(zfill(intToBin(h),(int)hidden.size()), "h"); }
    
    string getVisBinary() {
        string bin = "";
        for (int v=0; v<visible.size(); ++v) {
            bin += visible(v) > 0 ? "1" : "0";
        }
        return bin;
    }
    
    string getHidBinary() {
        string bin = "";
        for (int h=0; h<hidden.size(); ++h) {
            bin += hidden(h) > 0 ? "1" : "0";
        }
        return bin;
    }
    
    int visState() { return binToInt(getVisBinary()); }
    int hidState() { return binToInt(getHidBinary()); }
    
    int getTotVisState() { return totVisState; }
    int getTotHidState() { return totHidState; }
    
    double getE() { return visible.dot(-Weights*hidden - vis_bias) - hidden.dot(hid_bias); }
    double getE(VectorXd v, VectorXd h) {
        assert(v.size() == visible.size() && h.size()==hidden.size());
        return v.dot(-Weights*h - vis_bias) - h.dot(hid_bias);
    }
    
    double getZ() {
        double Z = 0.0;
        for (int v=0; v<totVisState; ++v) {
            VectorXd vVec = stateNumToVec(v, (int)visible.size());
            for (int h=0; h<totHidState; ++h) {
                VectorXd hVec = stateNumToVec(h, (int)hidden.size());

                Z += exp(-getE(vVec,hVec));
            }
        }
        return Z;
    }
    
    double jointP()                         { return exp(-getE())/getZ(); }
    double jointP(VectorXd v, VectorXd h)   { return exp(-getE(v,h))/getZ(); }
    
    double sum_joint_Pv() {
        double sum_p = 0.0;
        for (int v=0; v<totVisState; ++v) {
            VectorXd vis = stateNumToVec(v,(int)visible.size());
            sum_p += jointP(vis, hidden);
        }
        return sum_p;
    }
    double sum_joint_Ph() {
        double sum_p = 0.0;
        for (int h=0; h<totHidState; ++h) {
            VectorXd hid = stateNumToVec(h,(int)hidden.size());
            sum_p += jointP(visible, hid);
        }
        return sum_p;
    }
    
    double marginalP(string layer) {
        double sumP;
        if (layer=="v")      { sumP = sum_joint_Ph(); }
        else if (layer=="h") { sumP = sum_joint_Pv(); }
        else                 { throw invalid_argument("Invalid mode (\"v\" or \"h\" only)"); }
        return sumP/getZ();
    }
    
    double condiP(string layer) {
        double joint = jointP();
        double sumP;
        if (layer=="v|h")      { sumP = sum_joint_Pv(); }
        else if (layer=="h|v") { sumP = sum_joint_Ph(); }
        else                   { throw invalid_argument("Invalid mode (\"v|h\" or \"h|v\" only)"); }
        return joint/sumP;
    }
    
    
    
    
    
    
    void GibbsHV() {
        uniform_real_distribution<double> rand(0,1);
        VectorXd p_hv = ph_v_vec();
        for (int i=0; i<hidden.size(); ++i) { hidden(i) = rand(mt) < p_hv(i) ? 1.0 : -1.0; }
    }
    
    void GibbsVH() {
        uniform_real_distribution<double> rand(0,1);
        VectorXd p_vh = pv_h_vec();
        for (int i=0; i<visible.size(); ++i) { visible(i) = rand(mt) < p_vh(i) ? 1.0 : -1.0; }
    }
    
    void GibbsSampling(int iter=1) {
        while (iter>0) {
            GibbsHV();
            GibbsVH();
            --iter;
        }
    }
};

RBM::RBM(int num_vis, int num_hid) {
    uniform_real_distribution<double> rand(-1,1);
    visible  = VectorXd::NullaryExpr(num_vis, [&](){ return rand(mt)>0 ? 1.0 : -1.0; });
    vis_temp = visible;
    totVisState = visible.size()<20 ? (int)pow(2,visible.size()) : 0;
    
    hidden   = VectorXd::NullaryExpr(num_hid, [&](){ return rand(mt)>0 ? 1.0 : -1.0; });
    hid_temp = hidden;
    totHidState = hidden.size()<20 ? (int)pow(2,hidden.size()) : 0;
    
    vis_bias = VectorXd::NullaryExpr(num_vis, [&](){ return rand(mt); });
    hid_bias = VectorXd::NullaryExpr(num_hid, [&](){ return rand(mt); });
    Weights  = MatrixXd::NullaryExpr(num_vis, num_hid, [&](){ return rand(mt); });
}

#endif /* RBM_h */
