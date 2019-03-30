//
//  ml.cpp
//  MachineLearning
//
//  Created by Minh on 03/13/19.
//  Copyright © 2019 Minh. All rights reserved.
//

#include "FFNN.h"

int main(int argc, const char * argv[]) {
///////////////////////////////////////// HOPFIELD NETWORK /////////////////////////////////////////
    /*  HOPFIELD PARAMETERS
    string  neur_file           = "neurons.txt",
            link_file           = "links.txt";
    vector<string> pokemon {"pikachu", "mew", "snorlax"};
    int num_neurons         = 100,
        num_init_states     = 10;
     */
    
    /*  MAKING CONVERGING ENERGIES
    WriteConvergeStates("converge_states.txt", neur_file, link_file, num_neurons, num_init_states);
        vector<string> input_memories {face, tree};
        cout << input_memories.size() << endl;
        string temp_neighbor = out_folder + "temp_neighbor.txt";
    
        WriteNetworkMemory(input_memories, neur_file, link_file, mt);
        Hopfield H_net(neur_file, link_file);
        H_net.printNeurons();
        H_net.printNeighbors(temp_neighbor);
     */

    
    /*  MAKING HAMMING GRAPH
     RunHamming(brain_size, neur_file, link_file);
     */
    
    /*  MAKING ENERGY LANDSCAPE
    int brain_size = 7;
    RunLandscapeHopfield(brain_size, neur_file, link_file);
     */
    
    /*  REMEMBERING POKEMON IMAGES
    RunPokemonHopfield(neur_file, pokemon);
     */
//////////////////////////////////////////////////////////////////////////////////////////////////////
    
///////////////////////////////////// FEEDFORWARD NEURAL NETWORK /////////////////////////////////////

//    MatrixXd mat(2,3);
//    mat << 1,2,3,4,5,6;
//    cout << mat << endl << endl;
//    cout << mat(0,1) << endl;
    
    vector<int> neurons {4,3,2};
    uniform_int_distribution<int> rand_int(0,1);
    VectorXd v1 = VectorXd::NullaryExpr(4, [&]() { return rand_int(mt); }),
             v2 = VectorXd::NullaryExpr(4, [&]() { return rand_int(mt); }),
             v3 = VectorXd::NullaryExpr(4, [&]() { return rand_int(mt); }),
             v4 = VectorXd::NullaryExpr(4, [&]() { return rand_int(mt); }),
             v5 = VectorXd::NullaryExpr(2, [&]() { return rand_int(mt); });
    VectorXd v6 = v5.rowwise().reverse();
    
    v1.normalize(); //cout << v1 << endl << endl;
    v2.normalize(); //cout << v2 << endl << endl;
    v3.normalize(); //cout << v3 << endl << endl;

    vector<pair<VectorXd, VectorXd>> Data {make_pair(v1, v5),
                                           make_pair(v2, v5),
                                           make_pair(v3, v6),
                                           make_pair(v4, v6)};

    FFNN myNetwork(neurons, Data);
    myNetwork.setLearnRate(1.0);
    
    
    
    
    
    /*      TESTING FOR dC_dz
    uniform_real_distribution<double> rand(0,1);
    for (int n=0; n<50; ++n) {
        VectorXd v = VectorXd::NullaryExpr(5, [&](){ return rand(mt); });
        VectorXd w = VectorXd::NullaryExpr(5, [&](){ return rand_int(mt); });
        VectorXd exactDiff = dC_dz(v, w);
        
        for (int i=0; i<v.size(); ++i) {
            double Ci = CostCrossEntropy(v, w);
            v(i) += delta;
            double Cf = CostCrossEntropy(v, w);
            v(i) -= delta;
            double dC = (Cf - Ci)/delta;
            cout << dC - exactDiff(i) << " ";
            assert(dC - exactDiff(i) <= 1E-3);
        }
        cout << endl;
    }
     */
    
    
    /*      TESTING FOR FINITE DIFFERENCE GRAD. DESCENT
    double init_cost = myNetwork.Cost();
    cout << init_cost << endl << endl;
    for (int n=0; n<50; ++n) {
        myNetwork.gradDescfinDiff();
        double curr_cost = myNetwork.Cost();
        cout << curr_cost << endl;
        assert(init_cost > curr_cost);
        init_cost = curr_cost;
    }
     */
    
//////////////////////////////////////////////////////////////////////////////////////////////////////
    return 0;
}
