//
//  ml.cpp
//  MachineLearning
//
//  Created by Minh on 03/13/19.
//  Copyright © 2019 Minh. All rights reserved.
//

//#include "FFNN.h"
#include "RBM.h"

vector<pair<VectorXd,VectorXd>> readDataset(string data_file, string label_file) {
    vector<pair<VectorXd,VectorXd>> Dataset;
    string data_name  = "datasets/" + data_file,
           label_name = "datasets/" + label_file;
    cout << "Reading " << data_name << " & " << label_name << "...";
    ifstream dataFile(data_name), labelFile(label_name);
    
    if (!dataFile || !labelFile) {
        cout << "Unable to open file" << endl;
        exit(1);
    }
    
    string dat, lbl;
    while ( getline(dataFile,dat) && getline(labelFile, lbl) ) {
        vector<double> data_vec, label_vec;
        
        stringstream dat_ss(dat), lbl_ss(lbl);
        double dat_n, lbl_n;
        while (dat_ss >> dat_n) {
            data_vec.push_back(dat_n);
        }
        while (lbl_ss >> lbl_n) {
            label_vec.push_back(lbl_n);
        }
        VectorXd dataV  = Map<VectorXd>(data_vec.data(), data_vec.size()),
                 labelV = Map<VectorXd>(label_vec.data(), label_vec.size());
        
        Dataset.push_back(make_pair(dataV, labelV));
    }
    dataFile.close();
    labelFile.close();
    
    auto rng = default_random_engine {};
    shuffle(begin(Dataset), end(Dataset), rng);
    
    cout << "done" << endl;
    
    
    return Dataset;
}

vector<pair<VectorXd,VectorXd>> readDataset(string name) {
    return readDataset(name+"Data.txt", name+"Labels.txt");
}

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

    /*      TRAINING AND TESTING
    vector<pair<VectorXd,VectorXd>> Data = readDataset("training");
    vector<pair<VectorXd,VectorXd>> Test = readDataset("test");
    vector<pair<VectorXd,VectorXd>> betaJ = readDataset("testData.txt", "testBetaJs.txt");
    
    int epochs = 100, batch = 100;
    double eta = 0.1;
    vector<int> neurons {100,80,1};
    
    FFNN myNetwork(neurons, Data);
    myNetwork.setLearnRate(eta);
    
    cout << "Starting training protocol..." << endl;
    myNetwork.gradDescBP(epochs, batch);
    cout << "done" << endl;
    
    cout << "Commencing testing protocol..." << endl;
    double success = myNetwork.test(Test);
    cout << "done. Success rate: " << success*100 << "%" << endl;
    
    cout << "Writing z vs betaJ..." << endl;
    string out_file = out_folder_files + "z_vs_betaJ.txt";
    ofstream outFile(out_file, fstream::trunc);
    for (int i=0; i<betaJ.size(); ++i) {
        myNetwork.feedForward(betaJ[i].first);
        outFile << myNetwork.getOutput()(0) << " " << betaJ[i].second(0) << endl;
    }
    outFile.close();
    cout << "done" << endl;
     */
    
    /*      COMPARE FINITE DIFFERENCE AND BACKPROPAGATION
    double eta = 1.0;
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
    v4.normalize(); //cout << v4 << endl << endl;

    vector<pair<VectorXd, VectorXd>> Data {make_pair(v1, v5),
                                           make_pair(v2, v5),
                                           make_pair(v3, v6),
                                           make_pair(v4, v6)};
    FFNN myNetwork(neurons,Data);
    myNetwork.setLearnRate(eta);
    
    
    cout << "##################" << endl
         << "Finite Difference:" << endl
         << "##################" << endl;
    myNetwork.gradDescfinDiff();
    myNetwork.printdC();

    myNetwork.clearGrad();

    cout << "################" << endl
         << "Backpropagation:" << endl
         << "################" << endl;
    myNetwork.backPropagate(0, 4);
    myNetwork.printdC();
     */
    
    
    
    /*      TESTING FOR dC_dz
    uniform_real_distribution<double> rand(0,1);
    cout << "Difference between finite difference gradient and exact derivative formula:" << endl << endl;
    for (int n=0; n<30; ++n) {
        VectorXd v = VectorXd::NullaryExpr(5, [&](){ return rand(mt); });
        VectorXd w = VectorXd::NullaryExpr(5, [&](){ return rand_int(mt); });
        VectorXd exactDiff = dCE_dz(v, w);
        
        for (int i=0; i<v.size(); ++i) {
            double Ci = CostCrossEntropy(v, w);
            v(i) += fin_Delta;
            double Cf = CostCrossEntropy(v, w);
            v(i) -= fin_Delta;
            double dC = (Cf - Ci)/fin_Delta;
            cout << dC - exactDiff(i) << "\t";
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
    
    
///////////////////////////////////// RESTRICTED BOLTZMANN MACHINE /////////////////////////////////////
    
    int vis = 2, hid = 5;
    int num_samples = 1E5;
    
    RBM myRBM(vis,hid);
    
    vector<string> probs {"pv-h","ph-v","pv","ph","pvh"};
    
    //////////////// P(v|h) ////////////////
    string prob_str = "pv-h";
    string rbm_name = rbm_folder +"rbm_"+ prob_str + ".txt",
    test_name = rbm_folder +"test_"+ prob_str + ".txt";
    ofstream rbmFile(rbm_name,fstream::trunc), testFile(test_name, fstream::trunc);
    rbmFile << myRBM.hidState() << endl;
    for (int n=0; n<num_samples; ++n) {
        myRBM.GibbsVH();
        rbmFile << myRBM.visState() << " ";
    }
    rbmFile << endl;
    for (int s=0; s<myRBM.getTotVisState(); ++s) {
        myRBM.setVisState(s);
        double p = myRBM.condiP("v|h");
        testFile << s << " " << p << endl;
    }
    rbmFile.close();
    testFile.close();
    
    //////////////// P(v|h) ////////////////
    prob_str = "ph-v";
    rbm_name = rbm_folder +"rbm_"+ prob_str + ".txt";
    test_name = rbm_folder +"test_"+ prob_str + ".txt";
    rbmFile.open(rbm_name,fstream::trunc); testFile.open(test_name, fstream::trunc);
    rbmFile << myRBM.visState() << endl;
    for (int n=0; n<num_samples; ++n) {
        myRBM.GibbsHV();
        rbmFile << myRBM.hidState() << " ";
    }
    rbmFile << endl;
    for (int s=0; s<myRBM.getTotHidState(); ++s) {
        myRBM.setHidState(s);
        double p = myRBM.condiP("h|v");
        testFile << s << " " << p << endl;
    }
    rbmFile.close();
    testFile.close();
    
    
    
    
////////////////////////////////////////////////////////////////////////////////////////////////////////
    return 0;
}
