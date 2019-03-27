//
//  ml.cpp
//  MachineLearning
//
//  Created by Minh on 03/13/19.
//  Copyright © 2019 Minh. All rights reserved.
//

#include "Layer.h"

int main(int argc, const char * argv[]) {
//    string  neur_file           = "neurons.txt",
//            link_file           = "links.txt";
//    vector<string> pokemon {"pikachu", "mew", "snorlax"};
//    int num_neurons         = 100,
//        num_init_states     = 10;
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
    
    Layer L1(3,2);
    VectorXd v = VectorXd::Ones(3);
    VectorXd in = v.replicate(1,1);
    cout << in << endl;
    VectorXd out = L1.evaluate(in);
    cout << "Layer " << L1.layer() << endl;
    cout << out << endl << endl;
    cout << L1.getWeights() << endl;
    

    return 0;
}
