//
//  ml.cpp
//  MachineLearning
//
//  Created by Minh on 03/13/19.
//  Copyright © 2019 Minh. All rights reserved.
//
#include <iostream>
#include <boost/regex.h>
#include <Eigen/Dense>

using namespace std;

#include "input.h"
#include "Hopfield.h"

double sigmoid(double x) {
    return 1.0/(1.0 + exp(-x));
}

void printGrid(string str) {
    if (sqrt(str.size()) == (int)sqrt(str.size())) {
        int grid_size = (int)sqrt(str.size());
        for (int i=0; i<str.size(); ++i) {
            string out_str = str[i]=='0' ? "-" : "0";
            cout << out_str << " ";
            if (i % grid_size == grid_size-1) {
                cout << endl;
            }
        }
    }
}

void WriteConvergeStates(string outFile, string neur_file, string link_file,
                         int num_neurons, int num_states) {
    ofstream temp(outFile, fstream::trunc);
    for (int s=0; s<num_states; ++s) {
        WriteNetwork(num_neurons, neur_file, link_file, mt);
        Hopfield H_net(neur_file, link_file);
        H_net.randomize();
        for (int n=0; n<num_sweeps; ++n) {
            H_net.update(num_neurons);
            temp << H_net.getE() << " ";
        }
        temp << endl;
    }
    temp.close();
}

pair<vector<string>, vector<string>> getInputMemories(string in_mem_files) {
    ifstream inFile(in_mem_files);
    vector<string> name_list, memory_list;
    string name, memory;
    while (inFile >> name >> memory) {
        name_list.push_back(name);
        memory_list.push_back(memory);
    }
    pair<vector<string>,vector<string>> name_memory_pair = make_pair(name_list, memory_list);
    return name_memory_pair;
}

void viewMemories(vector<pair<string,string>> inp_memories) {
    for (int i=0; i<inp_memories.size(); ++i) {
        string  name    = inp_memories[i].first,
        memory  = inp_memories[i].second;
        cout << name << endl;
        printGrid(memory);
        cout << endl;
    }
}

void RunHammingHopfield(int brain_size, string neur_file, string link_file) {
    int max_num_corr        = 64,
        max_num_mem         = 100;
    
    WriteNetwork(brain_size, neur_file, link_file, mt);
    Hopfield H_net_mem(neur_file,link_file);
    assert(H_net_mem.getNumNeurons() == 100);
    string out_file_name = out_folder_files + "hamming.txt";
    ofstream out_file(out_file_name, fstream::trunc);
    for (int k=1; k<=max_num_corr; ++k) {
        for (int p=1; p<=max_num_mem; ++p) {
            H_net_mem.runHamming(k, p, out_file);
        }
    }
    out_file.close();
}

void RunLandscapeHopfield(int brain_size, string neur_file, string link_file) {
    WriteNetwork(brain_size, neur_file, link_file, mt);
    Hopfield H_net(neur_file, link_file);
    vector<string> input_memories = randomMemories(2, brain_size);
    
    ofstream outDigraph(out_folder_files + "landscape.digraph", fstream::trunc);
    outDigraph << "digraph landscape {" << endl;
    for (const string& mem : input_memories) {
        int s = binToInt(mem);
        cout << s+1 << " ";
        outDigraph << s+1 << " [shape=star, style=filled, fillcolor=red]" << endl;
    }
    cout << endl;
    
    H_net.setConnections(input_memories);
    
    ofstream outEnergy(out_folder_files + "landscape_energy.txt", fstream::trunc);
    for (int b=0; b<(int)pow(2.0,brain_size); ++b) {
        string neur_str = zfill(intToBin(b), brain_size);
        H_net.binToNeurons(neur_str);
        outEnergy << b << " " << H_net.getE() << endl;
    }
    outEnergy.close();
    
    ofstream outConnections(out_folder_files + "landscape_connections.txt", fstream::trunc);
    for (int state=0; state<(int)pow(2.0, H_net.getNumNeurons()); ++state) {
        const string init_str = zfill(intToBin(state), H_net.getNumNeurons());
        H_net.binToNeurons(init_str);
        H_net.setNewDefault();
        for (int neur=1; neur<=H_net.getNumNeurons(); ++neur) {
            H_net.updateSingle(neur);
            string updated_str = H_net.getBinary();
            if (updated_str != init_str) {
                int updated_state = binToInt(updated_str);
                outConnections << state << " " << updated_state << endl;
                outDigraph << state+1 << " -> " << updated_state+1 << ";" << endl;
            }
            H_net.reset();
        }
    }
    outDigraph << "}";
    
    outDigraph.close();
    outConnections.close();
}

void RunPokemonHopfield(string neur_file, vector<string> pokemon) {
    cout << "Getting memories..." << endl;
    vector<string> input_memories;
    for (const string& poke : pokemon) {
        string mem = readSingleMemory(out_folder_files+ "pokemon/" + poke +".txt");
        input_memories.push_back(mem);
    }
    int brain_size = (int)input_memories[0].size();
    
    cout << "Creating a Hopfield network..." << endl;
    WriteNetwork(brain_size, neur_file, mt);
    Hopfield HNet(neur_file, input_memories);
    
    cout << "Corrupting a random memory..." << endl;
    int corr_size = 64*8*20;
    cout << input_memories.size() << endl;
    uniform_int_distribution<int> rand_mem(0,(int)input_memories.size()-1);
    int mem = rand_mem(mt);
    string rand_corrupted = input_memories[2];
    rand_corrupted = corruptRand(rand_corrupted, corr_size);
    
    cout << "Running the neural network..." << endl;
    HNet.binToNeurons(rand_corrupted);
    int freq = 10000;
    for (int swp=0; swp<num_sweeps; swp+=freq) {
        cout << "Step " << swp << endl;
        string out_file =out_folder_files+ "pokemon/pokemon_" + zfill(swp,6) + ".txt";
        ofstream outFile(out_file, fstream::trunc);
        outFile << HNet.getBinary();
        HNet.update(freq);
        outFile.close();
    }
    
    string out_file =out_folder_files+ "pokemon/pokemon_" + to_string(num_sweeps) + ".txt";
    ofstream outFile(out_file, fstream::trunc);
    outFile << HNet.getBinary();
    outFile.close();
    
    cout << endl;
    
    cout << "...done" << endl;
}

int main(int argc, const char * argv[]) {
    string  neur_file           = "neurons.txt",
            link_file           = "links.txt";
    vector<string> pokemon {"pikachu", "mew", "snorlax"};
//    int num_neurons         = 100,
//        num_init_states     = 10;
    //WriteConvergeStates("converge_states.txt", neur_file, link_file, num_neurons, num_init_states);
    //    vector<string> input_memories {face, tree};
    //    cout << input_memories.size() << endl;
    //    string temp_neighbor = out_folder + "temp_neighbor.txt";
    //
    //    WriteNetworkMemory(input_memories, neur_file, link_file, mt);
    //    Hopfield H_net(neur_file, link_file);
    //    H_net.printNeurons();
    //    H_net.printNeighbors(temp_neighbor);
    
//    pair<vector<string>,vector<string>> names_and_memories = getInputMemories(input_file);
//    vector<string> memory_names     = names_and_memories.first;
//    vector<string> input_memories   = names_and_memories.second;
    
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
    
    
    

    return 0;
}
