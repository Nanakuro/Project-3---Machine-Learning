//
//  Hopfield.h
//
//  Created by Minh on 02/28/19.
//  Copyright © 2019 Minh. All rights reserved.
//

#ifndef Hopfield_h
#define Hopfield_h

#include "misc.h"

void WriteNetwork(int size,
                  string neuronFile, string bondFile,
                  mt19937 &m,
                  bool file_exists=true) {
    
    if (file_exists) {
        ofstream sp_file, b_file;
        sp_file.open(neuronFile, fstream::trunc);
        b_file.open(bondFile, fstream::trunc);
        
        uniform_real_distribution<double> b_dist(-1.0,1.0);
        uniform_real_distribution<double> W_dist(-1.0,1.0);
        
        for (int i=0; i < size; ++i) {
            int node = i+1;
            //double h = 0.0, W = 1.0;
            int status = 1;
            double b = b_dist(m);
            
            sp_file << node << " " << status << " " << b << endl;
            
            for (int j=i+1; j<size; ++j) {
                double W = W_dist(m);
                int other_node = j+1;
                b_file << node << " " << other_node << " " << W << endl;
            }
        }
        sp_file.close();
        b_file.close();
    }
}

void WriteNetwork(int size,
                  string neuronFile,
                  mt19937 &m,
                  bool file_exists=true) {
    
    if (file_exists) {
        ofstream sp_file;
        sp_file.open(neuronFile, fstream::trunc);
        uniform_real_distribution<double> b_dist(-1.0,1.0);
        for (int i=0; i < size; ++i) {
            int node = i+1;
            //double h = 0.0, W = 1.0;
            int status = 1;
            double b = b_dist(m);
            sp_file << node << " " << status << " " << b << endl;
        }
        sp_file.close();
    }
}

vector<string> randomMemories(int num_mem, int mem_size) {
    vector<string> memories(num_mem);
    for (int i=0; i<num_mem; ++i) {
        memories[i] = "";
        for (int j=0; j<mem_size; ++j) {
            uniform_int_distribution<int> rand_bit(0,1);
            int r = rand_bit(mt);
            memories[i] += r==0 ? "0" : "1";
        }
    }
    return memories;
}

string readSingleMemory(string fileName) {
    ifstream inFile(fileName);
    if (!inFile) {
        cerr << "Unable to open file " << fileName << endl;
        exit(1);   // call system to stop
    }
    string memory;
    inFile >> memory;
    return memory;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Hopfield {
    random_device rd;
    mt19937 mt;
    vector<vector<pair<int,double>>> connections; // { _node1_:{(connection1, W11),(connection2, W12) ...}, _node2_:{(_,_),...},... }
    vector<int> original_neurons; // = all_neurons. Keep the initial neuron configuration
    vector<string> encoded_memories;
public:
    vector<int> all_neurons;
    vector<double> biases;
    Hopfield(string, string);
    Hopfield(string, vector<string>);
    
    vector<vector<pair<int,double>>> getConnections()       { return connections; }
    vector<string> getEncodedMemories()                     { return encoded_memories; }
    void    flipNeuron(int neuron_flip)                     { all_neurons[neuron_flip-1] *= -1; }
    void    turnOn(int neuron)                              { all_neurons[neuron-1] = 1; }
    void    turnOff(int neuron)                             { all_neurons[neuron-1] = -1; }
    void    reset()                                         { all_neurons = original_neurons; }
    void    setNewDefault()                                 { original_neurons = all_neurons; }
    void    setNewDefault(vector<int> vec)                  { original_neurons = vec; }
    void    setMemories(vector<string> memories)            { encoded_memories = memories; }
    int     getNumNeurons()                                 { return (int)all_neurons.size(); }
    int     getSize()                                       { return (int)sqrt(getNumNeurons()); }
    
    void copyState(Hopfield other_state) {
        encoded_memories    = other_state.getEncodedMemories();
        connections         = other_state.getConnections();
        all_neurons         = other_state.all_neurons;
        biases              = other_state.biases;
        
        original_neurons = all_neurons;
    }
    
    void printConnections() {
        for (int i=0; i < connections.size(); ++i) {
            cout << "Node " << i+1 << ":";
            for (int j=0; j < connections[i].size(); ++j) {
                pair<int,double> connection_pair = connections[i][j];
                cout << " (" << connection_pair.first << ", " << connection_pair.second << ")";
            }
            cout << endl;
        }
    }
    
    void printConnections(string fileName) {
        ofstream file(fileName, fstream::trunc);
        for (int i=0; i < connections.size(); ++i) {
            file << "Node " << i+1 << ":";
            for (int j=0; j < connections[i].size(); ++j) {
                pair<int,double> connection_pair = connections[i][j];
                file << " (" << connection_pair.first << ", " << connection_pair.second << ")";
            }
            file << endl;
        }
        file.close();
    }
    
    void printNeurons() {
        int size = (int) getNumNeurons();
        for (int i=0; i<size; ++i) {
            string sp_str = all_neurons[i]==1 ? "0" : "-";
            cout << sp_str << " ";
        }
        cout << endl;
    }
    
    void printNeurons(string fileName) {
        ofstream f(fileName, fstream::trunc);
        int size = (int) getNumNeurons();
        for (int i=0; i<size; ++i) {
            string sp_str = all_neurons[i]==1 ? "+" : "-";
            f << sp_str << " ";
        }
        f << endl;
    }
    
    void printMemories() {
        for (const string& mem : encoded_memories) {
            cout << mem << endl;
        }
    }
    
    void setConnections(vector<string> memories) {
        for (int i=0; i<getNumNeurons(); ++i) {
            for (int j=i+1; j<getNumNeurons(); ++j) {
                int current_node = i+1,
                    other_node   = j+1;
                double W = 0.0;
                for (const string& mem : memories) {
                    int current_bit  = (mem[i] == '1') ? 1 : -1,
                        other_bit    = (mem[j] == '1') ? 1 : -1;
                    W += current_bit * other_bit;
                }
                W /= (memories.size()*1.0);
                
                connections[i][j-1] = make_pair(other_node, W);
                connections[j][i] = make_pair(current_node, W);
            }
        }
    }
    
//    void setConnectionsTest(vector<string> memories) {
//        for (int i=0; i<getNumNeurons(); ++i) {
//            for (int j=i+1; j<getNumNeurons(); ++j) {
//                int current_node = i+1,
//                    other_node   = j+1;
//                double W = 0.0;
//                for (const string& mem : memories) {
//                    int current_bit  = (mem[i] == '0') ? -1 : 1,
//                        other_bit    = (mem[j] == '0') ? -1 : 1;
//                    W += current_bit * other_bit;
//                }
//                W /= (memories.size()*1.0);
//
//                connections[i][j-i-1] = make_pair(other_node, W);
//            }
//        }
//    }
    
    double getE() {
        double E=0.0;
        for (int i=0; i < connections.size(); ++i) {
            int node = i+1, neuron = all_neurons[i];
            double b = biases[i];
            for (int j=0; j < connections[i].size(); ++j) {
                int connection = connections[i][j].first;
                if (node < connection) {
                    double W = connections[i][j].second;
                    int neuron_connection = all_neurons[connection-1];
                    E += -W * neuron * neuron_connection;
                }
            }
            E += b*neuron;
        }
        return E;
    }
    
//    double getETest() {
//        double E=0.0;
//        int i=0;
//        while (i < connections.size()) {
//            int neuron = all_neurons[i];
//            double b = biases[i];
//
//            for (int j=0; j < connections[i].size(); ++j) {
//                int connection = connections[i][j].first;
//                double W = connections[i][j].second;
//                int neuron_connection = all_neurons[connection-1];
//                E += -0.5 * W * neuron * neuron_connection;
//            }
//            E += b*neuron;
//            ++i;
//        }
//        E += biases[i] * all_neurons[i];
//        return E;
//    }
    
    void binToNeurons(string binary) {
        assert(binary.size() == getNumNeurons());
        for (string::size_type i=0; i<binary.size(); ++i) {
            all_neurons[i] = (binary[i]=='1') ? 1 : -1;
        }
    }
    
    string getBinary() {
        string bin = "";
        for (const auto &s : all_neurons) {
            bin += s==1 ? "1" : "0";
        }
        return bin;
    }
    
    void randomize(int size=0) {
        if (size > 0) { all_neurons.resize(size); }
        uniform_int_distribution<int> rand_neuron(0,1);
        for (int i=0; i<getNumNeurons(); ++i) {
            int r = rand_neuron(mt);
            all_neurons[i] = r==1 ? 1 : -1;
        }
    }
    
    double getTotalWeight(int neuron) {
        double total_weight = 0.0;
        int idx_neur = neuron-1;
        for (int j=0; j<connections[idx_neur].size(); ++j) {
            pair<int, double> conn = connections[idx_neur][j];
            int idx_conn = conn.first-1;
            double weight = conn.second;
            total_weight += (all_neurons[idx_conn] * weight);
        }
        return total_weight;
    }
    
//    double getTotalWeightTest(int neuron) {
//        double total_weight = 0.0;
//        int idx_neur = neuron-1;
//        int j=0;
//        for (int i=0; i<getNumNeurons(); ++i) {
//            int next_neur = i+1;
//            if (neuron == next_neur) { continue; }
//            int conn_i = min(i, idx_neur);
//            int conn_j = abs(neuron - next_neur) + conn_i-1;
//            pair<int,double> conn = connections[conn_i][conn_j];
//            double weight = conn.second;
//            total_weight += all_neurons[i] * weight;
//            ++j;
//        }
//        return total_weight;
//    }
    
    double getWeight(int neuron) {
        double W = 0.0;
        int idx_neur = neuron - 1;
        for (int idx_othr=0; idx_othr<getNumNeurons(); ++idx_othr) {
            if (idx_othr == idx_neur) { continue; }
            for (const string& mem : encoded_memories) {
                int stat_neur   = mem[idx_neur]=='1' ? 1 : -1,
                    stat_othr   = mem[idx_othr]=='1' ? 1 : -1;
                double weight = 1.0*all_neurons[idx_othr]*(stat_neur)*(stat_othr)
                                / (double)encoded_memories.size();
                W += weight;
            }
        }
        return W;
    }
    
    void updateSingle(int neuron) {
        double weight = getWeight(neuron);
        (weight > biases[neuron-1]) ? turnOn(neuron) : turnOff(neuron);
    }
    
    void update(int steps=1) {
        while (steps > 0) {
            uniform_int_distribution<int> rand_neur(1,getNumNeurons());
            int neuron = rand_neur(mt);
            updateSingle(neuron);
            steps -= 1;
        }
    }
    
    void runMemory(string corr_img) {
        binToNeurons(corr_img);
        update(num_sweeps);
    }
    
    void runMemory(string corr_img, ofstream& outFile) {
        binToNeurons(corr_img);
        const int steps = 50;
        for (int swp=0; swp<=num_sweeps; swp+=steps) {
            outFile << swp << " " << getBinary() << endl;
            update(steps);
        }
    }
    
    void runHamming(int num_corr, int num_mem, ofstream& outFile) {
        outFile << num_corr << " " << num_mem << " ";
        const int num_set     = 5,
                  num_tries   = 20;
        double hamming_avg = 0.0;
        for (int n=0; n<num_set; ++n) {
            vector<string> memory_set = randomMemories(num_mem, getNumNeurons());
            setConnections(memory_set);
            
            for (int i=0; i< num_tries; ++i) {
                uniform_int_distribution<int> rand_mem(0, num_mem-1);
                int r = rand_mem(mt);
                string memory = memory_set[r];
                string corrupted = corruptRand(memory, num_corr);
                runMemory(corrupted);
                hamming_avg += 1.0*hamming(getBinary(), memory)/(num_set * num_tries);
            }
        }
        outFile << hamming_avg << endl;
    }
    
    static int hamming(string s1, string s2) {
        assert(s1.size() == s2.size());
        int h_dist = 0;
        for (int i=0; i<s1.size(); ++i) {
            if (s1[i] != s2[i]) { ++h_dist; }
        }
        return h_dist;
    }
    
};

Hopfield::Hopfield(string neuronFile, string bondsFile) : rd(), mt(rd()) {
    ifstream neuron_inp(neuronFile);
    ifstream bonds_inp(bondsFile);
    int node, sp, link;
    int line = 0;
    double h, W;
    while (neuron_inp >> node >> sp >> h) {
        all_neurons.push_back(sp);
        biases.push_back(h);
        line++;
    }
    original_neurons = all_neurons;
    connections.resize(line);
    while (bonds_inp >> node >> link >> W) {
        connections[node-1].push_back(make_pair(link, W));
        connections[link-1].push_back(make_pair(node, W));
    }
}

Hopfield::Hopfield(string neuronFile, vector<string> memories) : rd(), mt(rd()) {
    ifstream neuron_inp(neuronFile);
    int neuron, status;
    int line = 0;
    double h;
    while (neuron_inp >> neuron >> status >> h) {
        all_neurons.push_back(status);
        biases.push_back(h);
        line++;
    }
    original_neurons = all_neurons;
    encoded_memories = memories;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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
#endif /* Hopfield_h */
