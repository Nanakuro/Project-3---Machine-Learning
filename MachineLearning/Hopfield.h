//
//  Hopfield.h
//  Project 2 - Statistical Mechanics, Universality, and Renormalization Group
//
//  Created by Minh on 02/28/19.
//  Copyright © 2019 Minh. All rights reserved.
//

#ifndef Hopfield_h
#define Hopfield_h
#include <string>
#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include <utility>
#include <cmath>
#include <sys/stat.h>
#include <unistd.h>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <algorithm>

random_device rd;
mt19937 mt(rd());
const int num_sweeps = 1000;
//const int start_time = 0;
string  out_folder_files    = "files/",
        out_folder_img      = "img/";

string intToBin(unsigned int n) {
    string str = "";
    if (n / 2 != 0) {
        str += intToBin(n / 2);
    }
    str += to_string(n % 2);
    return str;
}

int binToInt(string s) {
    int bin_int = 0;
    for (int i=0; i<s.size(); ++i) {
        assert(s[i]=='1' || s[i]=='0');
        int bit = s[i]=='1' ? (int)pow(2.0, s.size()-i-1) : 0;
        bin_int += bit;
    }
    return bin_int;
}

inline bool fileExists (const string& name) {
    struct stat buffer;
    return (stat (name.c_str(), &buffer) == 0);
}

string zfill(string s, unsigned int z) {
    s = string(z - s.length(),'0') + s;
    return s;
}

string dtos(double db, int prec) {
    stringstream stream;
    stream << fixed << setprecision(prec) << db;
    string s = stream.str();
    return s;
}

vector<double> makeSequenceVec(double start, double end, int num_steps, bool endpoint=false) {
    vector<double> list;
    double step = (endpoint) ? (end - start)/(num_steps-1.0) : (end - start)/num_steps;
    double s = start;
    if (start <= end) {
        while (s - end <= 1e-6) {
            list.push_back(s);
            s += step;
        }
        if (!endpoint) { list.pop_back(); }
    } else {
        cout << "Error: start is larger than end!" << endl;
    }
    return list;
}
string corruptRand(string str, int num_char) {
    if (num_char > str.size()) {
        cout << "Number of chars > string length" << endl;
        return str;
    }
    int length = (int)str.size();
    uniform_int_distribution<int> rand_char(0,length-1);
    vector<int> idx_list;
    int rand_idx = rand_char(mt);
    while(idx_list.size()<num_char) {
        while (find(idx_list.begin(), idx_list.end(), rand_idx) != idx_list.end()) {
            rand_idx = rand_char(mt);
        }
        char my_char = str[rand_idx]=='0' ? '1' : '0';
        str = str.substr(0,rand_idx) + my_char + str.substr(rand_idx+1);
        idx_list.push_back(rand_idx);
    }
    return str;
}

string corruptLeft(string str, double frac) {
    if (frac > 1 || frac <= 0) {
        cout << "Invalid fraction of grid" << endl;
        return str;
    }
    int grid_size   = (int)sqrt(str.size());
    int num_columns = (int)(grid_size * frac);
    
    for (int i=0; i<str.size(); ++i) {
        if (i % grid_size < num_columns) {
            str[i] = str[i]=='0' ? '1' : '0';
        }
    }
    return str;
}

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
                    int current_bit  = (mem[i] == '0') ? -1 : 1,
                        other_bit    = (mem[j] == '0') ? -1 : 1;
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
                    E += -0.5 * W * neuron * neuron_connection;
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
        for (string::size_type i=0; i<binary.size(); ++i) {
            all_neurons[i] = binary[i]=='1' ? 1 : -1;
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
        for (int i=0; i<getNumNeurons(); ++i) {
            if (i == idx_neur) { continue; }
            for (const string& mem : encoded_memories) {
                double weight = 1.0*(mem[idx_neur]-'0') * (mem[i]-'0')
                                    / encoded_memories.size();
                W += weight;
            }
        }
        return W;
    }
    
    void updateSingle(int neuron) {
        double weight = getTotalWeight(neuron);
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
        const int steps = 50;
        for (int swp=0; swp<=num_sweeps; swp+=steps) {
            update(steps);
        }
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
    connections.resize(line-1);
    while (bonds_inp >> node >> link >> W) {
        pair<int,double> link_pair = make_pair(link, W);
        connections[node-1].push_back(link_pair);
//        if (link >= 1) {
//            pair<int,double> link_pair = make_pair(node, W);
//            connections[link-1].push_back(link_pair);
//        }
    }
}

Hopfield::Hopfield(string neuronFile, vector<string> memories) : rd(), mt(rd()) {
    ifstream neuron_inp(neuronFile);
    int node, sp;
    int line = 0;
    double h;
    while (neuron_inp >> node >> sp >> h) {
        all_neurons.push_back(sp);
        biases.push_back(h);
        line++;
    }
    original_neurons = all_neurons;
    encoded_memories = memories;
}

#endif /* Hopfield_h */
