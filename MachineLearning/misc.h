//
//  misc.h
//  
//
//  Created by Minh on 03/27/19.
//

#ifndef misc_h
#define misc_h

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

#include <Eigen/Eigen>
using namespace std;
using namespace Eigen;

random_device rd;
mt19937 mt(rd());
const int num_sweeps = 200000;
const double fin_Delta = 1E-10;
//const int start_time = 0;
string  out_folder_files    = "files/",
        out_folder_img      = "img/",
        rbm_folder          = out_folder_files + "rbm/";


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

string zfill(int s, unsigned int z) {
    string str = to_string(s);
    str = string(z - str.length(),'0') + str;
    return str;
}

string dtos(double db, int prec=6) {
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


double sigmoid(double x) {
    return 1.0/(1.0 + exp(-x));
}

double CostQuad(VectorXd& outV, VectorXd& expV) {
    assert(outV.size() == expV.size());
    VectorXd v = outV - expV;
    return v.dot(v);
}

double CostCrossEntropy(VectorXd& outV, VectorXd& expV) {
    assert(outV.size() == expV.size());
    VectorXd ones = VectorXd::Ones(outV.size());
    VectorXd logOutV = outV.array().log().matrix();
    VectorXd log1minusOutV = (ones-outV).array().log().matrix();
    VectorXd CVec = -expV.cwiseProduct(logOutV) - (ones-expV).cwiseProduct(log1minusOutV);
    return CVec.sum();
}

VectorXd dSigmoid_dy(VectorXd& y) {
    VectorXd ones = VectorXd::Ones(y.size());
    VectorXd fy = y.unaryExpr(&sigmoid);
    return (fy).cwiseProduct(ones - fy);
}

VectorXd dCE_dz(VectorXd& outV, VectorXd& expV) {
    assert(outV.size() == expV.size());
    VectorXd ones = VectorXd::Ones(outV.size());
    VectorXd dCdz = -expV.cwiseQuotient(outV) + (ones-expV).cwiseQuotient(ones-outV);
    return dCdz;
}

VectorXd stateNumToVec(unsigned int state, unsigned int len) {
    string str = zfill(intToBin(state), len);
    VectorXd v(str.size());
    for (int i=0; i<str.size(); ++i) { v(i) = (str[i]=='1') ? 1.0 : -1.0; }
    return v;
}


int reverseInt (int i)
{
    unsigned char c1, c2, c3, c4;
    
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
void read_mnist(/*string full_path*/)
{
    ifstream file (/*full_path*/"t10k-images-idx3-ubyte");
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number));
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    
                }
            }
        }
    }
}

#endif /* misc_h */
