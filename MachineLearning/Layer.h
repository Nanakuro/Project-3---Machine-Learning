//
//  Layer.h
//  
//
//  Created by Minh on 03/26/19.
//

#ifndef Layer_h
#define Layer_h

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

random_device rd;
mt19937 mt(rd());

class Layer {
    const int layer, num_input, num_output;
    MatrixXd Weights;
    VectorXd bias;
public:
    
}


#endif /* Layer_h */
