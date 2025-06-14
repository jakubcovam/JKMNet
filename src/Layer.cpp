#include "Layer.hpp"
#include <iostream>
#include <random>
#include <cmath>
#include "eigen-3.4/Eigen/Dense"

using namespace std;

/**
 * The constructor
 */
Layer::Layer(): weights(),
        inputs(),
        //activations(),
        //bias(),
        output() {

}

/**
 * The destructor
 */
Layer::~Layer(){

}

/**
 * The copy constructor
 */
Layer::Layer(const Layer& other): weights(),
        inputs(),
        //activations(),
        //bias(), 
        output() {

    weights = other.weights;
    inputs = other.inputs;
    //activations= other.activations;
    //bias = other.bias; 
    output= other.output;

}

/**
 * The assignment operator
 */
Layer& Layer::operator=(const Layer& other){
    if (this == &other) return *this;
  else {
    weights = other.weights;
    inputs = other.inputs;
    //activations= other.activations;
    //bias = other.bias; 
    output= other.output;
    
  }
  return *this;

}

/**
 * Initialize the layer with the specified number of neurons and input size
 */
void Layer::initLayer(unsigned numInputs, unsigned numNeurons, weight_init_type initType, double minVal, double maxVal) {

    // Initialize weights
    //weights = Eigen::MatrixXd::Random(numNeurons, numInputs);
    initWeights(numNeurons, numInputs, initType, minVal, maxVal);

    // Initialize inputs (last element will always be 1.0 for bias)
    inputs = Eigen::VectorXd(numInputs);
    inputs.setZero();  // Set all to zero first
    inputs(numInputs - 1) = 1.0;  // Set last element (bias) to 1.0
    
    // Initialize bias
    //bias = Eigen::VectorXd(numNeurons);
    //bias.setOnes();  // Set all to one

    // Initialize activations
    //activations = Eigen::VectorXd(numNeurons);
    //activations.setZero();  // Set all to zero

    // Initialize output 
    output = Eigen::VectorXd(numNeurons);
    output.setZero();  // Set all to zero

}

/**
 * Initialize weights using specified technique
 */
void Layer::initWeights(unsigned numNeurons, unsigned numInputs, weight_init_type initType, double minVal, double maxVal) {
    // Initialize the weight matrix
    //weights = Eigen::MatrixXd(numNeurons, numInputs);
    
    switch (initType) {
        case weight_init_type::RANDOM: {
            // Random initialization between minVal and maxVal
            weights = Eigen::MatrixXd::Random(numNeurons, numInputs);
            // Scale to desired range
            // TODO: Check!!
            weights = weights.array() * (maxVal - minVal) / 2.0 + (maxVal + minVal) / 2.0;
            break;

        }
        
        case weight_init_type::LHS: {
            // Latin Hypercube Sampling initialization
            // TODO: Change!!
            weights = Eigen::MatrixXd::Random(numNeurons, numInputs);
            // Scale to desired range
            weights = weights.array() * (maxVal - minVal) / 2.0 + (maxVal + minVal) / 2.0;
            break;
        }
        
        default:
            std::cerr << "Error: Unknown weight initialization type! Selected RANDOM initialization." << std::endl;
            // Select random initialization
            weights = Eigen::MatrixXd::Random(numNeurons, numInputs);
            // Scale to desired range
            weights = weights.array() * (maxVal - minVal) / 2.0 + (maxVal + minVal) / 2.0;
            break;
    }
}

/**
 * Getter for the input vector to the layer
 */
Eigen::VectorXd Layer::getInputs() {
    return inputs;  
}

/**
 * Setter for the input vector to the layer
 */
void Layer::setInputs(const Eigen::VectorXd& newInputs) {
   
    inputs = newInputs;
    inputs(inputs.size() - 1) = 1.0;  // Ensure bias input is always 1.0
}

/**
 * Getter for the weight matrix of the layer
 */
Eigen::MatrixXd Layer::getWeights() {
    return weights; 
}

/**
 * Setter for the weight matrix of the layer
 */
void Layer::setWeights(const Eigen::MatrixXd& newWeights) {
    weights = newWeights;
}

/**
 * Calculate the weighted sum (linear combination of inputs)
 */
Eigen::VectorXd Layer::calculateWeightedSum() {
    //std::cout << "weights dimensions: " << weights.rows() << "x" << weights.cols() << std::endl;
    //std::cout << "inputs dimensions: " << inputs.size() << std::endl;
    //std::cout << "bias dimensions: " << bias.size() << std::endl;
    Eigen::VectorXd activation = weights * inputs ;//+ bias;  // Compute the weighted sum

    return activation;
}

/**
 * Apply activation function to the weighted sum
 */
Eigen::VectorXd Layer::applyActivationFunction(const Eigen::VectorXd& weightedSum, activ_func_type activFuncType) {
    Eigen::VectorXd activatedOutput = weightedSum;  // Copy the weighted sum

    // Apply the activation function based on the activation function type
    switch (activFuncType) {
        case activ_func_type::RELU:  // f(x) = max(0, x)
            activatedOutput = activatedOutput.array().max(0.0);
            break;
        
        case activ_func_type::SIGMOID:  // f(x) = 1 / (1 + exp(-x))
            // For loop approach: (might be slower for large datasets)
            // for (int i = 0; i < numActivation; ++i) {
            //     activatedOutput[i] = 1.0 / (1.0 + std::exp(-activatedOutput[i]));
            // }

            // Lambda approach: (might be faster for large datasets)
            activatedOutput = activatedOutput.array().unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
            break;

        case activ_func_type::LINEAR:  // f(x) = x
            activatedOutput = activatedOutput; 
            break;

        case activ_func_type::TANH:  // f(x) = (2 / (1 + exp(-2x))) - 1
            activatedOutput = activatedOutput.array().unaryExpr([](double x) { return (2.0 / (1.0 + std::exp(-2.0 * x))) - 1.0; });
            break;
        
        case activ_func_type::GAUSSIAN:  // f(x) = exp(-x^2)
            activatedOutput = activatedOutput.array().unaryExpr([](double x) { return std::exp(-x * x); });
            break;

        case activ_func_type::IABS:  // f(x) = x / (1 + |x|)
            activatedOutput = activatedOutput.array().unaryExpr([](double x) { return x / (1.0 + std::abs(x)); });
            break;  

        case activ_func_type::LOGLOG:  // f(x) = exp(-exp(-x))
            activatedOutput = activatedOutput.array().unaryExpr([](double x) { return std::exp(-1 * std::exp(-x)); });
            break; 
        
        case activ_func_type::CLOGLOG:  // f(x) = 1 - exp(-exp(x))
            activatedOutput = activatedOutput.array().unaryExpr([](double x) { return 1 - std::exp(- std::exp(x)); });
            break;

        case activ_func_type::CLOGLOGM:  // f(x) = -log(1 - exp(-x)) + 1
            activatedOutput = activatedOutput.array().unaryExpr([](double x) { return 1 - 2 * std::exp(-0.7 * std::exp(x)); });
            break;

        case activ_func_type::ROOTSIG:  // f(x) = x / ((1 + sqrt(1 + x^2)) * sqrt(1 + x^2))
            activatedOutput = activatedOutput.array().unaryExpr([](double x) { return x / (1 + std::sqrt(1.0 + std::exp(-x * x))); });
            break;

        case activ_func_type::LOGSIG:  // f(x) = sigmoid(x)^2
            activatedOutput = activatedOutput.array().unaryExpr([](double x) { return pow((1 / (1 + std::exp(-x))), 2); });
            break;

        case activ_func_type::SECH:  // f(x) = 2 / (exp(x) + exp(-x))
            activatedOutput = activatedOutput.array().unaryExpr([](double x) { return 2 / (std::exp(x) + std::exp(-x)); });
            break;

        case activ_func_type::WAVE:  // f(x) = (1 − 𝑎2) exp(−𝑎2)
            activatedOutput = activatedOutput.array().unaryExpr([](double x) { return (1 - x * x) * exp(-x * x); });
            break;

        default:
            std::cerr << "Error: Unknown activation function type!" << std::endl;
            break;
    }

    return activatedOutput;
}

/**
 * Calculate complete layer output (weighted sum + activation function)
 */
Eigen::VectorXd Layer::calculateLayerOutput(activ_func_type activFuncType) {
    Eigen::VectorXd weightedSum = calculateWeightedSum();
    output = applyActivationFunction(weightedSum, activFuncType);

    return output;
}


/**
 * Getter for the output vector
 */
Eigen::VectorXd Layer::getOutput() {
    return output; 
}
