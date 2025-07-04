#include "Layer.hpp"
#include <iostream>
#include <random>
#include <cmath>

using namespace std;

/**
 * The constructor
 */
Layer::Layer(): weights(),
        inputs(),
        //activations(),
        //bias(),
        output(),
        activ_func(),
        weightGrad() {

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
        output(),
        activ_func(),
        weightGrad() {

    weights = other.weights;
    inputs = other.inputs;
    //activations= other.activations;
    //bias = other.bias; 
    output = other.output;
    activ_func = other.activ_func;
    weightGrad = other.weightGrad;

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
    output = other.output;
    activ_func = other.activ_func;
    weightGrad = other.weightGrad;
    
  }
  return *this;

}

/**
 * Mapping activ_func_type from enum to string
 */
std::string Layer::activationName(activ_func_type f) {
    switch (f) {
      case activ_func_type::RELU: return "RELU";
      case activ_func_type::SIGMOID: return "SIGMOID";
      case activ_func_type::LINEAR: return "LINEAR";
      case activ_func_type::TANH: return "TANH";
      case activ_func_type::GAUSSIAN: return "GAUSSIAN";
      case activ_func_type::IABS: return "IABS";
      case activ_func_type::LOGLOG: return "LOGLOG";
      case activ_func_type::CLOGLOG: return "CLOGLOG";
      case activ_func_type::CLOGLOGM: return "CLOGLOGM";
      case activ_func_type::ROOTSIG: return "ROOTSIG";
      case activ_func_type::LOGSIG: return "LOGSIG";
      case activ_func_type::SECH: return "SECH";
      case activ_func_type::WAVE: return "WAVE";

    }
    return "Unknown";
}

/**
 * Mapping weight_init_type from enum to string
 */
std::string Layer::wInitTypeName(weight_init_type w) {
    switch (w) {
        case weight_init_type::RANDOM: return "RANDOM";
        case weight_init_type::LHS: return "LHS";
        case weight_init_type::LHS2: return "LHS2";
    }
    return "Unknown";
}

/**
 * Initialize the layer with the specified number of neurons and input size
 */
void Layer::initLayer(unsigned numInputs,
                      unsigned numNeurons,
                      weight_init_type initType,
                      activ_func_type  activFunc,
                      double           minVal,
                      double           maxVal) 
{

    activ_func = activFunc;

    // Initialize weights
    //weights = Eigen::MatrixXd::Random(numNeurons, numInputs);
    initWeights(numNeurons, numInputs + 1, initType, minVal, maxVal);

    // Initialize inputs (add the last element as 1.0 for bias)
    inputs = Eigen::VectorXd(numInputs + 1);
    inputs.head(numInputs).setZero();   // all real inputs start at 0
    inputs(numInputs) = 1.0;            // bias input = 1
    
    // Initialize bias
    //bias = Eigen::VectorXd(numNeurons);
    //bias.setOnes();  // Set all to one

    // Initialize activations
    //activations = Eigen::VectorXd(numNeurons);
    //activations.setZero();  // Set all to zero

    // Initialize output 
    output = Eigen::VectorXd(numNeurons);
    output.setZero();  // Set all to zero

    // Initialize BP gradient
    weightGrad.setZero(numNeurons, numInputs);

}

/**
 * Initialize weights using specified technique
 */
void Layer::initWeights(unsigned numNeurons, unsigned numInputs, weight_init_type initType, double minVal, double maxVal) {
    // Initialize the weight matrix
    //weights = Eigen::MatrixXd(numNeurons, numInputs);
    
    switch (initType) {
        
        // Random initialization between minVal and maxVal
        case weight_init_type::RANDOM: { 
            weights = Eigen::MatrixXd::Random(numNeurons, numInputs);
            
            // Scale to desired range
            if (minVal != -1.0 || maxVal != 1.0) {
                weights = minVal + (weights.array() + 1.0) * 0.5 * (maxVal - minVal);
            }

            break;
        }
        
        // Latin Hypercube Sampling initialization
        case weight_init_type::LHS: {
            // one shared random number generator
            static std::mt19937 gen{std::random_device{}()};

            // use size_t for all sizes and indices
            const std::size_t rows = static_cast<std::size_t>(numNeurons);
            const std::size_t cols = static_cast<std::size_t>(numInputs);
            const std::size_t totalWeights = rows * cols;
            const double span = maxVal - minVal;

            // reserve space up front
            std::vector<double> samples;
            samples.reserve(totalWeights);

            // for each stratum (subinterval), draw one point
            for (std::size_t i = 0; i < totalWeights; ++i) {
                double a = minVal + span * (double(i) / double(totalWeights));
                double b = minVal + span * (double(i + 1) / double(totalWeights));
                std::uniform_real_distribution<double> dist(a, b);
                samples.push_back(dist(gen));
            }

            // shuffle so weights aren’t tied to particular strata (subinterval)
            std::shuffle(samples.begin(), samples.end(), gen);

            // write back into the matrix (row-major)
            weights.resize(rows, cols);
            std::size_t idx = 0;
            for (std::size_t r = 0; r < rows; ++r) {
                for (std::size_t c = 0; c < cols; ++c) {
                    weights(r, c) = samples[idx++];
                }
            }

            break;
        }
        // Marta - Latin hypercube sampling by columns
        case weight_init_type::LHS2: {
            float range = maxVal - minVal;
            float range_interval = range / static_cast<float>(numNeurons);

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);

            weights.resize(numNeurons, numInputs);  // output matrix

            for (unsigned int col = 0; col < numInputs; ++col) {
                std::vector<float> column(numNeurons);

                // sampling to one column
                for (unsigned int i = 0; i < numNeurons; ++i) {
                    float sample = minVal + (i + dist(gen)) * range_interval;
                    column[i] = sample;
                }

                // shuffle the weights in column
                std::shuffle(column.begin(), column.end(), gen);

                // fill the column of weight matrix
                for (unsigned int row = 0; row < numNeurons; ++row) {
                    weights(row, col) = column[row];
                }
            }

            break;
        }
               
        default:
            std::cerr << "[Error]: Unknown weight initialization type! Selected RANDOM initialization." << std::endl;
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
    inputs = Eigen::VectorXd(newInputs.size() + 1);  // inputs + bias
    inputs.head(newInputs.size()) = newInputs;  // copy real inputs
    inputs(newInputs.size()) = 1.0;         // bias = 1
}

/**
 * Getter for the gradient matrix of the layer
 */
Eigen::MatrixXd Layer::getGradient() {
  return weightGrad;
}

/**
 * Setter for the gradient matrix of the layer
 */
void Layer::setGradient(const Eigen::MatrixXd& grad) {
  weightGrad = grad;
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
 * Apply a gradient calculation: W = W – η·∂E/∂W
 */
void Layer::updateWeights(double learningRate) {
  weights -= learningRate * weightGrad;
}

/**
 * Calculate the weighted sum (linear combination of inputs), i.e. calculate activations
 */
Eigen::VectorXd Layer::calculateWeightedSum() {
    //std::cout << "weights dimensions: " << weights.rows() << "x" << weights.cols() << std::endl;
    //std::cout << "inputs dimensions: " << inputs.size() << std::endl;
    //std::cout << "bias dimensions: " << bias.size() << std::endl;
    Eigen::VectorXd activation = weights * inputs ;  // Compute the weighted sum, i.e. calculate activations

    return activation;
}

/**
 * Apply activation function to the weighted sum
 */
Eigen::VectorXd Layer::setActivationFunction(const Eigen::VectorXd& weightedSum, activ_func_type activFuncType) {
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
            std::cerr << "[Error]: Unknown activation function type!" << std::endl;
            break;
    }

    // Detect any NaN or infinite
    if (!activatedOutput.array().isFinite().all()) {
        std::cerr << "[Warning] Non-finite activations detected!\n";
    }

    return activatedOutput;
}

/**
 * Calculate complete layer output (weighted sum + activation function)
 */
Eigen::VectorXd Layer::calculateLayerOutput(activ_func_type activFuncType) {
    Eigen::VectorXd weightedSum = calculateWeightedSum();
    output = setActivationFunction(weightedSum, activFuncType);

    return output;
}


/**
 * Getter for the output vector
 */
Eigen::VectorXd Layer::getOutput() {
    return output; 
}
