#ifndef LSTMLAYER_HPP
#define LSTMLAYER_HPP

#include "eigen-3.4/Eigen/Dense"
#include "ConfigIni.hpp"

struct LSTMSettings{
    int cells = 0; // number of cells (neurons) in layer
    int cells4gates = 0; // number of gates in all cells (cells * 4)
    int inputs = 0; // number of input values to layer
    int timeSteps = 0; // number of time steps in one computation segment of layer
    int outTS = 0; // number of last time steps used as actual output
    bool initialized = false;
};

class LSTMLayer
{
public:
    LSTMLayer() = default;  //!< The constructor
    ~LSTMLayer() = default;  //!< The destructor
    LSTMLayer(const LSTMLayer&) = default;  //!< The copy constructor
    LSTMLayer& operator=(const LSTMLayer&) = default;  //!< The assignment operator
    LSTMLayer(LSTMLayer&&) = default;  //!< The move constructor
    LSTMLayer& operator=(LSTMLayer&&) = default;  //!< The move assignment operator

    Eigen::VectorXd sigmoidVector(const Eigen::VectorXd& vec);  //!< Sigmoid function for all elements in vector
    Eigen::VectorXd tanhVector(const Eigen::VectorXd& vec);     //!< Sigmoid derivative function for all elements in vector
    Eigen::VectorXd sigmoidDerivVector(const Eigen::VectorXd& vec);     //!< Tanh function for all elements in vector
    Eigen::VectorXd tanhDerivVector(const Eigen::VectorXd& vec);    //!< Tanh derivative function for all elements in vector

    void initLSTMLayer(const int numInputs,        
                    const int numCells,
                    const int numTimeSteps,
                    const int numOutputTimeSteps,
                    std::string initType = "RANDOM",
                    int rngSeed = 0,
                    double minVal = 0.0,
                    double maxVal = 1.0);   //!< Initialize LSTMLayer, its weights using specified technique

    void setInputTSSegment(const Eigen::MatrixXd& inputSegment);    //!< Set inputs for n time-steps
    void calculateTimeSteps();      //!< Calculate outputs for n time-steps
    void calculateGradients();
    void updateWeights(double learningRate);
    void eraseMemory();

private:
    LSTMSettings settings;  //!< Settings of the layer
    Eigen::MatrixXd W;  //!< The input weight matrix (all gates - by rows candidate, forget, input, output)
    Eigen::MatrixXd U;  //!< The hidden state (short memory) gate weight matrix (all gates - by rows candidate, forget, input, output)
    Eigen::VectorXd b;  //!< The bias vector (all gates...)
    Eigen::MatrixXd timeStepsInputs;  //!< The matrix of inputs for n time-steps (rows = time-steps)
    Eigen::MatrixXd gatesActivations;  //!< Computed activations all gates (by rows) all time steps (by columns)
    Eigen::MatrixXd gatesOutputs;   //!< Computed outputs all gates (by rows) all time steps (by columns)
    Eigen::MatrixXd cellState;      //!< Computed cell state (long memory) all cells (rows) all time steps (columns)
    Eigen::MatrixXd output;         //!< Computed outputs (short memory) all cells (rows) all time steps (columns)
    Eigen::MatrixXd Wgradient;
    Eigen::MatrixXd Ugradient;
    Eigen::VectorXd bGradient;
    Eigen::MatrixXd deltaOutput;
    Eigen::MatrixXd deltaCellState;
    Eigen::MatrixXd deltaGates;
};

#endif // LSTMLAYER_HPP