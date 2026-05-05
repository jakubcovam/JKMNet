#ifndef MLP_HPP
#define MLP_HPP

#include <vector>
#include "eigen-3.4/Eigen/Dense"

#include "Layer.hpp" 
#include "ConfigIni.hpp"
#include "Metrics.hpp"

using namespace std;

struct TrainingResult {
    double finalLoss = std::numeric_limits<double>::quiet_NaN();
    int iterations = 0;
    bool converged = false;
    int time = 0;
};

class MLP {

    public:
        MLP();  //!< The constructor
        ~MLP();  //!< The destructor     
        MLP(const MLP& other);  //!< The copy constructor
        MLP& operator=(const MLP& other);  //!< The assignment operator
        MLP(MLP&& other) noexcept;        //!< The move copy constructor
        MLP& operator=(MLP&& other) noexcept;  //!< The move assignment operator

        //MLP() = default;  //!< The constructor
       //~MLP() = default;  //!< The destructor 
        //MLP(const MLP&) = default;  //!< The copy constructor
       //MLP& operator=(const MLP&) = default;   //!< The assignment operator

        //!< Architecture (neurons per layer)
        std::vector<unsigned> getArchitecture();  //!< Getter for the architecture
        void setArchitecture(const std::vector<unsigned>& architecture);  //!< Setter for the architecture
        void printArchitecture();  //!< Print the architecture

        // Activations for each layer
        std::vector<activ_func_type> getActivations();  //!< Getter for the activation function
        //void setActivations(std::vector<activ_func_type>& funcs);  //!< Setter for the activation function
        void printActivations();  //!< Print the activation function

        void setActivations(const std::vector<activ_func_type>& funcs);
        void setWInitType(const std::vector<weight_init_type>& wInits);

         // Weight‐init types per layer 
        std::vector<weight_init_type> getWInitType();  //!< Getter for the weight initialization type
        //void setWInitType(std::vector<weight_init_type>& wInits);  //!< Setter for the weight initialization type
        void printWInitType();  //!< Print the weight initialization type

        //!< Number of layers
        size_t getNumLayers() const;  //!< Getter for the number of layers
        void setNumLayers(size_t layers);  //!< Setter for the number of layers

        //!< Number of inputs (neurons in first layer)
        unsigned getNumInputs();  //!< Getter for the number of inputs
        void setNumInputs(unsigned inputs);  //!< Setter for the number of inputs

        //!< Number of neurons at specific layer 
        unsigned getNumNeuronsInLayers(std::size_t index);  //!< Getter for the number of neurons at specific layer
        void setNumNeuronsInLayers(std::size_t index, unsigned count);  //!< Setter for the number of neurons at specific layer

        // Getter and Setter for inputs
        Eigen::VectorXd& getInps();  //!< Getter for the inputs
        void setInps(Eigen::VectorXd& inputs);   //!< Setter for the inputs

        // Getter and Setter for weigths
        Eigen::MatrixXd getWeights(size_t layerIndex) const;  //!< Getter for weights
        void setWeights(size_t layerIndex, const Eigen::MatrixXd& W);  //!< Setter for weights

        Eigen::VectorXd getWeightsVectorMlp(); //!< Getter for weights vector of MLP
        void weightsToVectorMlp(); //!< Merge weight vectors of all layers

        bool saveWeightsCsv(const std::string &path) const;  //!< Save weights in readable CSV text (per-layer blocks)
        bool saveWeightsBinary(const std::string &path) const;  //!< Save weights in compact binary
        bool saveWeightsVectorCsv(const std::string &path) const;  //!< Save vector of weights in readable CSV text (per-layer blocks)
        bool saveWeightsVectorBinary(const std::string &path) const;  //!< Save vector of weights in compact binary
        bool appendWeightsVectorCsv(const std::string &path, bool isFirstRun) const;  //!< Save vector of weights inside one file for ensemble run

        bool loadWeightsCsv(const std::string &wPath, const std::string &confPath);   //!< Load weights from CSV text (per-layer blocks)
        bool loadWeightsBinary(const std::string &wPath, const std::string &confPath);  //!< Load weights in compact binary
        bool loadWeightsVectorCsv(const std::string &wPath, const std::string &confPath);   //!< Load weights vector from CSV text
        Eigen::VectorXd& getOutput();   //!< Getter for output
        bool validateInputSize();  //!< Validate the size of the inputs compared to nNeurons[0]

        Eigen::VectorXd initMLP(const Eigen::VectorXd& input, int rngSeed);  //!< Forward pass through all layers     
        Eigen::VectorXd runMLP(const Eigen::VectorXd& input);  //!< Forward pass reusing existing weights
        bool compareInitAndRun(const Eigen::VectorXd& input, double tol = 1e-6, int rngSeed = 0) const;  //!< Compare if 'initMLP' and 'runMLP' produce the same output
        bool testRepeatable(const Eigen::VectorXd& input, int repeats = 10, double tol = 1e-8, int rngSeed = 0) const; //!< Repeatability check for 'runMLP'
        void runAndBP(const Eigen::VectorXd& input, const Eigen::VectorXd& obsOut, double learningRate); //!< Forward pass and update weights with backpropagation (one input)
        void runAndBPadam(const Eigen::VectorXd& input, const Eigen::VectorXd& obsOut, double learningRate, int iterationNum); //!< Forward pass and update weights with Adam algorithm (one input)

        void onlineBP(int maxIterations, double maxError, double learningRate, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);    //!< Online backpropagation - separete inp out matrices
        void onlineAdam(int maxIterations, double maxError, double learningRate, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);  //!< Online backpropagation using Adam algorithm - separete inp out matrices
        void batchAdam(int maxIterations, double maxError, int batchSize, double learningRate, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);  //!< Batch backpropagation using Adam algorithm - separete inp out matrices
        void batchBP(int maxIterations, double maxError, int batchSize, double learningRate, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);    //!< Batch backpropagation - separete inp out matrices

        std::vector<Eigen::MatrixXd> onlineAdamEpochVal(
            const Eigen::MatrixXd &Xtrain,
            const Eigen::MatrixXd &Ytrain,
            const Eigen::MatrixXd &Xval,
            const Eigen::MatrixXd &Yval,
            int maxIterations,
            double learningRate,
            int metricsAfterXEpochs);

        std::vector<Eigen::MatrixXd> batchAdamEpochVal(
            const Eigen::MatrixXd &Xtrain,
            const Eigen::MatrixXd &Ytrain,
            const Eigen::MatrixXd &Xval,
            const Eigen::MatrixXd &Yval,
            int batchSize,
            int maxIterations,
            double learningRate,
            int metricsAfterXEpochs);

        std::vector<Eigen::MatrixXd> onlineBpEpochVal(
            const Eigen::MatrixXd &Xtrain,
            const Eigen::MatrixXd &Ytrain,
            const Eigen::MatrixXd &Xval,
            const Eigen::MatrixXd &Yval,
            int maxIterations,
            double learningRate,
            int metricsAfterXEpochs);

        std::vector<Eigen::MatrixXd> batchBpEpochVal(
            const Eigen::MatrixXd &Xtrain,
            const Eigen::MatrixXd &Ytrain,
            const Eigen::MatrixXd &Xval,
            const Eigen::MatrixXd &Yval,
            int batchSize,
            int maxIterations,
            double learningRate,
            int metricsAfterXEpochs);

        void calcOneOutput(const Eigen::VectorXd& inputVec);  //!< Forward pass for one input
        void calculateOutputs(const Eigen::MatrixXd& inputMat); //!< Calculate outputs for given matrix of inputs
        Eigen::MatrixXd getOutputs() const;  //!< Getter for output matrix
        TrainingResult getResult() const {return result;}
        Eigen::VectorXd getFirstLayerDeltaSum(); //!< Sum deltas of first layer neurons to vector of size 1 for preceding networks

    protected:

    private:
        std::vector<unsigned> nNeurons;  //!< The vector of number of neurons per layer
        size_t numLayers;  //!< Cache of nNeurons.size()
        Eigen::VectorXd Inps;  //!< The vector of inputs  
        std::vector<activ_func_type> activFuncs;  //!< Vector of activation functions for each layer 
        std::vector<weight_init_type> wInitTypes;   //!< Vector of weights initialization for each layer
        std::vector<Layer> layers_;  //!< Private member of the class Layer to store each layer’s state
        Eigen::VectorXd output;  //!< The output vector of mlp
        Eigen::MatrixXd outputMat; //!< The output matrix of mlp
        Eigen::VectorXd weightsVectorMlp;  //!< The weights vector of all layers
        Eigen::MatrixXd calibCrit;  //>! Matrix of all calibration criteria
        Eigen::MatrixXd validCrit;  //>! Matrix of all validation criteria
        TrainingResult result; 
}; 

#endif // MLP_H
