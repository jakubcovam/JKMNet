#include "LSTMLayer.hpp"
#include <iostream>
#include <random>
#include <cmath>

using namespace std;

Eigen::VectorXd LSTMLayer::sigmoidVector(const Eigen::VectorXd& vec) {
    Eigen::VectorXd res = vec;
    res = res.array().unaryExpr([](double x) { return 1.0 / (1.0 + std::exp(-x)); });
    return res;
}

Eigen::VectorXd LSTMLayer::tanhVector(const Eigen::VectorXd& vec) {
    Eigen::VectorXd res = vec;
    res = res.array().unaryExpr([](double x) { return (2.0 / (1.0 + std::exp(-2.0 * x))) - 1.0; });
    return res;
}

Eigen::VectorXd LSTMLayer::sigmoidDerivVector(const Eigen::VectorXd& vec) {
    Eigen::VectorXd res = vec;
    res = res.array().unaryExpr([](double x) { return (1.0 / (1.0 + std::exp(-x))) * (1.0 - (1.0 / (1.0 + std::exp(-x)))); });
    return res;
}

Eigen::VectorXd LSTMLayer::tanhDerivVector(const Eigen::VectorXd& vec) {
    Eigen::VectorXd res = vec;
    res = res.array().unaryExpr([](double x) 
    { return 1.0 - ((2.0 / (1.0 + std::exp(-2.0 * x))) - 1.0) * ((2.0 / (1.0 + std::exp(-2.0 * x))) - 1.0); });
    return res;
}

void LSTMLayer::initLSTMLayer(const int numInputs,        
                const int numCells,
                const int numTimeSteps,
                const int numOutputTimeSteps,
                bool isFirstLayer,
                std::string initType,
                int rngSeed,
                double minVal,
                double maxVal){

    settings.cells = numCells;
    settings.cells4gates = 4 * settings.cells;
    settings.inputs = numInputs;
    settings.timeSteps = numTimeSteps;
    settings.outTS = numOutputTimeSteps;
    settings.isFirstLayer = isFirstLayer;
    auto weightInit = strToWeightInit(initType);

    switch (weightInit) {
        
        // Random initialization between minVal and maxVal
        case weight_init_type::RANDOM: { 
            std::mt19937 gen(rngSeed == 0 ? std::random_device{}() : rngSeed);
            std::uniform_real_distribution<> dist(minVal, maxVal);
            W = Eigen::MatrixXd(settings.cells4gates, settings.inputs);          
            W = W.unaryExpr([&](double) { return dist(gen); });
            U = Eigen::MatrixXd(settings.cells4gates, settings.cells);          
            U = U.unaryExpr([&](double) { return dist(gen); });
            break;
        }

        default:
            std::cerr << "[Error]: Unknown weight initialization type! Selected RANDOM initialization." << std::endl;
            // Select random initialization
            std::mt19937 gen(rngSeed == 0 ? std::random_device{}() : rngSeed);
            std::uniform_real_distribution<> dist(minVal, maxVal);
            W = Eigen::MatrixXd(settings.cells4gates, settings.inputs);          
            W = W.unaryExpr([&](double) { return dist(gen); });
            U = Eigen::MatrixXd(settings.cells4gates, settings.cells);          
            U = U.unaryExpr([&](double) { return dist(gen); });
            break;
    }

    b = Eigen::VectorXd::Zero(settings.cells4gates);
    timeStepsInputs = Eigen::MatrixXd::Zero(settings.timeSteps,settings.inputs);
    gatesActivations = Eigen::MatrixXd::Zero(settings.cells4gates,settings.timeSteps);
    gatesOutputs = Eigen::MatrixXd::Zero(settings.cells4gates,settings.timeSteps);
    cellState = Eigen::MatrixXd::Zero(settings.cells,settings.timeSteps);
    output = Eigen::MatrixXd::Zero(settings.cells,settings.timeSteps);
    forwardOutput = Eigen::MatrixXd::Zero(settings.cells,settings.outTS);
    Wgradient = Eigen::MatrixXd::Zero(settings.cells4gates, settings.inputs);
    Ugradient = Eigen::MatrixXd::Zero(settings.cells, settings.cells);
    bGradient = Eigen::VectorXd::Zero(settings.cells4gates);
    deltaOutput = Eigen::MatrixXd::Zero(settings.cells,settings.timeSteps);
    deltaCellState = Eigen::MatrixXd::Zero(settings.cells,settings.timeSteps);
    deltaGates = Eigen::MatrixXd::Zero(settings.cells4gates,settings.timeSteps);
    deltaInputs = Eigen::MatrixXd::Zero(settings.inputs, settings.timeSteps);

    settings.initialized = true;
}

void LSTMLayer::setInputTSSegment(const Eigen::MatrixXd& inputSegment){
    if (!settings.initialized)
        throw std::logic_error("[setInputTSSegment] called before [initLSTMLayer]");

    if (inputSegment.cols() != timeStepsInputs.cols())
        throw std::invalid_argument("[setInputTSSegment] Number of inputs (cols) doesn't match the initialized");

    if (inputSegment.rows() != timeStepsInputs.rows())
        std::cerr << "[Warning][setInputTSSegment] Number of time-steps (rows) doesn't match the initialized"<<"\n";

    timeStepsInputs = inputSegment;
}

void LSTMLayer::calculateTimeSteps(){
    if (!settings.initialized)
        throw std::logic_error("[calculateTimeSteps] called before [initLSTMLayer]");
    int c = settings.cells;

    //first TS
    gatesActivations.col(0) = W * timeStepsInputs.row(0).transpose() + b;
    gatesOutputs.col(0).segment(0, c) = tanhVector(gatesActivations.col(0).segment(0, c));   //candidate
    gatesOutputs.col(0).segment(c, 3 * c) = 
                                sigmoidVector(gatesActivations.col(0).segment(c, 3 * c));    //forget,input,output

    cellState.col(0) = gatesOutputs.col(0).segment(2 * c, c).array() *    //input 
                       gatesOutputs.col(0).segment(0, c).array();         //candidate
    
    output.col(0) = gatesOutputs.col(0).segment(3 * c, c).array() *       //output
                    tanhVector(cellState.col(0)).array();

    //rest of TS
    for(int i = 1 ; i < timeStepsInputs.rows() ; i++){
        gatesActivations.col(i) = W * timeStepsInputs.row(i).transpose() + b + U * output.col(i-1);
        gatesOutputs.col(i).segment(0, c) = tanhVector(gatesActivations.col(i).segment(0, c));    //candidate
        gatesOutputs.col(i).segment(c, 3 * c) = 
                                sigmoidVector(gatesActivations.col(i).segment(c, 3 * c));         //forget,input,output

        cellState.col(i) = cellState.col(i-1).array() * 
                           gatesOutputs.col(i).segment(c, c).array() +      //forget
                           gatesOutputs.col(i).segment(2 * c, c).array() *  //input  
                           gatesOutputs.col(i).segment(0, c).array();       //candidate
        
        output.col(i) = gatesOutputs.col(i).segment(3 * c, c).array() *     //output
                        tanhVector(cellState.col(i)).array();
    }
    forwardOutput = output.rightCols(settings.outTS);
}

void LSTMLayer::calculateGradients(){
    int t = settings.timeSteps - 1;
    int c = settings.cells;

    for(int i = t ; i >= 0 ; i--){
        Eigen::VectorXd delt = Eigen::VectorXd::Zero(c);
        if(i >= settings.timeSteps - settings.outTS){
            int idx = i - (settings.timeSteps - settings.outTS);
            delt = nextLayerDelta.col(idx);
        }

        if(i < t){
            delt += U.transpose() * deltaGates.col(i+1);
        }
        deltaOutput.col(i) = delt;

        deltaCellState.col(i) = deltaOutput.col(i).array() * 
                                gatesOutputs.col(i).segment(3 * c, c).array() *    //output
                                tanhDerivVector(cellState.col(i)).array();

        deltaGates.col(i).segment(0, c) =                                                                //candidate
                                         deltaCellState.col(i).array() *
                                         gatesOutputs.col(i).segment(2 * c, c).array() *                 //input
                                         tanhDerivVector(gatesActivations.col(i).segment(0, c)).array(); //candidate

        (i > 0) ? deltaGates.col(i).segment(c, c) =                                                                   //forget
                                                    deltaCellState.col(i).array() *
                                                    cellState.col(i-1).array() * 
                                                    sigmoidDerivVector(gatesActivations.col(i).segment(c, c)).array() :   //forget
                                                    deltaGates.col(0).segment(c, c).setZero();
        
        deltaGates.col(i).segment(2 * c, c) =                                                               //input
                                             deltaCellState.col(i).array() *
                                             gatesOutputs.col(i).segment(0, c).array() *                    //candidate
                                             sigmoidDerivVector(gatesActivations.col(i).segment(2 * c, c)).array(); //input

        deltaGates.col(i).segment(3 * c, c) =                                                               //output
                                             deltaOutput.col(i).array() * 
                                             tanhVector(cellState.col(i)).array() * 
                                             sigmoidDerivVector(gatesActivations.col(i).segment(3 * c, c)).array(); //output
        
    }

    Wgradient = deltaGates * timeStepsInputs;
    bGradient = deltaGates.rowwise().sum();
    Ugradient = deltaGates * output.transpose();

    if(!settings.isFirstLayer){
        deltaInputs = W.transpose() * deltaGates;
    }
}

void LSTMLayer::updateWeights(double learningRate){
    W -= learningRate * Wgradient;
    U -= learningRate * Ugradient;
    b -= learningRate * bGradient;
}

void LSTMLayer::eraseMemory(){
    cellState.setZero();
    output.setZero();
    // timeStepsInputs.setZero();
    // gatesActivations.setZero();
    // gatesOutputs.setZero();
    // Wgradient.setZero();
    // Ugradient.setZero();
    // bGradient.setZero();
    // deltaOutput.setZero();
    // deltaCellState.setZero();
    // deltaGates.setZero();

}

void LSTMLayer::setDeltaFromNextLayer(const Eigen::VectorXd& delta){
    if (delta.size() % settings.outTS != 0)
        throw std::logic_error("[setDeltaFromNextLayer] unable to split vector into columns (non-divisible)");
    nextLayerDelta =  Eigen::Map<const Eigen::MatrixXd>(delta.data(), (delta.size() / settings.outTS), settings.outTS);
}
    
void LSTMLayer::setDeltaFromNextLayer(const Eigen::MatrixXd& delta){
    nextLayerDelta = delta;
}

Eigen::MatrixXd LSTMLayer::getDeltaInputs(){
    return deltaInputs;
}

Eigen::MatrixXd LSTMLayer::getForwardOutput(){
    return forwardOutput.transpose();
}

Eigen::VectorXd LSTMLayer::getForwardOutputVector(){
    return forwardOutput.reshaped<Eigen::ColMajor>();
}