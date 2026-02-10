#include "JKMNet.hpp"
#include "Metrics.hpp"

#include <random>
#include <iostream>
#include "eigen-3.4/Eigen/Dense"
#include <filesystem>
#include <fstream>
#include <ostream> 

using namespace std;

/**
 * The constructor
 */
JKMNet::JKMNet(){

}

JKMNet::JKMNet(const RunConfig& cfg, unsigned nthreads)
    : cfg_(cfg), nthreads_(nthreads) {
    }

/**
 * The destructor
 */
JKMNet::~JKMNet(){

}

/**
 * The copy constructor
 */
JKMNet::JKMNet(const JKMNet& other) {
   
}

/**
 * The assignment operator
 */
JKMNet& JKMNet::operator=(const JKMNet& other){
    if (this == &other) return *this;
  else {
        
  }
  return *this;

}

/**
 * Helper function for schuffle
 */
// static Eigen::MatrixXd shuffleMatrix(const Eigen::MatrixXd &mat, const std::vector<int> &perm) {
//     Eigen::MatrixXd shuffled(mat.rows(), mat.cols());
//     for (std::size_t i = 0; i < perm.size(); ++i) {
//         shuffled.row(static_cast<int>(i)) = mat.row(perm[i]);
//     }
//     return shuffled;
// }

/**
 * Train an MLP with online Adam using calibrated matrices built from Data
 */
TrainingResult JKMNet::trainAdamOnlineSplit(
  MLP &mlp,
  Data &data,
  const std::vector<unsigned> &mlpArchitecture,
  const std::vector<int> &numbersOfPastVarsValues,
  activ_func_type activationType,
  weight_init_type weightsInitType,
  int maxIterations,
  double maxError,
  double learningRate,
  bool shuffle,
  unsigned rngSeed) 
  {
    TrainingResult result;

    // Basic check of architecture
    if (mlpArchitecture.empty())
        throw std::invalid_argument("mlpArchitecture must be non-empty");
    if (numbersOfPastVarsValues.empty())
        throw std::invalid_argument("numbersOfPastVarsValues must be non-empty");
    if (maxIterations <= 0)
        throw std::invalid_argument("maxIterations must be > 0");

    // Build calib mats inside Data
    data.makeCalibMatsSplit(numbersOfPastVarsValues, static_cast<int>(mlpArchitecture.back()));

    // Get calib matrices
    Eigen::MatrixXd X = data.getCalibInpsMat(); 
    Eigen::MatrixXd Y = data.getCalibOutsMat();

    // Shuffle rows
    std::vector<int> perm;
    if (shuffle) {
        perm = data.permutationVector(static_cast<int>(X.rows()));
        X = data.shuffleMatrix(X, perm);
        Y = data.shuffleMatrix(Y, perm);
    }

    // Configure MLP
    mlp.setArchitecture(const_cast<std::vector<unsigned>&>(mlpArchitecture));
    std::vector<activ_func_type> activations(mlpArchitecture.size(), activationType);
    std::vector<weight_init_type> weightInits(mlpArchitecture.size(), weightsInitType);
    mlp.setActivations(activations);
    mlp.setWInitType(weightInits);

    // Initialize MLP
    int inputSize = static_cast<int>(X.cols()); // each pattern is a row of inputs (flattened)
    Eigen::VectorXd zeroIn = Eigen::VectorXd::Zero(inputSize);
    mlp.initMLP(zeroIn, rngSeed);

    // Run training with onlineAdam method
    try {
        mlp.onlineAdam(maxIterations, maxError, learningRate, X, Y);
    } catch (const std::exception &ex) {
        std::cerr << "[trainAdamOnlineSplit] training failed: " << ex.what() << "\n";
        throw;
    }

    // Compute final loss on training set (simple MSE)
    Eigen::MatrixXd preds(X.rows(), Y.cols());
    for (int r = 0; r < X.rows(); ++r) {
        Eigen::VectorXd xcol = X.row(r).transpose();
        Eigen::VectorXd yhat = mlp.runMLP(xcol);
        preds.row(r) = yhat.transpose();
    }
    Eigen::MatrixXd diff = preds - Y;
    double mse = diff.array().square().mean();

    result.finalLoss = mse;
    result.iterations = maxIterations;
    result.converged = (mse <= maxError);

    return result;
}

/**
 * Train an MLP with batch Adam using calibrated matrices built from Data
 */
TrainingResult JKMNet::trainAdamBatchSplit(
  MLP &mlp,
  Data &data,
  const std::vector<unsigned> &mlpArchitecture,
  const std::vector<int> &numbersOfPastVarsValues,
  activ_func_type activationType,
  weight_init_type weightsInitType,
  int batchSize,
  int maxIterations,
  double maxError,
  double learningRate,
  bool shuffle,
  unsigned rngSeed) 
  {
    TrainingResult result;

    // Basic check of architecture
    if (mlpArchitecture.empty())
        throw std::invalid_argument("mlpArchitecture must be non-empty");
    if (numbersOfPastVarsValues.empty())
        throw std::invalid_argument("numbersOfPastVarsValues must be non-empty");
    if (maxIterations <= 0 || batchSize <= 0)
        throw std::invalid_argument("maxIterations and batchSize must be > 0");

    // Build calib mats inside Data
    data.makeCalibMatsSplit(numbersOfPastVarsValues, static_cast<int>(mlpArchitecture.back()));

    // Get calib matrices
    Eigen::MatrixXd X = data.getCalibInpsMat(); 
    Eigen::MatrixXd Y = data.getCalibOutsMat();

    // Shuffle rows
    std::vector<int> perm;
    if (shuffle) {
        perm = data.permutationVector(static_cast<int>(X.rows()));
        X = data.shuffleMatrix(X, perm);
        Y = data.shuffleMatrix(Y, perm);
    }

    // Configure MLP
    mlp.setArchitecture(const_cast<std::vector<unsigned>&>(mlpArchitecture));
    std::vector<activ_func_type> activations(mlpArchitecture.size(), activationType);
    std::vector<weight_init_type> weightInits(mlpArchitecture.size(), weightsInitType);
    mlp.setActivations(activations);
    mlp.setWInitType(weightInits);

    // Initialize MLP
    int inputSize = static_cast<int>(X.cols()); // each pattern is a row of inputs (flattened)
    Eigen::VectorXd zeroIn = Eigen::VectorXd::Zero(inputSize);
    mlp.initMLP(zeroIn, rngSeed);

    // Run training with batchAdam method
    try {
        mlp.batchAdam(maxIterations, maxError, batchSize, learningRate, X, Y);
    } catch (const std::exception &ex) {
        std::cerr << "[trainAdamBatchSplit] training failed: " << ex.what() << "\n";
        throw;
    }

    // Compute final loss on training set (simple MSE)
    Eigen::MatrixXd preds(X.rows(), Y.cols());
    for (int r = 0; r < X.rows(); ++r) {
        Eigen::VectorXd xcol = X.row(r).transpose();
        Eigen::VectorXd yhat = mlp.runMLP(xcol);
        preds.row(r) = yhat.transpose();
    }
    Eigen::MatrixXd diff = preds - Y;
    double mse = diff.array().square().mean();

    result.finalLoss = mse;
    result.iterations = maxIterations;
    result.converged = (mse <= maxError);

    return result;
}

/**
 * K-fold validation (online Adam) 
 */
void JKMNet::KFold(
    Data &data,
    const std::vector<unsigned> &mlpArchitecture,
    const std::vector<int> &numbersOfPastVarsValues,
    activ_func_type activationType,
    weight_init_type weightsInitType,
    int kFolds,
    bool shuffle,
    bool largerPieceCalib,
    unsigned seed,
    int maxIterations,
    double maxError,
    double learningRate,
    int runsPerFold)
{

    // Basic check of architecture
    if (mlpArchitecture.empty())
        throw std::invalid_argument("mlpArchitecture must be non-empty");
    if (numbersOfPastVarsValues.empty())
        throw std::invalid_argument("numbersOfPastVarsValues must be non-empty");
    if (maxIterations <= 0)
        throw std::invalid_argument("maxIterations must be > 0");

    // Set MLP
    MLP testMlp;
    testMlp.setArchitecture(const_cast<std::vector<unsigned>&>(mlpArchitecture));
    std::vector<activ_func_type> activations(mlpArchitecture.size(), activationType);
    std::vector<weight_init_type> weightInits(mlpArchitecture.size(), weightsInitType);
    testMlp.setActivations(activations);
    testMlp.setWInitType(weightInits);

    std::vector<double> foldsMse;

    for(int foldIdx = 0; foldIdx < kFolds; foldIdx++){
        double meanFoldMse = 0.0; 
        auto [trainInps, trainOuts, validInps, validOuts] = data.makeKFoldMats(numbersOfPastVarsValues,
                                                                    static_cast<int>(mlpArchitecture.back()),
                                                                    kFolds,
                                                                    foldIdx,
                                                                    shuffle,
                                                                    largerPieceCalib,
                                                                    seed);
        for(int run = 0; run < runsPerFold; run++){
            // Initialize MLP
            int inputSize = static_cast<int>(trainInps.cols()); // each pattern is a row of inputs (flattened)
            Eigen::VectorXd zeroIn = Eigen::VectorXd::Zero(inputSize);
            testMlp.initMLP(zeroIn, seed);

            // Run training with onlineAdam method
            try {
                testMlp.onlineAdam(maxIterations, maxError, learningRate, trainInps, trainOuts);
            } catch (const std::exception &ex) {
                std::cerr << "[trainAdamOnlineSplit] training failed: " << ex.what() << "\n";
                throw;
            }

            // Validation
            testMlp.calculateOutputs(validInps);
            meanFoldMse += Metrics::mse(data.inverseTransformOutputs(validOuts),
                                        data.inverseTransformOutputs(testMlp.getOutputs()));
        }
        foldsMse.push_back(meanFoldMse / runsPerFold);
    }
    for(int fold = 0; fold < kFolds; fold++){
        std::cout<<"Fold "<<fold<<" validation mean MSE = "<<foldsMse[fold]<<"\n";
    }
}

/**
 * Setter for number of MLPs
 */
void JKMNet::setNmlps(unsigned nmlp){
    Nmlps = nmlp;
}

/**
 * Getter for number of MLPs
 */
unsigned JKMNet::getNmlps(){
    return Nmlps;
}

/**
 * Initialization of MLPs vector
 */
void JKMNet::init_mlps(){
    mlps_.clear();
    mlps_ = std::vector<MLP>(Nmlps);
    MLP setMlp;
    setMlp.setArchitecture(cfg_.mlp_architecture);
    setMlp.setActivations(std::vector<activ_func_type>(cfg_.mlp_architecture.size(), strToActivation(cfg_.activation)));
    setMlp.setWInitType(std::vector<weight_init_type>(cfg_.mlp_architecture.size(), strToWeightInit(cfg_.weight_init)));
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(std::accumulate(cfg_.input_numbers.begin(),cfg_.input_numbers.end(),0,
                        [](int s, const std::vector<int>& v) {return s + static_cast<int>(v.size());}));
    
    #pragma omp parallel for num_threads(nthreads_)
    for(unsigned i = 0; i < Nmlps; i++){
        mlps_[i] = setMlp;
        mlps_[i].initMLP(x0, cfg_.seed);
    }
}

/**
 * Ensemble run - load settings, read data, train, test
 */
void JKMNet::ensembleRunMlpVector(){
    std::cout << "The number of threads is " << nthreads_ << std::endl;

    // ------------------------------------------------------
    // Load & preprocess data
    // ------------------------------------------------------
    std::unordered_set<std::string> idFilter;
    if (!cfg_.id.empty()) {
        std::vector<std::string> ids = parseStringList(cfg_.id);
        idFilter = std::unordered_set<std::string>(ids.begin(), ids.end());
    }

    std::cout << "-> Loading data..." << std::endl;
    data_.loadFilteredCSV(cfg_.data_file, idFilter, cfg_.columns, cfg_.timestamp, cfg_.id_col);
    std::cout << "-> Data loaded." << std::endl;

    std::cout << "-> Transforming data..." << std::endl;
    data_.setTransform(strToTransformType(cfg_.transform),
                       cfg_.transform_alpha,
                       cfg_.exclude_last_col_from_transform);
    data_.applyTransform();
    std::cout << "-> Data transformed." << std::endl;

    auto [X_train, Y_train, X_valid, Y_valid, pat_indices, calIdxForUnshuffle] = data_.makeMats(cfg_.input_numbers,
                                                                                static_cast<int>(cfg_.mlp_architecture.back()),
                                                                                cfg_.train_fraction,
                                                                                cfg_.shuffle,
                                                                                cfg_.seed);

    std::cout << "-> Data split into training and validation sets." << std::endl;

    std::vector<std::string> colNames;
    for (int c = 0; c < Y_train.cols(); ++c) {
        colNames.push_back("h" + std::to_string(c+1));
    }

    // Save real data
    Eigen::MatrixXd Y_true_calib_save = data_.unshuffleMatrix(Y_train, calIdxForUnshuffle);
    Eigen::MatrixXd Y_true_valid_save = Y_valid;
    try {
        Y_true_calib_save = data_.inverseTransformOutputs(Y_true_calib_save);
        Y_true_valid_save = data_.inverseTransformOutputs(Y_true_valid_save);
    } catch (const std::exception &ex) {
        std::cerr << "[Warning] inverseTransformOutputs failed (save GT): " << ex.what() << "\n";
    }
    data_.saveMatrixCsv(cfg_.real_calib, Y_true_calib_save, colNames);
    data_.saveMatrixCsv(cfg_.real_valid, Y_true_valid_save, colNames);
    data_.saveVector(pat_indices, cfg_.pattern_indices);
    std::cout << "-> Real calibration and validation data saved." << std::endl;

    // Configure MLP
    setNmlps(cfg_.ensemble_runs);
    init_mlps();

    std::cout << "-> Ensemble run starting..." << std::endl;
    // ------------------------------------------------------
    // Ensemble loop
    // ------------------------------------------------------
    #pragma omp parallel for num_threads(nthreads_)
    for (unsigned run = 0; run < mlps_.size() ; ++run) {
        std::string run_id = std::to_string(run+1);  // string
        unsigned run_id_integer = run + 1; // integer

        // Check, if exists folder 'outputs/logs'
        if (!std::filesystem::exists(cfg_.log_dir)) {
            std::filesystem::create_directories(cfg_.log_dir);
        }
        // Set the output file for the log messages        
        std::string filename = cfg_.log_dir + "log_run" + run_id + ".log";
        std::ofstream logFile(filename);

        // Log all settings 
        data_.logRunSettings(logFile, cfg_, run_id_integer);

        logFile << "Run " << run_id << " starting...\n";

        // Save init weights
        mlps_[run].saveWeightsCsv(Metrics::addRunIdToFilename(cfg_.weights_csv_init, run_id));
        mlps_[run].weightsToVectorMlp();
        mlps_[run].saveWeightsVectorCsv(Metrics::addRunIdToFilename(cfg_.weights_vec_csv_init, run_id));
        //mlps_[run].appendWeightsVectorCsv(cfg_.weights_vec_csv_init, run == 0);  // all init weights in one file
        logFile << "-> Initial weights saved.\n";

        // Train
        logFile << "-> Training starting...\n";

        std::vector<Eigen::MatrixXd> resultMetrics;
        TrainerType trainerType = strToTrainerType(cfg_.trainer);
        int runIndex = run + 1;

        std::string mseCalFile   = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "MSE_epochs");
        std::string rmseCalFile  = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "RMSE_epochs");
        std::string piCalFile    = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "PI_epochs");
        std::string nsCalFile    = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "NS_epochs");
        std::string kgeCalFile   = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "KGE_epochs");
        std::string pbiasCalFile = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "PBIAS_epochs");
        std::string rsrCalFile   = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "RSR_epochs");
        std::string mseValFile   = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "MSE_epochs");
        std::string rmseValFile  = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "RMSE_epochs");
        std::string piValFile    = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "PI_epochs");
        std::string nsValFile    = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "NS_epochs");
        std::string kgeValFile   = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "KGE_epochs");
        std::string pbiasValFile = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "PBIAS_epochs");
        std::string rsrValFile   = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "RSR_epochs");
        
        // Switch for trainer type
        switch (trainerType) {
            case TrainerType::ONLINE:
                mlps_[run].onlineAdam(
                    cfg_.max_iterations, cfg_.max_error,
                    cfg_.learning_rate,  X_train, Y_train
                );
                break;

            case TrainerType::BATCH:
                mlps_[run].batchAdam(
                    cfg_.max_iterations, cfg_.max_error,
                    cfg_.batch_size, cfg_.learning_rate,
                    X_train, Y_train
                );
                break;

            case TrainerType::ONLINE_EPOCH:
                resultMetrics = mlps_[run].onlineAdamEpochVal(
                    X_train, Y_train, X_valid, Y_valid,
                    cfg_.max_iterations, cfg_.learning_rate, cfg_.max_metrics_step
                );
                Metrics::saveMetricsCsv(mseCalFile, resultMetrics[0]);
                Metrics::saveMetricsCsv(rmseCalFile, resultMetrics[1]);
                Metrics::saveMetricsCsv(piCalFile, resultMetrics[2]);
                Metrics::saveMetricsCsv(nsCalFile, resultMetrics[3]);
                Metrics::saveMetricsCsv(kgeCalFile, resultMetrics[4]);
                Metrics::saveMetricsCsv(pbiasCalFile, resultMetrics[5]);
                Metrics::saveMetricsCsv(rsrCalFile, resultMetrics[6]);
                Metrics::saveMetricsCsv(mseValFile, resultMetrics[7]);
                Metrics::saveMetricsCsv(rmseValFile, resultMetrics[8]);
                Metrics::saveMetricsCsv(piValFile, resultMetrics[9]);
                Metrics::saveMetricsCsv(nsValFile, resultMetrics[10]);
                Metrics::saveMetricsCsv(kgeValFile, resultMetrics[11]);
                Metrics::saveMetricsCsv(pbiasValFile, resultMetrics[12]);
                Metrics::saveMetricsCsv(rsrValFile, resultMetrics[13]);
                break;

            case TrainerType::BATCH_EPOCH:
                resultMetrics = mlps_[run].batchAdamEpochVal(
                    X_train, Y_train, X_valid, Y_valid,
                    cfg_.batch_size, cfg_.max_iterations,
                    cfg_.learning_rate, cfg_.max_metrics_step
                );
                Metrics::saveMetricsCsv(mseCalFile, resultMetrics[0]);
                Metrics::saveMetricsCsv(rmseCalFile, resultMetrics[1]);
                Metrics::saveMetricsCsv(piCalFile, resultMetrics[2]);
                Metrics::saveMetricsCsv(nsCalFile, resultMetrics[3]);
                Metrics::saveMetricsCsv(kgeCalFile, resultMetrics[4]);
                Metrics::saveMetricsCsv(pbiasCalFile, resultMetrics[5]);
                Metrics::saveMetricsCsv(rsrCalFile, resultMetrics[6]);
                Metrics::saveMetricsCsv(mseValFile, resultMetrics[7]);
                Metrics::saveMetricsCsv(rmseValFile, resultMetrics[8]);
                Metrics::saveMetricsCsv(piValFile, resultMetrics[9]);
                Metrics::saveMetricsCsv(nsValFile, resultMetrics[10]);
                Metrics::saveMetricsCsv(kgeValFile, resultMetrics[11]);
                Metrics::saveMetricsCsv(pbiasValFile, resultMetrics[12]);
                Metrics::saveMetricsCsv(rsrValFile, resultMetrics[13]);
                break;

            default:
                throw std::invalid_argument("Unknown trainer type");
        }


        logFile << "-> Training finished.\n";

        // Evaluate calibration
        logFile << "-> Evaluating calibration set...\n";
        mlps_[run].calculateOutputs(X_train);
        Eigen::MatrixXd Y_pred_calib = mlps_[run].getOutputs();
        Eigen::MatrixXd Y_true_calib = Y_train;
        try {
            Y_true_calib = data_.inverseTransformOutputs(Y_true_calib);
            Y_pred_calib = data_.inverseTransformOutputs(Y_pred_calib);
        } catch (...) {}

        // One file for each metrics per each run
        std::vector<double> mseVals, rmseVals, piVals, nsVals, kgeVals, pbiasVals, rsrVals;

        for (int c = 0; c < Y_true_calib.cols(); ++c) {
            mseVals.push_back(Metrics::mse(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            rmseVals.push_back(Metrics::rmse(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            piVals.push_back(Metrics::pi(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            nsVals.push_back(Metrics::ns(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            kgeVals.push_back(Metrics::kge(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            pbiasVals.push_back(Metrics::pbias(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
            rsrVals.push_back(Metrics::rsr(Y_true_calib.col(c).eval(), Y_pred_calib.col(c).eval()));
        }
        // per-metric filenames for this run
        std::string mseFile   = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "MSE");
        std::string rmseFile  = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "RMSE");
        std::string piFile    = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "PI");
        std::string nsFile    = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "NS");
        std::string kgeFile   = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "KGE");
        std::string pbiasFile = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "PBIAS");
        std::string rsrFile   = Metrics::makeMetricFilename(cfg_.metrics_cal, runIndex, "RSR");
        // always write header once per file (this is one run)
        Metrics::saveMetricRow(mseFile,   colNames, mseVals,   true);
        Metrics::saveMetricRow(rmseFile,  colNames, rmseVals,  true);
        Metrics::saveMetricRow(piFile,    colNames, piVals,    true);
        Metrics::saveMetricRow(nsFile,    colNames, nsVals,    true);
        Metrics::saveMetricRow(kgeFile,   colNames, kgeVals,   true);
        Metrics::saveMetricRow(pbiasFile, colNames, pbiasVals, true);
        Metrics::saveMetricRow(rsrFile,   colNames, rsrVals,   true);


        data_.saveMatrixCsv(Metrics::addRunIdToFilename(cfg_.pred_calib, run_id), 
                            data_.unshuffleMatrix(Y_pred_calib, calIdxForUnshuffle), 
                            colNames);

        logFile << "-> Calibration metrics and predictions saved.\n";

        // Save final weights
        mlps_[run].saveWeightsCsv(Metrics::addRunIdToFilename(cfg_.weights_csv, run_id));
        mlps_[run].weightsToVectorMlp();
        mlps_[run].saveWeightsVectorCsv(Metrics::addRunIdToFilename(cfg_.weights_vec_csv, run_id));
        // mlps_[run].appendWeightsVectorCsv(cfg_.weights_vec_csv, run == 0);  // all final weights in one file
        logFile << "-> Final weights saved.\n";

        // TODO: predikce, tedy samostatna metoda 
        // Evaluate validation
        logFile << "-> Evaluating validation set...\n";
        mlps_[run].calculateOutputs(X_valid);
        Eigen::MatrixXd Y_pred_valid = mlps_[run].getOutputs();
        Eigen::MatrixXd Y_true_valid = Y_valid;
        try {
            Y_true_valid = data_.inverseTransformOutputs(Y_true_valid);
            Y_pred_valid = data_.inverseTransformOutputs(Y_pred_valid);
        } catch (...) {}

        // One file for each metrics per each run
        std::vector<double> mseValsV, rmseValsV, piValsV, nsValsV, kgeValsV, pbiasValsV, rsrValsV;

        for (int c = 0; c < Y_true_valid.cols(); ++c) {
            mseValsV.push_back(Metrics::mse(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            rmseValsV.push_back(Metrics::rmse(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            piValsV.push_back(Metrics::pi(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            nsValsV.push_back(Metrics::ns(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            kgeValsV.push_back(Metrics::kge(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            pbiasValsV.push_back(Metrics::pbias(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
            rsrValsV.push_back(Metrics::rsr(Y_true_valid.col(c).eval(), Y_pred_valid.col(c).eval()));
        }

        std::string mseFileV   = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "MSE");
        std::string rmseFileV  = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "RMSE");
        std::string piFileV    = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "PI");
        std::string nsFileV    = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "NS");
        std::string kgeFileV   = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "KGE");
        std::string pbiasFileV = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "PBIAS");
        std::string rsrFileV   = Metrics::makeMetricFilename(cfg_.metrics_val, runIndex, "RSR");

        Metrics::saveMetricRow(mseFileV,   colNames, mseValsV,   true);
        Metrics::saveMetricRow(rmseFileV,  colNames, rmseValsV,  true);
        Metrics::saveMetricRow(piFileV,    colNames, piValsV,    true);
        Metrics::saveMetricRow(nsFileV,    colNames, nsValsV,    true);
        Metrics::saveMetricRow(kgeFileV,   colNames, kgeValsV,   true);
        Metrics::saveMetricRow(pbiasFileV, colNames, pbiasValsV, true);
        Metrics::saveMetricRow(rsrFileV,   colNames, rsrValsV,   true);

        
        data_.saveMatrixCsv(Metrics::addRunIdToFilename(cfg_.pred_valid, run_id), Y_pred_valid, colNames);
        logFile << "-> Validation metrics and predictions saved.\n";

        logFile << "Run " << run_id << " finished.\n";
        logFile << "-------------------------------------------\n";

    }

    std::cout << "-> Ensemble run finished." << std::endl;

    std::cout << "\n===========================================\n";
    std::cout << " Running Ensemble finished \n";
    std::cout << "===========================================\n";
}

/**
 * Predict outputs from saved weights
 */
void JKMNet::predictFromSavedWeights(const std::string &weightsPath)
{
    std::cout << " Prediction from saved weights\n";

    // Load and transform data
    std::unordered_set<std::string> idFilter;
    if (!cfg_.id.empty()) {
        auto ids = parseStringList(cfg_.id);
        idFilter = std::unordered_set<std::string>(ids.begin(), ids.end());
    }

    std::cout << "-> Loading data...\n";
    data_.loadFilteredCSV(cfg_.data_file, idFilter, cfg_.columns, cfg_.timestamp, cfg_.id_col);
    std::cout << "-> Data loaded.\n";

    std::cout << "-> Transforming data...\n";
    data_.setTransform(strToTransformType(cfg_.transform),
                       cfg_.transform_alpha,
                       cfg_.exclude_last_col_from_transform);
    data_.applyTransform();
    std::cout << "-> Data transformed.\n";

    auto [X_train, Y_train, X_valid, Y_valid, pat_indices, calIdxForUnshuffle] = data_.makeMats(cfg_.input_numbers,
                                                                                static_cast<int>(cfg_.mlp_architecture.back()),
                                                                                cfg_.train_fraction,
                                                                                cfg_.shuffle,
                                                                                cfg_.seed);
    Eigen::MatrixXd X_pred = X_valid;
    Eigen::MatrixXd Y_true = Y_valid;

    std::vector<std::string> colNames;
    for (int c = 0; c < Y_true.cols(); ++c) {
        colNames.push_back("h" + std::to_string(c+1));
    }

    // Reconstruct the network
    std::cout << "-> Initializing MLP...\n";
    MLP mlp;
    mlp.setArchitecture(cfg_.mlp_architecture);

    std::vector<activ_func_type> activs(cfg_.mlp_architecture.size(), strToActivation(cfg_.activation));
    mlp.setActivations(activs);

    std::vector<weight_init_type> wInits(cfg_.mlp_architecture.size(), strToWeightInit(cfg_.weight_init));
    mlp.setWInitType(wInits);

    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(cfg_.mlp_architecture.front());
    mlp.initMLP(x0, cfg_.seed);

    std::cout << "-> Loading weights from: " << weightsPath << "\n";
    if (!mlp.loadWeightsCsv(weightsPath, "")) {
        throw std::runtime_error("Failed to load weights file: " + weightsPath);
    }

    // Predict
    std::cout << "-> Calculating outputs...\n";
    mlp.calculateOutputs(X_pred);
    Eigen::MatrixXd Y_pred = mlp.getOutputs();

    try {
        Y_true = data_.inverseTransformOutputs(Y_true);
        Y_pred = data_.inverseTransformOutputs(Y_pred);
    } catch (const std::exception &ex) {
        std::cerr << "[Warning] inverseTransformOutputs failed: " << ex.what() << "\n";
    }

    // Save results
    std::cout << "-> Saving predictions...\n";
    data_.saveMatrixCsv(cfg_.pred_valid, Y_pred, colNames);

    // Compute metrics
    std::vector<double> rmseVals;
    for (int c = 0; c < Y_true.cols(); ++c) {
        rmseVals.push_back(Metrics::rmse(Y_true.col(c).eval(), Y_pred.col(c).eval()));
    }
    Metrics::saveMetricRow(cfg_.metrics_val, colNames, rmseVals, true);

    std::cout << "-> Prediction completed successfully.\n";
}