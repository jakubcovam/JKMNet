#include "MLP.hpp"

#include <iostream>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cerrno>
#include <cstring>

using namespace std;
using namespace std::chrono;

/**
 * The constructor
 */
MLP::MLP()
    : nNeurons(),
      numLayers(0),
      Inps(),
      activFuncs(),
      wInitTypes(),
      layers_(),
      output(),
      outputMat(),
      weightsVectorMlp(),
      calibCrit(),
      validCrit(),
      result()
       {
   
}


/**
 * The destructor
 */
MLP::~MLP(){

}

/**
 * The copy constructor
 */
MLP::MLP(const MLP& other)
    : nNeurons(other.nNeurons),
      numLayers(other.numLayers),
      Inps(other.Inps),
      activFuncs(other.activFuncs),
      wInitTypes(other.wInitTypes),
      layers_(other.layers_),
      output(other.output),
      outputMat(other.outputMat),
      weightsVectorMlp(other.weightsVectorMlp),
      calibCrit(other.calibCrit),
      validCrit(other.validCrit),
      result(other.result) {
    // deep copies handled automatically by Eigen and std::vector
}


/**
 * The assignment operator
 */
MLP& MLP::operator=(const MLP& other){
    if (this == &other) return *this;
  else {
    nNeurons        = other.nNeurons;
        numLayers       = other.numLayers;
        Inps            = other.Inps;
        activFuncs      = other.activFuncs;
        wInitTypes      = other.wInitTypes;
        layers_         = other.layers_;
        output          = other.output;
        outputMat       = other.outputMat;
        weightsVectorMlp = other.weightsVectorMlp;
        calibCrit = other.calibCrit;
        validCrit = other.validCrit;
        result = other.result;
  }
  return *this;

}

/**
 *  Move constructor
 */
MLP::MLP(MLP&& other) noexcept
    : nNeurons(std::move(other.nNeurons)),
      numLayers(other.numLayers),
      Inps(std::move(other.Inps)),
      activFuncs(std::move(other.activFuncs)),
      wInitTypes(std::move(other.wInitTypes)),
      layers_(std::move(other.layers_)),
      output(std::move(other.output)),
      outputMat(std::move(other.outputMat)),
      weightsVectorMlp(std::move(other.weightsVectorMlp)),
      calibCrit(std::move(other.calibCrit)),
      validCrit(std::move(other.validCrit)),
      result(std::move(other.result)) {
    
}

/**
 *  Move assignment operator
 */
MLP& MLP::operator=(MLP&& other) noexcept {
    if (this != &other) {
        nNeurons       = std::move(other.nNeurons);
        numLayers      = other.numLayers;
        Inps           = std::move(other.Inps);
        activFuncs     = std::move(other.activFuncs);
        wInitTypes     = std::move(other.wInitTypes);
        layers_        = std::move(other.layers_);
        output         = std::move(other.output);
        outputMat      = std::move(other.outputMat);
        weightsVectorMlp = std::move(other.weightsVectorMlp);
        calibCrit = std::move(other.calibCrit);
        validCrit = std::move(other.validCrit);
        result = std::move(other.result);
    }
    return *this;
}

/**
 *  Getter: Returns the current architecture (vector of neurons in each layer)
 */
std::vector<unsigned> MLP::getArchitecture() {
    return nNeurons;
}

/**
 *  Setter: Sets the architecture from a vector
 */
void MLP::setArchitecture(const std::vector<unsigned>& architecture) {
    nNeurons = architecture;
    numLayers = nNeurons.size();
    // if user hasn’t supplied activations yet, give them a default (e.g. ReLU)
    if (activFuncs.size() != architecture.size()) {
        activFuncs.assign(architecture.size(), activ_func_type::RELU);
    }
}

/**
 *  Print the architecture
 */
void MLP::printArchitecture() {
    std::cout << "MLP Architecture: ";
    numLayers = nNeurons.size();

    for (size_t i = 0; i < numLayers; ++i) {
        std::cout << nNeurons[i];
        if (i != numLayers - 1) {
            std::cout << " -> ";
        }
    }

    std::cout << std::endl;
}

/**
 *  Getter: Returns the current activation functions for each layer
 */
std::vector<activ_func_type> MLP::getActivations() {
    return activFuncs;
}

/**
 *  Setter: Sets the activation functions
 */
void MLP::setActivations(const std::vector<activ_func_type>& funcs) {
    if (funcs.size() != nNeurons.size()) {
        throw std::invalid_argument("[MLP] Activation vector length must match number of layers");
    }
    activFuncs = funcs;
}

/**
 *  Print the activation functions
 */
void MLP::printActivations() {
    if (activFuncs.empty()) {
        std::cout << "No activation functions set.\n";
        return;
    }
    std::cout << "Activations per layer:\n";
    for (size_t i = 0; i < activFuncs.size(); ++i) {
        std::cout << "  Layer " << i
                  << " (" << nNeurons[i] << " neurons): "
                  << Layer::activationName(activFuncs[i]) 
                  << "\n";
    }
}

/**
 *  Getter: Returns the weight initialization type for each layer
 */
std::vector<weight_init_type> MLP::getWInitType() {
    return wInitTypes;
}

/**
 *  Setter: Sets the weight initialization type
 */
void MLP::setWInitType(const std::vector<weight_init_type>& wInits) {
    if (wInits.size() != nNeurons.size()) {
        throw std::invalid_argument(
          "[MLP] wInitTypes length must equal number of layers (" +
          std::to_string(nNeurons.size()) + ")");
    }
    wInitTypes = wInits;
}

/**
 *  Print the weight initialization type
 */
void MLP::printWInitType() {
    if (wInitTypes.empty()) {
        std::cout << "No weight init types set.\n";
        return;
    }
    std::cout << "Weight initialization per layer:\n";
    for (size_t i = 0; i < wInitTypes.size(); ++i) {
        std::cout << "  Layer " << i
                  << " (" << nNeurons[i] << " neurons): "
                  << Layer::wInitTypeName(wInitTypes[i]) 
                  << "\n";
    }
}

/**
 * Getter for the number of layers
 */
size_t MLP::getNumLayers() const {
    return numLayers; 
}

/**
 * Setter for the number of layers
 */
void MLP::setNumLayers(size_t layers) {
    numLayers = layers; 
  
    if (nNeurons.size() != layers) {
        // Error message if number of layers and architecture do not match
        std::cerr << "-> [Error]: Number of layers does not match the architecture!" << std::endl;
    }
    else { 
        // Positive feedback message
        std::cout << "-> [Info]: Number of layers matches the architecture." << std::endl;
    }
}

/**
 * Getter for the number of inputs
 */
unsigned MLP::getNumInputs() {
    if (nNeurons.empty()) {
        return 0u;  // an unsigned zero
    }

    return nNeurons.front();  // first element of the vector
}

/**
 * Setter for the number of inputs
 */
void MLP::setNumInputs(unsigned inputs) {
    // if no layers are defined yet, we create the first layer and set its size to 'inputs'
    if (nNeurons.empty()) {
        nNeurons.push_back(inputs);

    // if we already have at least one layer, overwrite the neuron count of that first layer
    } else {
        nNeurons[0] = inputs;
    }

    numLayers = nNeurons.size();
}

/**
 * Get the number of neurons at a specific layer index
 */
unsigned MLP::getNumNeuronsInLayers(std::size_t index) {
    if (index >= nNeurons.size()) {
        throw std::out_of_range("[Error]: Layer index out of range");
    }
    return nNeurons[index];
}

/**
 * Set the number of neurons at a specific layer index
 */
void MLP::setNumNeuronsInLayers(std::size_t index, unsigned count) {
    if (index >= nNeurons.size()) {
        throw std::out_of_range("[Error]: Layer index out of range");
    }
    nNeurons[index] = count;
}

/**
 * Getter for the inputs
 */
Eigen::VectorXd& MLP::getInps() {
    return Inps;
}

/**
 * Setter for the inputs
 */
void MLP::setInps(Eigen::VectorXd& inputs) {
    // Resize to (real inputs + bias):
    Inps.resize(inputs.size() + 1);
    Inps.head(inputs.size()) = inputs;       
    Inps(inputs.size()) = 1.0;           
}

/**
 * Getter for the weights
 */
Eigen::MatrixXd MLP::getWeights(size_t idx) const {
    if (idx >= layers_.size())
      throw std::out_of_range("Layer index out of range in getWeights");
    return layers_[idx].getWeights();
}

/**
 * Setter for the weights
 */
void MLP::setWeights(size_t idx, const Eigen::MatrixXd& W) {
    if (idx >= layers_.size())
      throw std::out_of_range("Layer index out of range in setWeights");
    layers_[idx].setWeights(W);
}

/**
 * Getter for weights vector of MLP
 */
Eigen::VectorXd MLP::getWeightsVectorMlp(){
    return weightsVectorMlp;
}

/**
 * Merge weight vectors of all layers
 */
void MLP::weightsToVectorMlp(){
    int length = 0;
    for(size_t i = 0; i < layers_.size(); i++){
        layers_[i].weightsToVector();
        length += layers_[i].getWeightsVector().size();
    }
    weightsVectorMlp = Eigen::VectorXd(length);
    
    int pos = 0;
    for(size_t i = 0; i < layers_.size(); i++){
        weightsVectorMlp.segment(pos, layers_[i].getWeightsVector().size()) = layers_[i].getWeightsVector();
        pos += layers_[i].getWeightsVector().size();
    }
}

/**
 * Save weights in readable CSV text (per-layer blocks)
 */
bool MLP::saveWeightsCsv(const std::string &path) const {
    namespace fs = std::filesystem;

    if (path.empty()) {
        std::cerr << "[MLP::saveWeightsCsv] Cannot open: path is empty\n";
        return false;
    }

    try {
        fs::path p(path);
        if (p.has_parent_path()) {
            fs::create_directories(p.parent_path()); // no-op if exists
        }
    } catch (const std::exception &e) {
        std::cerr << "[MLP::saveWeightsCsv] Cannot create parent directories for: " << path
                  << "  (" << e.what() << ")\n";
        return false;
    }

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        int e = errno;
        std::cerr << "[MLP::saveWeightsCsv] Cannot open file: " << path
                  << "  errno=" << e << " (" << std::strerror(e) << ")\n";
        return false;
    }

    ofs << std::setprecision(12);
    // Optionally write a small header
    ofs << "# MLP weights CSV\n";
    // Write per-layer weights; format: a comment line with layer index and dimensions, then rows of CSV
    for (size_t li = 0; li < layers_.size(); ++li) {
        const Eigen::MatrixXd &W = layers_[li].getWeights(); // ensure Layer::getWeights() is const
        ofs << "#layer," << li << "," << W.rows() << "," << W.cols() << "\n";
        for (Eigen::Index r = 0; r < W.rows(); ++r) {
            for (Eigen::Index c = 0; c < W.cols(); ++c) {
                ofs << W(r, c);
                if (c + 1 < W.cols()) ofs << ",";
            }
            ofs << "\n";
        }
    }
    ofs.close();
    if (ofs.fail()) {
        std::cerr << "[MLP::saveWeightsCsv] Write failure for: " << path << "\n";
        return false;
    }
    return true;
}

/**
 * Save weights in compact binary
 */
bool MLP::saveWeightsBinary(const std::string &path) const {
    try {
        std::filesystem::path p(path);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) { std::cerr << "[MLP::saveWeightsBinary] Cannot open: " << path << "\n"; return false; }
        size_t L = getNumLayers();
        ofs.write(reinterpret_cast<const char*>(&L), sizeof(L));
        for (size_t i = 0; i < L; ++i) {
            Eigen::MatrixXd W = getWeights(i);
            uint64_t rows = static_cast<uint64_t>(W.rows());
            uint64_t cols = static_cast<uint64_t>(W.cols());
            ofs.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
            ofs.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
            // write doubles row-major
            for (Eigen::Index r = 0; r < W.rows(); ++r) {
                for (Eigen::Index c = 0; c < W.cols(); ++c) {
                    double v = W(r,c);
                    ofs.write(reinterpret_cast<const char*>(&v), sizeof(v));
                }
            }
        }
        ofs.close();
        return true;
    } catch (const std::exception &ex) {
        std::cerr << "[MLP::saveWeightsBinary] Exception: " << ex.what() << "\n";
        return false;
    }
}

/**
 * Save vector of weights in readable CSV text (per-layer blocks)
 */
bool MLP::saveWeightsVectorCsv(const std::string &path) const {
    namespace fs = std::filesystem;
    if (path.empty()) {
        std::cerr << "[MLP::saveWeightsVectorCsv] Path is empty\n";
        return false;
    }
    try {
        fs::path p(path);
        if (p.has_parent_path()) fs::create_directories(p.parent_path());
    } catch (const std::exception &e) {
        std::cerr << "[MLP::saveWeightsVectorCsv] Cannot create parent directories: "
                  << e.what() << "\n";
        return false;
    }

    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cerr << "[MLP::saveWeightsVectorCsv] Cannot open file: " << path << "\n";
        return false;
    }
    ofs << std::setprecision(12);

    // save as column vector
    for (int i = 0; i < weightsVectorMlp.size(); ++i) {
        ofs << weightsVectorMlp[i] << "\n";
    }

    // save as row vector
    // for (int i = 0; i < weightsVectorMlp.size(); ++i) {
    //     ofs << weightsVectorMlp[i];
    //     if (i + 1 < weightsVectorMlp.size()) ofs << ",";
    // }

    ofs << "\n";
    ofs.close();
    return !ofs.fail();
}

/**
 * Save vector of weights in compact binary
 */
bool MLP::saveWeightsVectorBinary(const std::string &path) const {
    try {
        std::filesystem::path p(path);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
        std::ofstream ofs(path, std::ios::binary);
        if (!ofs.is_open()) {
            std::cerr << "[MLP::saveWeightsVectorBinary] Cannot open: " << path << "\n";
            return false;
        }
        int64_t len = weightsVectorMlp.size();
        ofs.write(reinterpret_cast<const char*>(&len), sizeof(len));
        ofs.write(reinterpret_cast<const char*>(weightsVectorMlp.data()), len * sizeof(double));
        ofs.close();
        return true;
    } catch (const std::exception &ex) {
        std::cerr << "[MLP::saveWeightsVectorBinary] Exception: " << ex.what() << "\n";
        return false;
    }
}

/**
 * Save vector of weights inside one file for ensemble run
 */
bool MLP::appendWeightsVectorCsv(const std::string &path, bool isFirstRun) const {
    namespace fs = std::filesystem;
    if (path.empty()) {
        std::cerr << "[MLP::appendWeightsVectorCsv] Path is empty\n";
        return false;
    }
    try {
        fs::path p(path);
        if (p.has_parent_path()) fs::create_directories(p.parent_path());
    } catch (const std::exception &e) {
        std::cerr << "[MLP::appendWeightsVectorCsv] Cannot create parent directories: "
                  << e.what() << "\n";
        return false;
    }

    // On first run: create/truncate and write column 1
    if (isFirstRun) {
        std::ofstream ofs(path, std::ios::out | std::ios::trunc);
        if (!ofs.is_open()) {
            std::cerr << "[MLP::appendWeightsVectorCsv] Cannot open file: " << path << "\n";
            return false;
        }
        ofs << std::setprecision(12);
        for (int i = 0; i < weightsVectorMlp.size(); ++i) {
            ofs << weightsVectorMlp[i] << "\n";
        }
        ofs.close();
        return true;
    }

    // For subsequent runs: read, append new column, overwrite
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::cerr << "[MLP::appendWeightsVectorCsv] Cannot open file for appending: " << path << "\n";
        return false;
    }

    std::vector<std::vector<double>> matrix;
    std::string line;
    while (std::getline(ifs, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        double val;
        while (ss >> val) {
            row.push_back(val);
            if (ss.peek() == ',') ss.ignore();
        }
        matrix.push_back(row);
    }
    ifs.close();

    if ((int)matrix.size() != weightsVectorMlp.size()) {
        std::cerr << "[MLP::appendWeightsVectorCsv] Mismatch in vector size vs file rows\n";
        return false;
    }

    for (int i = 0; i < weightsVectorMlp.size(); ++i) {
        matrix[i].push_back(weightsVectorMlp[i]);
    }

    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
        std::cerr << "[MLP::appendWeightsVectorCsv] Cannot reopen file: " << path << "\n";
        return false;
    }
    ofs << std::setprecision(12);
    for (const auto &row : matrix) {
        for (size_t j = 0; j < row.size(); ++j) {
            ofs << row[j];
            if (j + 1 < row.size()) ofs << ",";
        }
        ofs << "\n";
    }
    ofs.close();
    return true;
}

/**
 * Load weights from CSV text (per-layer blocks)
 */
bool MLP::loadWeightsCsv(const std::string &wPath, const std::string &confPath) {
    if (wPath.empty()) {
        std::cerr << "[MLP::loadWeightsCsv] wPath is empty\n";
        return false;
    }

    std::ifstream ifs(wPath);
    if (!ifs.is_open()) {
        int e = errno;
        std::cerr << "[MLP::loadWeightsCsv] Cannot open file: " << wPath
                  << " errno=" << e << " (" << std::strerror(e) << ")\n";
        return false;
    }

    // Parse CSV structure
    std::string line;
    struct LayerInfo {
        size_t idx;
        size_t rows;
        size_t cols;
        Eigen::MatrixXd W;
    };
    std::vector<LayerInfo> infos;

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            if (line.rfind("#layer", 0) == 0) {
                std::stringstream ss(line);
                std::string token;
                std::getline(ss, token, ','); // "#layer"
                size_t idx=0, rows=0, cols=0;
                if (!(std::getline(ss, token, ',') && (idx = std::stoul(token), true))) continue;
                if (!(std::getline(ss, token, ',') && (rows = std::stoul(token), true))) continue;
                if (!(std::getline(ss, token, ',') && (cols = std::stoul(token), true))) continue;

                LayerInfo li;
                li.idx = idx;
                li.rows = rows;
                li.cols = cols;
                li.W.resize(rows, cols);

                for (size_t r = 0; r < rows; ++r) {
                    if (!std::getline(ifs, line)) {
                        std::cerr << "[MLP::loadWeightsCsv] Unexpected EOF at layer " << idx << "\n";
                        return false;
                    }
                    std::stringstream ls(line);
                    std::string val;
                    for (size_t c = 0; c < cols; ++c) {
                        if (!std::getline(ls, val, ',')) {
                            std::cerr << "[MLP::loadWeightsCsv] Not enough columns at layer "
                                      << idx << " row " << r << "\n";
                            return false;
                        }
                        li.W(r, c) = std::stod(val);
                    }
                }
                infos.push_back(std::move(li));
            }
            continue;
        }
    }

    if (infos.empty()) {
        std::cerr << "[MLP::loadWeightsCsv] No layer info found in file\n";
        return false;
    }

    // Determine activation function
    std::string loadAct = "LINEAR"; // fallback
    if (!confPath.empty()) {
        auto kv = parseIniToMap(confPath);
        std::string sact;
        std::string key = "activation";
        for (char &c : key)
            c = static_cast<char>(std::tolower((unsigned char)c));

        auto it = kv.find(key);
        if (it != kv.end()) sact = it->second;
        if (!sact.empty()) loadAct = trimStr(sact);
    } else {
        std::cerr << "[MLP::loadWeightsCsv] confPath is empty, using default activation LINEAR\n";
    }

    // Build layers from weights
    layers_.clear();
    layers_.resize(infos.size());
    nNeurons.clear();

    for (const auto &li : infos) {
        if (li.idx >= layers_.size()) {
            std::cerr << "[MLP::loadWeightsCsv] Layer index " << li.idx
                      << " out of range (have " << layers_.size() << ")\n";
            return false;
        }

        layers_[li.idx].initLayer(
            li.cols,
            li.rows,
            weight_init_type::RANDOM,
            strToActivation(loadAct),
            0
        );
        layers_[li.idx].setWeights(li.W);
        nNeurons.push_back(li.rows);
    }

    numLayers = nNeurons.size();
    return true;
}

/**
 * Load weights in compact binary
 */
bool MLP::loadWeightsBinary(const std::string &wPath, const std::string &confPath) {
    try {
        std::ifstream ifs(wPath, std::ios::binary);
        if (!ifs.is_open()) {
            std::cerr << "[MLP::loadWeightsBinary] Cannot open: " << wPath << "\n";
            return false;
        }
        if (confPath.empty()) {
            std::cerr << "[MLP::loadWeightsCsv] confPath is empty\n";
            return false;
        }

        size_t L = 0;
        ifs.read(reinterpret_cast<char*>(&L), sizeof(L));
        if (!ifs) {
            std::cerr << "[MLP::loadWeightsBinary] Failed to read layer count\n";
            return false;
        }

        layers_.clear();
        layers_.resize(L);

        auto kv = parseIniToMap(confPath);
        std::string sact, loadAct;
        {
            std::string key = "activation";
            for (char &c : key)
                c = static_cast<char>(std::tolower((unsigned char)c));

            auto it = kv.find(key);
            if (it != kv.end())
                sact = it->second;
        }
        if (!sact.empty()) loadAct = trimStr(sact);

        for (size_t i = 0; i < L; ++i) {
            uint64_t rows = 0, cols = 0;
            ifs.read(reinterpret_cast<char*>(&rows), sizeof(rows));
            ifs.read(reinterpret_cast<char*>(&cols), sizeof(cols));
            if (!ifs) {
                std::cerr << "[MLP::loadWeightsBinary] Failed to read dims for layer " << i << "\n";
                return false;
            }

            Eigen::MatrixXd W(rows, cols);
            for (uint64_t r = 0; r < rows; ++r) {
                for (uint64_t c = 0; c < cols; ++c) {
                    double v = 0.0;
                    ifs.read(reinterpret_cast<char*>(&v), sizeof(v));
                    if (!ifs) {
                        std::cerr << "[MLP::loadWeightsBinary] Failed to read weight at "
                                  << "layer " << i << " pos (" << r << "," << c << ")\n";
                        return false;
                    }
                    W(r, c) = v;
                }
            }

            layers_[i].initLayer(
                /*numInputs=*/ static_cast<size_t>(cols),
                /*numNeurons=*/ static_cast<size_t>(rows),
                /*wInitType=*/ weight_init_type::RANDOM,
                /*func=*/ strToActivation(loadAct), 
                /*seed=*/ 0
            );
            layers_[i].setWeights(W);
        }

        return true;
    } catch (const std::exception &ex) {
        std::cerr << "[MLP::loadWeightsBinary] Exception: " << ex.what() << "\n";
        return false;
    }
}

/**
 * Load weights vector from CSV text
 */
bool MLP::loadWeightsVectorCsv(const std::string &wPath, const std::string &confPath) {
    if (wPath.empty()) {
        std::cerr << "[MLP::loadWeightsVectorCsv] wPath is empty\n";
        return false;
    }
    if (confPath.empty()) {
        std::cerr << "[MLP::loadWeightsVectorCsv] confPath is empty\n";
        return false;
    }

    std::ifstream ifs(wPath);
    if (!ifs.is_open()) {
        int e = errno;
        std::cerr << "[MLP::loadWeightsVectorCsv] Cannot open file: " << wPath
                  << " errno=" << e << " (" << std::strerror(e) << ")\n";
        return false;
    }

    auto kv = parseIniToMap(confPath);
    auto get = [&](const std::string &k)->std::string {
        std::string kl = k;
        for (char &c : kl) c = static_cast<char>(std::tolower((unsigned char)c));
        auto it = kv.find(kl);
        if (it == kv.end()) return std::string{};
        return it->second;
    };

    std::vector<unsigned> loadArch;
    std::string loadAct;
    std::string sarch = get("architecture");
    if (!sarch.empty()) loadArch = parseUnsignedList(sarch);

    std::string sact = get("activation");
    if (!sact.empty()) loadAct = trimStr(sact);

    nNeurons = loadArch;
    numLayers = nNeurons.size();
    layers_.clear();
    layers_.resize(numLayers);

    std::vector<double> values;
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty()) continue;
        values.push_back(std::stod(line));
    }
    ifs.close();

    weightsVectorMlp.resize(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        weightsVectorMlp(i) = values[i];
    }

    int pos = weightsVectorMlp.size();

    for (int i = numLayers - 1; i > 0; --i) {
        int rows = nNeurons[i];
        int cols = nNeurons[i - 1] + 1;
        int size = rows * cols;

        pos -= size;
        if (pos < 0) throw std::runtime_error("Vector too short for given architecture");

        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            W(weightsVectorMlp.data() + pos, rows, cols);

        layers_[i].initLayer(
            /*numInputs=*/ static_cast<size_t>(cols),
            /*numNeurons=*/ static_cast<size_t>(rows),
            /*wInitType=*/ weight_init_type::RANDOM,
            /*func=*/ strToActivation(loadAct), 
            /*seed=*/ 0
        );
        layers_[i].setWeights(W);
    }

    int remaining = pos;
    if (remaining % nNeurons[0] != 0)
        throw std::runtime_error("Cannot deduce input size from vector");

    int cols0 = remaining / nNeurons[0];
    int rows0 = nNeurons[0];

    Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        W0(weightsVectorMlp.data(), rows0, cols0);

    layers_[0].initLayer(
        /*numInputs=*/ static_cast<size_t>(cols0),
        /*numNeurons=*/ static_cast<size_t>(rows0),
        /*wInitType=*/ weight_init_type::RANDOM,
        /*func=*/ strToActivation(loadAct),
        /*seed=*/ 0
    );
    layers_[0].setWeights(W0);

    return true;
}

/**
 * Getter for output
 */
Eigen::VectorXd& MLP::getOutput(){
    return output;
}

/**
 * Validate the size of the inputs compared to nNeurons[0]
 */
bool MLP::validateInputSize() {
    // There has to be an input layer
    if (nNeurons.empty()) {
        std::cerr << "[MLP] No input layer defined!\n";
        return false;
    }

    // Inps must have at least the bias slot
    if (Inps.size() < 1) {
        std::cerr << "[MLP] Inps is empty!\n";
        return false;
    }

    // realInputs = total slots minus the bias slot
    auto realInputs = Inps.size() - 1;

    // Compare to nNeurons[0]
    if (realInputs != nNeurons[0]) {
        std::cerr << "[MLP] Mismatch: nNeurons[0] = "
                  << nNeurons[0] << " but got " << realInputs
                  << " real inputs (Inps.size() = " << Inps.size() << ")\n";
        return false;
    }

    return true;
}

/**
 * Forward pass through all layers
 */
Eigen::VectorXd MLP::initMLP(const Eigen::VectorXd& input, int rngSeed) {
    if (nNeurons.empty() || activFuncs.size() != nNeurons.size() || wInitTypes.size() != nNeurons.size())
        throw std::logic_error("MLP not fully configured (architecture/activ/weight init mismatch)");

    layers_.clear();
    layers_.reserve(nNeurons.size());

    // Layer[0] from real inputs
    layers_.emplace_back();
    layers_[0].initLayer(
        /*numInputs=*/ input.size(),
        /*numNeurons=*/ nNeurons[0],
        /*wInitType=*/ wInitTypes[0],
        /*func=*/ activFuncs[0],
        /*seed=*/ rngSeed
    );
    layers_[0].setInputs(input);
    layers_[0].calculateLayerOutput(activFuncs[0]);
    Eigen::VectorXd currentOutput = layers_[0].getOutput();

    // Remaining layers in a for loop
    for (size_t i = 1; i < nNeurons.size(); ++i) {
        layers_.emplace_back();
        layers_[i].initLayer(
            /*numInputs=*/ currentOutput.size(),
            /*numNeurons=*/ nNeurons[i],
            /*wInitType=*/ wInitTypes[i],
            /*func=*/ activFuncs[i],
        /*seed=*/ rngSeed
        );
        layers_[i].setInputs(currentOutput);
        layers_[i].calculateLayerOutput(activFuncs[i]);
        currentOutput = layers_[i].getOutput();
    }
   
    return currentOutput;
}

/**
 * Forward pass reusing existing weights
 */
Eigen::VectorXd MLP::runMLP(const Eigen::VectorXd& input) {
    if (layers_.empty())
        throw std::logic_error("runMLP called before initMLP");

    // First layer
    layers_[0].setInputs(input);
    layers_[0].calculateLayerOutput(activFuncs[0]);
    Eigen::VectorXd currentOutput = layers_[0].getOutput();

    // Remaining layers
    for (size_t i = 1; i < layers_.size(); ++i) {
        layers_[i].setInputs(currentOutput);
        layers_[i].calculateLayerOutput(activFuncs[i]);
        currentOutput = layers_[i].getOutput();
    }
    return currentOutput;
}

/**
 * Compare if 'initMLP' and 'runMLP' produce the same output
 */
bool MLP::compareInitAndRun(const Eigen::VectorXd& input, double tol, int rngSeed) const {
    // Make a local copy of *this* so we can init on the copy
    MLP tmp = *this;
    Eigen::VectorXd outInit = tmp.initMLP(input, rngSeed);
    Eigen::VectorXd outRun = tmp.runMLP(input);

    return outInit.isApprox(outRun, tol);
}

/**
 * Test that 'runMLP' produces the same results over times
 */
bool MLP::testRepeatable(const Eigen::VectorXd& input, int repeats, double tol, int rngSeed) const {
    // Initialize once to get the baseline output
    MLP tmp = *this;
    Eigen::VectorXd baseline = tmp.initMLP(input, rngSeed);

    // Repeat run several times and compare
    for (int i = 0; i < repeats; ++i) {
        Eigen::VectorXd out = tmp.runMLP(input);
        if (!out.isApprox(baseline, tol)) {
            return false;
        }
    }
    return true;
}

/**
 * Forward pass and update weights with backpropagation (one input)
 */
void MLP::runAndBP(const Eigen::VectorXd& input, const Eigen::VectorXd& obsOut, double learningRate) {
    if (layers_.empty())
        throw std::logic_error("runMLP called before initMLP");

    calcOneOutput(input);

    // Output layer BP
    layers_[layers_.size()-1].setDeltas(layers_[layers_.size()-1].getOutput() - obsOut);
    layers_[layers_.size()-1].calculateOnlineGradient();

    // Remaining layers BP
    if(layers_.size() > 1){
        for(int i = layers_.size() - 2; i >= 0; --i){
            layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
            layers_[i].calculateOnlineGradient();
        }
    }
    for(size_t i = 0; i < getNumLayers(); i++){
        layers_[i].updateWeights(learningRate);
    }
}

/**
 * Forward pass and update weights with Adam algorithm (one input)
 */
void MLP::runAndBPadam(const Eigen::VectorXd& input, const Eigen::VectorXd& obsOut, double learningRate, int iterationNum) {
    if (layers_.empty())
        throw std::logic_error("runMLP called before initMLP");

    calcOneOutput(input);

    // Output layer BP
    layers_[layers_.size()-1].setDeltas(layers_[layers_.size()-1].getOutput() - obsOut);
    layers_[layers_.size()-1].calculateOnlineGradient();

    // Remaining layers BP
    if(layers_.size() > 1){
        for(int i = layers_.size() - 2; i >= 0; --i){
            layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
            layers_[i].calculateOnlineGradient();
        }
    }
    for(size_t i = 0; i < getNumLayers(); i++){
        layers_[i].updateAdam(learningRate,iterationNum,0.9, 0.99, 1e-8);
    }
}

/**
 * Online backpropagation - separete inp out matrices
 */
void MLP::onlineBP(int maxIterations, double maxError, double learningRate, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    if (layers_.empty())
        throw std::logic_error("onlineBP called before initMLP");

    if (X.rows() == 0 || Y.rows() == 0)
        throw std::invalid_argument("Empty training data passed to onlineBP");

    if (X.rows() != Y.rows())
        throw std::invalid_argument("matrices have different number of rows");

    if (maxIterations <= 0 || maxError < 0.0)
        throw std::invalid_argument("maxIterations and maxError must be positive");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    int numOfPatterns = X.rows();                // number of patterns in calibration matrix
    int inpSize = layers_[0].getWeights().cols()-1;   // number of inputs to first layer (without bias)
    int outSize = nNeurons.back();                   // number of output neurons
    int lastLayerIndex = getNumLayers()-1;

    if (inpSize != X.cols())
        throw std::runtime_error("Input matrix row length doesnt match the initialized input size");

    if (outSize != Y.cols())
        throw std::runtime_error("Output matrix row length doesnt match the initialized output size");

    double Error;
    auto start = high_resolution_clock::now();
    for (int iter = 1; iter < maxIterations + 1; iter++){
        Error = 0.0;
        for (int pat = 0; pat < numOfPatterns; pat++){
            Eigen::VectorXd currentInp = X.row(pat);
            Eigen::VectorXd currentObs = Y.row(pat);
            
            calcOneOutput(currentInp);

            // Output layer BP
            layers_[lastLayerIndex].setDeltas(layers_[lastLayerIndex].getOutput() - currentObs);
            layers_[lastLayerIndex].calculateOnlineGradient();

            // Remaining layers BP
            if(layers_.size() > 1){
                for(int i = lastLayerIndex - 1; i >= 0; --i){
                    layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
                    layers_[i].calculateOnlineGradient();
                }
            }
            Error += layers_[lastLayerIndex].getDeltas().squaredNorm();
            for(size_t i = 0; i < getNumLayers(); i++){
                layers_[i].updateWeights(learningRate);
            }
        }
        Error = Error / numOfPatterns;
        if(Error <= maxError){
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(stop - start);

            // Store results
            result.iterations = iter;
            result.finalLoss = Error;
            result.time = duration.count();
            result.converged = true;

            break;
        }
    }
    if(Error > maxError){
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    // Store results
    result.iterations = maxIterations;
    result.finalLoss = Error;
    result.time = duration.count();
    result.converged = false;
    }
}

/**
 * Online backpropagation using Adam algorithm - separete inp out matrices
 */
void MLP::onlineAdam(int maxIterations, double maxError, double learningRate, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    if (layers_.empty())
        throw std::logic_error("onlineAdam called before initMLP");

    if (X.rows() == 0 || Y.rows() == 0)
        throw std::invalid_argument("Empty training data passed to onlineAdam");
    
    if (X.rows() != Y.rows())
        throw std::invalid_argument("matrices have different number of rows");
    
    if (maxIterations <= 0 || maxError < 0.0)
        throw std::invalid_argument("maxIterations and maxError must be positive");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    int numOfPatterns = X.rows();                  // number of patterns in calibration matrix
    int inpSize = layers_[0].getWeights().cols()-1;   // number of inputs to first layer (without bias)
    int outSize = nNeurons.back();                   // number of output neurons
    int lastLayerIndex = getNumLayers()-1;

    if (inpSize != X.cols())
        throw std::runtime_error("Input matrix row length doesnt match the initialized input size");

    if (outSize != Y.cols())
        throw std::runtime_error("Output matrix row length doesnt match the initialized output size");

    double Error;
    auto start = high_resolution_clock::now();
    for (int iter = 1; iter < maxIterations + 1; iter++){
        Error = 0.0;
        for (int pat = 0; pat < numOfPatterns; pat++){
            Eigen::VectorXd currentInp = X.row(pat);
            Eigen::VectorXd currentObs = Y.row(pat);

            calcOneOutput(currentInp);

            // Output layer BP
            layers_[lastLayerIndex].setDeltas(layers_[lastLayerIndex].getOutput() - currentObs);
            layers_[lastLayerIndex].calculateOnlineGradient();

            // Remaining layers BP
            if(layers_.size() > 1){
                for(int i = lastLayerIndex - 1; i >= 0; --i){
                    layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
                    layers_[i].calculateOnlineGradient();
                }
            }
            Error += layers_[lastLayerIndex].getDeltas().squaredNorm();
            for(size_t i = 0; i < getNumLayers(); i++){
                layers_[i].updateAdam(learningRate,iter,0.9, 0.99, 1e-8);
            }
        }
        Error = Error / numOfPatterns;
        if(Error <= maxError){
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(stop - start);

            // Store results
            result.iterations = iter;
            result.finalLoss = Error;
            result.time = duration.count();
            result.converged = true;

            break;
        }
    }
    if(Error > maxError){
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    // Store results
    result.iterations = maxIterations;
    result.finalLoss = Error;
    result.time = duration.count();
    result.converged = false;
    }
}

/**
 * Batch backpropagation using Adam algorithm  - separete inp out matrices
 */
void MLP::batchAdam(int maxIterations, double maxError, int batchSize, double learningRate, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    if (layers_.empty())
        throw std::logic_error("batchAdam called before initMLP");

    if (X.rows() == 0 || Y.rows() == 0)
        throw std::invalid_argument("Empty training data passed to batchAdam");
            
    if (X.rows() != Y.rows())
        throw std::invalid_argument("matrices have different number of rows");

    if (maxIterations <= 0 || batchSize <= 0|| maxError < 0.0)
        throw std::invalid_argument("maxIterations, batchSize and maxError must be positive");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    int numOfPatterns = X.rows();                  // number of patterns in calibration matrix
    int inpSize = layers_[0].getWeights().cols()-1;   // number of inputs to first layer (without bias)
    int outSize = nNeurons.back();                   // number of output neurons
    int lastLayerIndex = getNumLayers()-1;

    if (inpSize != X.cols())
        throw std::runtime_error("Input matrix row length doesnt match the initialized input size");

    if (outSize != Y.cols())
        throw std::runtime_error("Output matrix row length doesnt match the initialized output size");

    double Error;
    auto start = high_resolution_clock::now();
    for (int iter = 1; iter < maxIterations + 1; iter++){
        Error = 0.0;
        for(int batch = 0; batch < (numOfPatterns + batchSize - 1)/batchSize; batch++){
            int start = batch * batchSize;
            int end   = std::min(start + batchSize,numOfPatterns);
            for (int pat = start; pat < end; pat++){

                Eigen::VectorXd currentInp = X.row(pat);
                Eigen::VectorXd currentObs = Y.row(pat);
                
                calcOneOutput(currentInp);

                // Output layer gradient
                layers_[lastLayerIndex].setDeltas(layers_[lastLayerIndex].getOutput() - currentObs);
                layers_[lastLayerIndex].calculateBatchGradient();

                // Remaining layers BP
                if(layers_.size() > 1){
                    for(int i = lastLayerIndex - 1; i >= 0; --i){
                        layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
                        layers_[i].calculateBatchGradient();
                    }
                }
                Error += layers_[lastLayerIndex].getDeltas().squaredNorm();
                }
                for (int i = 0; i <= lastLayerIndex; i++){
                    layers_[i].updateAdam(learningRate,iter,0.9, 0.99, 1e-8);
                    layers_[i].setGradient(layers_[i].getGradient().setZero());
                }
            }
            Error = Error / numOfPatterns;
        if(Error <= maxError){
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(stop - start);

            // Store results
            result.iterations = iter;
            result.finalLoss = Error;
            result.time = duration.count();
            result.converged = true;

            break;
        }
    }
    if(Error > maxError){
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    // Store results
    result.iterations = maxIterations;
    result.finalLoss = Error;
    result.time = duration.count();
    result.converged = false;
    }
}

/**
 * Batch backpropagation - separete inp out matrices
 */
void MLP::batchBP(int maxIterations, double maxError, int batchSize, double learningRate, const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) {
    if (layers_.empty())
        throw std::logic_error("batchBP called before initMLP");

    if (X.rows() == 0 || Y.rows() == 0)
        throw std::invalid_argument("Empty training data passed to batchBP");
            
    if (X.rows() != Y.rows())
        throw std::invalid_argument("matrices have different number of rows");

    if (maxIterations <= 0 || batchSize <= 0|| maxError < 0.0)
        throw std::invalid_argument("maxIterations, batchSize and maxError must be positive");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    int numOfPatterns = X.rows();                  // number of patterns in calibration matrix
    int inpSize = layers_[0].getWeights().cols()-1;   // number of inputs to first layer (without bias)
    int outSize = nNeurons.back();                   // number of output neurons
    int lastLayerIndex = getNumLayers()-1;

    if (inpSize != X.cols())
        throw std::runtime_error("Input matrix row length doesnt match the initialized input size");

    if (outSize != Y.cols())
        throw std::runtime_error("Output matrix row length doesnt match the initialized output size");

    double Error;
    auto start = high_resolution_clock::now();
    for (int iter = 1; iter < maxIterations + 1; iter++){
        Error = 0.0;
        for(int batch = 0; batch < (numOfPatterns + batchSize - 1)/batchSize; batch++){
            int start = batch * batchSize;
            int end   = std::min(start + batchSize,numOfPatterns);
            for (int pat = start; pat < end; pat++){

                Eigen::VectorXd currentInp = X.row(pat);
                Eigen::VectorXd currentObs = Y.row(pat);
                
                calcOneOutput(currentInp);

                // Output layer gradient
                layers_[lastLayerIndex].setDeltas(layers_[lastLayerIndex].getOutput() - currentObs);
                layers_[lastLayerIndex].calculateBatchGradient();

                // Remaining layers BP
                if(layers_.size() > 1){
                    for(int i = lastLayerIndex - 1; i >= 0; --i){
                        layers_[i].calculateDeltas(layers_[i+1].getWeights(),layers_[i+1].getDeltas(),activFuncs[i]);
                        layers_[i].calculateBatchGradient();
                    }
                }
                Error += layers_[lastLayerIndex].getDeltas().squaredNorm();
                }
                for (int i = 0; i <= lastLayerIndex; i++){
                    layers_[i].updateWeights(learningRate);
                    layers_[i].setGradient(layers_[i].getGradient().setZero());
                }
            }
            Error = Error / numOfPatterns;
        if(Error <= maxError){
            auto stop = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(stop - start);

            // Store results
            result.iterations = iter;
            result.finalLoss = Error;
            result.time = duration.count();
            result.converged = true;

            break;
        }
    }
    if(Error > maxError){
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    // Store results
    result.iterations = maxIterations;
    result.finalLoss = Error;
    result.time = duration.count();
    result.converged = false;
    }
}

std::vector<Eigen::MatrixXd> MLP::onlineAdamEpochVal(
    const Eigen::MatrixXd &Xtrain,
    const Eigen::MatrixXd &Ytrain,
    const Eigen::MatrixXd &Xval,
    const Eigen::MatrixXd &Yval,
    int maxIterations,
    double learningRate,
    int metricsAfterXEpochs)
{
    if (layers_.empty())
        throw std::logic_error("onlineAdamEpochVal called before initMLP");

    if (Xtrain.rows() == 0 || Ytrain.rows() == 0 || Xval.rows() == 0 || Yval.rows() == 0)
        throw std::invalid_argument("Empty training data passed to onlineAdamEpochVal");
            
    if (Xtrain.rows() != Ytrain.rows())
        throw std::invalid_argument("train matrices have different number of rows");

    if (Xval.rows() != Yval.rows())
        throw std::invalid_argument("validation matrices have different number of rows");

    if (maxIterations <= 0 || metricsAfterXEpochs <= 0)
        throw std::invalid_argument("maxIterations and metricsAfterXEpochs must be positive");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    if (metricsAfterXEpochs > maxIterations)
        throw std::invalid_argument("metricsAfterXEpochs can't be bigger than maxIterations");

    std::vector<Eigen::MatrixXd> resultMetrics;
    int rowControl;
    if(metricsAfterXEpochs == 1){
        Eigen::MatrixXd prepMat = Eigen::MatrixXd(maxIterations,Ytrain.cols());   
        for(int i = 0; i < 14; i++){
            resultMetrics.push_back(prepMat);
        }
    } else {
        rowControl = maxIterations / metricsAfterXEpochs;
        if(maxIterations % metricsAfterXEpochs == 0){
            Eigen::MatrixXd prepMat = Eigen::MatrixXd(rowControl, Ytrain.cols());   
            for(int i = 0; i < 14; i++){
                resultMetrics.push_back(prepMat);
            }
        } else {
            Eigen::MatrixXd prepMat = Eigen::MatrixXd(rowControl + 1, Ytrain.cols());   
            for(int i = 0; i < 14; i++){
                resultMetrics.push_back(prepMat);
            }
        }
    }

    auto start = high_resolution_clock::now();
    for(int epoch = 0; epoch < maxIterations; epoch++){
        // Run online Adam
        try {
            onlineAdam(1, 0.0, learningRate, Xtrain, Ytrain);
        } catch (const std::exception &ex) {
            std::cerr << "[onlineAdam] training failed: " << ex.what() << "\n";
            throw;
        }

        if(metricsAfterXEpochs == 1){
            calculateOutputs(Xtrain);
            for (int c = 0; c < Ytrain.cols(); ++c) {
                resultMetrics[0](epoch,c) = Metrics::mse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[1](epoch,c) = Metrics::rmse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[2](epoch,c) = Metrics::pi(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[3](epoch,c) = Metrics::ns(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[4](epoch,c) = Metrics::kge(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[5](epoch,c) = Metrics::pbias(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[6](epoch,c) = Metrics::rsr(Ytrain.col(c).eval(), outputMat.col(c).eval());
            }

            calculateOutputs(Xval);
            for (int c = 0; c < Ytrain.cols(); ++c) {
                resultMetrics[7](epoch,c) = Metrics::mse(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[8](epoch,c) = Metrics::rmse(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[9](epoch,c) = Metrics::pi(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[10](epoch,c) = Metrics::ns(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[11](epoch,c) = Metrics::kge(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[12](epoch,c) = Metrics::pbias(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[13](epoch,c) = Metrics::rsr(Yval.col(c).eval(), outputMat.col(c).eval());
            }
        } else {
            if(epoch % metricsAfterXEpochs == 0 && epoch != 0){
                int row = epoch / metricsAfterXEpochs - 1;
                calculateOutputs(Xtrain);
                for (int c = 0; c < Ytrain.cols(); ++c) {
                    resultMetrics[0](row,c) = Metrics::mse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[1](row,c) = Metrics::rmse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[2](row,c) = Metrics::pi(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[3](row,c) = Metrics::ns(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[4](row,c) = Metrics::kge(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[5](row,c) = Metrics::pbias(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[6](row,c) = Metrics::rsr(Ytrain.col(c).eval(), outputMat.col(c).eval());
                }

                calculateOutputs(Xval);
                for (int c = 0; c < Ytrain.cols(); ++c) {
                    resultMetrics[7](row,c) = Metrics::mse(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[8](row,c) = Metrics::rmse(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[9](row,c) = Metrics::pi(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[10](row,c) = Metrics::ns(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[11](row,c) = Metrics::kge(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[12](row,c) = Metrics::pbias(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[13](row,c) = Metrics::rsr(Yval.col(c).eval(), outputMat.col(c).eval());
                }
            }
            if(epoch == maxIterations - 1){
                int row;
                if(maxIterations % metricsAfterXEpochs == 0){
                    row = rowControl - 1;
                } else {
                    row = rowControl;
                }
                calculateOutputs(Xtrain);
                for (int c = 0; c < Ytrain.cols(); ++c) {
                    resultMetrics[0](row,c) = Metrics::mse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[1](row,c) = Metrics::rmse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[2](row,c) = Metrics::pi(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[3](row,c) = Metrics::ns(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[4](row,c) = Metrics::kge(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[5](row,c) = Metrics::pbias(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[6](row,c) = Metrics::rsr(Ytrain.col(c).eval(), outputMat.col(c).eval());
                }

                calculateOutputs(Xval);
                for (int c = 0; c < Ytrain.cols(); ++c) {
                    resultMetrics[7](row,c) = Metrics::mse(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[8](row,c) = Metrics::rmse(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[9](row,c) = Metrics::pi(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[10](row,c) = Metrics::ns(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[11](row,c) = Metrics::kge(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[12](row,c) = Metrics::pbias(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[13](row,c) = Metrics::rsr(Yval.col(c).eval(), outputMat.col(c).eval());
                }
            }
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);
    
    calculateOutputs(Xtrain);

    result.converged = false;
    result.finalLoss = Metrics::mse(Ytrain,outputMat);
    result.iterations = maxIterations;
    result.time = duration.count();

    return resultMetrics;
}

std::vector<Eigen::MatrixXd> MLP::onlineBpEpochVal(
    const Eigen::MatrixXd &Xtrain,
    const Eigen::MatrixXd &Ytrain,
    const Eigen::MatrixXd &Xval,
    const Eigen::MatrixXd &Yval,
    int maxIterations,
    double learningRate,
    int metricsAfterXEpochs)
{
    if (layers_.empty())
        throw std::logic_error("onlineBpEpochVal called before initMLP");

    if (Xtrain.rows() == 0 || Ytrain.rows() == 0 || Xval.rows() == 0 || Yval.rows() == 0)
        throw std::invalid_argument("Empty training data passed to onlineBpEpochVal");
            
    if (Xtrain.rows() != Ytrain.rows())
        throw std::invalid_argument("train matrices have different number of rows");

    if (Xval.rows() != Yval.rows())
        throw std::invalid_argument("validation matrices have different number of rows");

    if (maxIterations <= 0 || metricsAfterXEpochs <= 0)
        throw std::invalid_argument("maxIterations and metricsAfterXEpochs must be positive");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    if (metricsAfterXEpochs > maxIterations)
        throw std::invalid_argument("metricsAfterXEpochs can't be bigger than maxIterations");

    std::vector<Eigen::MatrixXd> resultMetrics;
    int rowControl;
    if(metricsAfterXEpochs == 1){
        Eigen::MatrixXd prepMat = Eigen::MatrixXd(maxIterations,Ytrain.cols());   
        for(int i = 0; i < 14; i++){
            resultMetrics.push_back(prepMat);
        }
    } else {
        rowControl = maxIterations / metricsAfterXEpochs;
        if(maxIterations % metricsAfterXEpochs == 0){
            Eigen::MatrixXd prepMat = Eigen::MatrixXd(rowControl, Ytrain.cols());   
            for(int i = 0; i < 14; i++){
                resultMetrics.push_back(prepMat);
            }
        } else {
            Eigen::MatrixXd prepMat = Eigen::MatrixXd(rowControl + 1, Ytrain.cols());   
            for(int i = 0; i < 14; i++){
                resultMetrics.push_back(prepMat);
            }
        }
    }

    auto start = high_resolution_clock::now();
    for(int epoch = 0; epoch < maxIterations; epoch++){
        // Run online BP
        try {
            onlineBP(1, 0.0, learningRate, Xtrain, Ytrain);
        } catch (const std::exception &ex) {
            std::cerr << "[onlineBP] training failed: " << ex.what() << "\n";
            throw;
        }

        if(metricsAfterXEpochs == 1){
            calculateOutputs(Xtrain);
            for (int c = 0; c < Ytrain.cols(); ++c) {
                resultMetrics[0](epoch,c) = Metrics::mse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[1](epoch,c) = Metrics::rmse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[2](epoch,c) = Metrics::pi(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[3](epoch,c) = Metrics::ns(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[4](epoch,c) = Metrics::kge(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[5](epoch,c) = Metrics::pbias(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[6](epoch,c) = Metrics::rsr(Ytrain.col(c).eval(), outputMat.col(c).eval());
            }

            calculateOutputs(Xval);
            for (int c = 0; c < Ytrain.cols(); ++c) {
                resultMetrics[7](epoch,c) = Metrics::mse(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[8](epoch,c) = Metrics::rmse(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[9](epoch,c) = Metrics::pi(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[10](epoch,c) = Metrics::ns(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[11](epoch,c) = Metrics::kge(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[12](epoch,c) = Metrics::pbias(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[13](epoch,c) = Metrics::rsr(Yval.col(c).eval(), outputMat.col(c).eval());
            }
        } else {
            if(epoch % metricsAfterXEpochs == 0 && epoch != 0){
                int row = epoch / metricsAfterXEpochs - 1;
                calculateOutputs(Xtrain);
                for (int c = 0; c < Ytrain.cols(); ++c) {
                    resultMetrics[0](row,c) = Metrics::mse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[1](row,c) = Metrics::rmse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[2](row,c) = Metrics::pi(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[3](row,c) = Metrics::ns(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[4](row,c) = Metrics::kge(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[5](row,c) = Metrics::pbias(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[6](row,c) = Metrics::rsr(Ytrain.col(c).eval(), outputMat.col(c).eval());
                }

                calculateOutputs(Xval);
                for (int c = 0; c < Ytrain.cols(); ++c) {
                    resultMetrics[7](row,c) = Metrics::mse(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[8](row,c) = Metrics::rmse(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[9](row,c) = Metrics::pi(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[10](row,c) = Metrics::ns(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[11](row,c) = Metrics::kge(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[12](row,c) = Metrics::pbias(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[13](row,c) = Metrics::rsr(Yval.col(c).eval(), outputMat.col(c).eval());
                }
            }
            if(epoch == maxIterations - 1){
                int row;
                if(maxIterations % metricsAfterXEpochs == 0){
                    row = rowControl - 1;
                } else {
                    row = rowControl;
                }
                calculateOutputs(Xtrain);
                for (int c = 0; c < Ytrain.cols(); ++c) {
                    resultMetrics[0](row,c) = Metrics::mse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[1](row,c) = Metrics::rmse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[2](row,c) = Metrics::pi(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[3](row,c) = Metrics::ns(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[4](row,c) = Metrics::kge(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[5](row,c) = Metrics::pbias(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[6](row,c) = Metrics::rsr(Ytrain.col(c).eval(), outputMat.col(c).eval());
                }

                calculateOutputs(Xval);
                for (int c = 0; c < Ytrain.cols(); ++c) {
                    resultMetrics[7](row,c) = Metrics::mse(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[8](row,c) = Metrics::rmse(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[9](row,c) = Metrics::pi(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[10](row,c) = Metrics::ns(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[11](row,c) = Metrics::kge(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[12](row,c) = Metrics::pbias(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[13](row,c) = Metrics::rsr(Yval.col(c).eval(), outputMat.col(c).eval());
                }
            }
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);
    
    calculateOutputs(Xtrain);

    result.converged = false;
    result.finalLoss = Metrics::mse(Ytrain,outputMat);
    result.iterations = maxIterations;
    result.time = duration.count();

    return resultMetrics;
}

std::vector<Eigen::MatrixXd> MLP::batchAdamEpochVal(
    const Eigen::MatrixXd &Xtrain,
    const Eigen::MatrixXd &Ytrain,
    const Eigen::MatrixXd &Xval,
    const Eigen::MatrixXd &Yval,
    int batchSize,
    int maxIterations,
    double learningRate,
    int metricsAfterXEpochs)
{
    if (layers_.empty())
        throw std::logic_error("batchAdamEpochVal called before initMLP");

    if (Xtrain.rows() == 0 || Ytrain.rows() == 0 || Xval.rows() == 0 || Yval.rows() == 0)
        throw std::invalid_argument("Empty training data passed to batchAdamEpochVal");
            
    if (Xtrain.rows() != Ytrain.rows())
        throw std::invalid_argument("train matrices have different number of rows");

    if (Xval.rows() != Yval.rows())
        throw std::invalid_argument("validation matrices have different number of rows");

    if (maxIterations <= 0 || batchSize <= 0 || metricsAfterXEpochs <= 0)
        throw std::invalid_argument("maxIterations, batchSize and metricsAfterXEpochs must be positive");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    if (metricsAfterXEpochs > maxIterations)
        throw std::invalid_argument("metricsAfterXEpochs can't be bigger than maxIterations");

    std::vector<Eigen::MatrixXd> resultMetrics;
    int rowControl;
    if(metricsAfterXEpochs == 1){
        Eigen::MatrixXd prepMat = Eigen::MatrixXd(maxIterations,Ytrain.cols());   
        for(int i = 0; i < 14; i++){
            resultMetrics.push_back(prepMat);
        }
    } else {
        rowControl = maxIterations / metricsAfterXEpochs;
        if(maxIterations % metricsAfterXEpochs == 0){
            Eigen::MatrixXd prepMat = Eigen::MatrixXd(rowControl, Ytrain.cols());   
            for(int i = 0; i < 14; i++){
                resultMetrics.push_back(prepMat);
            }
        } else {
            Eigen::MatrixXd prepMat = Eigen::MatrixXd(rowControl + 1, Ytrain.cols());   
            for(int i = 0; i < 14; i++){
                resultMetrics.push_back(prepMat);
            }
        }
    }

    auto start = high_resolution_clock::now();
    for(int epoch = 0; epoch < maxIterations; epoch++){
        // Run batch Adam
        try {
            batchAdam(1, 0.0, batchSize, learningRate, Xtrain, Ytrain);
        } catch (const std::exception &ex) {
            std::cerr << "[batchAdam] training failed: " << ex.what() << "\n";
            throw;
        }

        if(metricsAfterXEpochs == 1){
            calculateOutputs(Xtrain);
            for (int c = 0; c < Ytrain.cols(); ++c) {
                resultMetrics[0](epoch,c) = Metrics::mse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[1](epoch,c) = Metrics::rmse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[2](epoch,c) = Metrics::pi(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[3](epoch,c) = Metrics::ns(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[4](epoch,c) = Metrics::kge(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[5](epoch,c) = Metrics::pbias(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[6](epoch,c) = Metrics::rsr(Ytrain.col(c).eval(), outputMat.col(c).eval());
            }

            calculateOutputs(Xval);
            for (int c = 0; c < Ytrain.cols(); ++c) {
                resultMetrics[7](epoch,c) = Metrics::mse(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[8](epoch,c) = Metrics::rmse(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[9](epoch,c) = Metrics::pi(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[10](epoch,c) = Metrics::ns(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[11](epoch,c) = Metrics::kge(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[12](epoch,c) = Metrics::pbias(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[13](epoch,c) = Metrics::rsr(Yval.col(c).eval(), outputMat.col(c).eval());
            }
        } else {
            if(epoch % metricsAfterXEpochs == 0 && epoch != 0){
                int row = epoch / metricsAfterXEpochs - 1;
                calculateOutputs(Xtrain);
                for (int c = 0; c < Ytrain.cols(); ++c) {
                    resultMetrics[0](row,c) = Metrics::mse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[1](row,c) = Metrics::rmse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[2](row,c) = Metrics::pi(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[3](row,c) = Metrics::ns(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[4](row,c) = Metrics::kge(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[5](row,c) = Metrics::pbias(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[6](row,c) = Metrics::rsr(Ytrain.col(c).eval(), outputMat.col(c).eval());
                }

                calculateOutputs(Xval);
                for (int c = 0; c < Ytrain.cols(); ++c) {
                    resultMetrics[7](row,c) = Metrics::mse(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[8](row,c) = Metrics::rmse(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[9](row,c) = Metrics::pi(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[10](row,c) = Metrics::ns(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[11](row,c) = Metrics::kge(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[12](row,c) = Metrics::pbias(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[13](row,c) = Metrics::rsr(Yval.col(c).eval(), outputMat.col(c).eval());
                }
            }
            if(epoch == maxIterations - 1){
                int row;
                if(maxIterations % metricsAfterXEpochs == 0){
                    row = rowControl - 1;
                } else {
                    row = rowControl;
                }
                calculateOutputs(Xtrain);
                for (int c = 0; c < Ytrain.cols(); ++c) {
                    resultMetrics[0](row,c) = Metrics::mse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[1](row,c) = Metrics::rmse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[2](row,c) = Metrics::pi(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[3](row,c) = Metrics::ns(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[4](row,c) = Metrics::kge(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[5](row,c) = Metrics::pbias(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[6](row,c) = Metrics::rsr(Ytrain.col(c).eval(), outputMat.col(c).eval());
                }

                calculateOutputs(Xval);
                for (int c = 0; c < Ytrain.cols(); ++c) {
                    resultMetrics[7](row,c) = Metrics::mse(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[8](row,c) = Metrics::rmse(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[9](row,c) = Metrics::pi(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[10](row,c) = Metrics::ns(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[11](row,c) = Metrics::kge(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[12](row,c) = Metrics::pbias(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[13](row,c) = Metrics::rsr(Yval.col(c).eval(), outputMat.col(c).eval());
                }
            }
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    calculateOutputs(Xtrain);

    result.converged = false;
    result.finalLoss = Metrics::mse(Ytrain,outputMat);
    result.iterations = maxIterations;
    result.time = duration.count();

    return resultMetrics;
}

std::vector<Eigen::MatrixXd> MLP::batchBpEpochVal(
    const Eigen::MatrixXd &Xtrain,
    const Eigen::MatrixXd &Ytrain,
    const Eigen::MatrixXd &Xval,
    const Eigen::MatrixXd &Yval,
    int batchSize,
    int maxIterations,
    double learningRate,
    int metricsAfterXEpochs)
{
    if (layers_.empty())
        throw std::logic_error("batchBpEpochVal called before initMLP");

    if (Xtrain.rows() == 0 || Ytrain.rows() == 0 || Xval.rows() == 0 || Yval.rows() == 0)
        throw std::invalid_argument("Empty training data passed to batchBpEpochVal");
            
    if (Xtrain.rows() != Ytrain.rows())
        throw std::invalid_argument("train matrices have different number of rows");

    if (Xval.rows() != Yval.rows())
        throw std::invalid_argument("validation matrices have different number of rows");

    if (maxIterations <= 0 || batchSize <= 0 || metricsAfterXEpochs <= 0)
        throw std::invalid_argument("maxIterations, batchSize and metricsAfterXEpochs must be positive");
    
    if (learningRate <= 0.0 || learningRate > 1.0)
        throw std::invalid_argument("learningRate must be between 0 and 1");

    if (metricsAfterXEpochs > maxIterations)
        throw std::invalid_argument("metricsAfterXEpochs can't be bigger than maxIterations");

    std::vector<Eigen::MatrixXd> resultMetrics;
    int rowControl;
    if(metricsAfterXEpochs == 1){
        Eigen::MatrixXd prepMat = Eigen::MatrixXd(maxIterations,Ytrain.cols());   
        for(int i = 0; i < 14; i++){
            resultMetrics.push_back(prepMat);
        }
    } else {
        rowControl = maxIterations / metricsAfterXEpochs;
        if(maxIterations % metricsAfterXEpochs == 0){
            Eigen::MatrixXd prepMat = Eigen::MatrixXd(rowControl, Ytrain.cols());   
            for(int i = 0; i < 14; i++){
                resultMetrics.push_back(prepMat);
            }
        } else {
            Eigen::MatrixXd prepMat = Eigen::MatrixXd(rowControl + 1, Ytrain.cols());   
            for(int i = 0; i < 14; i++){
                resultMetrics.push_back(prepMat);
            }
        }
    }

    auto start = high_resolution_clock::now();
    for(int epoch = 0; epoch < maxIterations; epoch++){
        // Run batch Adam
        try {
            batchBP(1, 0.0, batchSize, learningRate, Xtrain, Ytrain);
        } catch (const std::exception &ex) {
            std::cerr << "[batchBP] training failed: " << ex.what() << "\n";
            throw;
        }

        if(metricsAfterXEpochs == 1){
            calculateOutputs(Xtrain);
            for (int c = 0; c < Ytrain.cols(); ++c) {
                resultMetrics[0](epoch,c) = Metrics::mse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[1](epoch,c) = Metrics::rmse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[2](epoch,c) = Metrics::pi(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[3](epoch,c) = Metrics::ns(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[4](epoch,c) = Metrics::kge(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[5](epoch,c) = Metrics::pbias(Ytrain.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[6](epoch,c) = Metrics::rsr(Ytrain.col(c).eval(), outputMat.col(c).eval());
            }

            calculateOutputs(Xval);
            for (int c = 0; c < Ytrain.cols(); ++c) {
                resultMetrics[7](epoch,c) = Metrics::mse(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[8](epoch,c) = Metrics::rmse(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[9](epoch,c) = Metrics::pi(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[10](epoch,c) = Metrics::ns(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[11](epoch,c) = Metrics::kge(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[12](epoch,c) = Metrics::pbias(Yval.col(c).eval(), outputMat.col(c).eval());
                resultMetrics[13](epoch,c) = Metrics::rsr(Yval.col(c).eval(), outputMat.col(c).eval());
            }
        } else {
            if(epoch % metricsAfterXEpochs == 0 && epoch != 0){
                int row = epoch / metricsAfterXEpochs - 1;
                calculateOutputs(Xtrain);
                for (int c = 0; c < Ytrain.cols(); ++c) {
                    resultMetrics[0](row,c) = Metrics::mse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[1](row,c) = Metrics::rmse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[2](row,c) = Metrics::pi(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[3](row,c) = Metrics::ns(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[4](row,c) = Metrics::kge(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[5](row,c) = Metrics::pbias(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[6](row,c) = Metrics::rsr(Ytrain.col(c).eval(), outputMat.col(c).eval());
                }

                calculateOutputs(Xval);
                for (int c = 0; c < Ytrain.cols(); ++c) {
                    resultMetrics[7](row,c) = Metrics::mse(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[8](row,c) = Metrics::rmse(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[9](row,c) = Metrics::pi(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[10](row,c) = Metrics::ns(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[11](row,c) = Metrics::kge(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[12](row,c) = Metrics::pbias(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[13](row,c) = Metrics::rsr(Yval.col(c).eval(), outputMat.col(c).eval());
                }
            }
            if(epoch == maxIterations - 1){
                int row;
                if(maxIterations % metricsAfterXEpochs == 0){
                    row = rowControl - 1;
                } else {
                    row = rowControl;
                }
                calculateOutputs(Xtrain);
                for (int c = 0; c < Ytrain.cols(); ++c) {
                    resultMetrics[0](row,c) = Metrics::mse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[1](row,c) = Metrics::rmse(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[2](row,c) = Metrics::pi(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[3](row,c) = Metrics::ns(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[4](row,c) = Metrics::kge(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[5](row,c) = Metrics::pbias(Ytrain.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[6](row,c) = Metrics::rsr(Ytrain.col(c).eval(), outputMat.col(c).eval());
                }

                calculateOutputs(Xval);
                for (int c = 0; c < Ytrain.cols(); ++c) {
                    resultMetrics[7](row,c) = Metrics::mse(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[8](row,c) = Metrics::rmse(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[9](row,c) = Metrics::pi(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[10](row,c) = Metrics::ns(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[11](row,c) = Metrics::kge(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[12](row,c) = Metrics::pbias(Yval.col(c).eval(), outputMat.col(c).eval());
                    resultMetrics[13](row,c) = Metrics::rsr(Yval.col(c).eval(), outputMat.col(c).eval());
                }
            }
        }
    }
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(stop - start);

    calculateOutputs(Xtrain);

    result.converged = false;
    result.finalLoss = Metrics::mse(Ytrain,outputMat);
    result.iterations = maxIterations;
    result.time = duration.count();

    return resultMetrics;
}

/**
 * Forward pass reusing existing weights
 */
void MLP::calcOneOutput(const Eigen::VectorXd& inputVec){      
    // First layer
    layers_[0].setInputs(inputVec);
    layers_[0].calculateOutput(activFuncs[0]);

    // Remaining layers
    for (size_t i = 1; i < getNumLayers(); ++i) {
        layers_[i].setInputs(layers_[i-1].getOutput());
        layers_[i].calculateOutput(activFuncs[i]);    
    }
    output = layers_[getNumLayers()-1].getOutput();
}

/**
 * Calculate outputs for given matrix of inputs
 */
void MLP::calculateOutputs(const Eigen::MatrixXd& inputMat){
    if (layers_.empty())
        throw std::logic_error("calculateOutputs called before initMLP");

    int inpSize = layers_[0].getWeights().cols()-1;
    if (inpSize != inputMat.cols())
        throw std::invalid_argument("Input matrix row length doesnt match the initialized input size");

    if (inputMat.rows() <= 0)
        throw std::invalid_argument("Input matrix is empty");

    outputMat = Eigen::MatrixXd(inputMat.rows(),nNeurons.back());
    for(int i = 0; i < inputMat.rows(); i++){
        calcOneOutput(inputMat.row(i));
        outputMat.row(i) = output;
    }
}

Eigen::MatrixXd MLP::getOutputs() const{
    return outputMat;
}

Eigen::VectorXd MLP::getFirstLayerDeltaSum(){
    Eigen::VectorXd delt(1);
    delt(0) = layers_[0].getDeltas().sum();
    return delt;
}

