#include "Data.hpp"
#include "ConfigIni.hpp"

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <limits>
#include <tuple>
#include <filesystem>

/**
 * Trim helper function
 */
static inline std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b-1])) --b;
    return s.substr(a, b-a);
}

/**
 * Helper function to parse a CSV line into fields (handles quotes)
 */
void Data::splitCSVLine(const std::string& line, std::vector<std::string>& outFields) {
    outFields.clear();
    outFields.reserve(16);

    std::string cur;
    bool inQuote = false;
    for (size_t i = 0; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"' ) {
            // handle double quotes inside quotes ("")
            if (inQuote && i + 1 < line.size() && line[i+1] == '"') {
                cur.push_back('"');
                ++i; // skip second quote
            } else {
                inQuote = !inQuote;
            }
        } else if (c == ',' && !inQuote) {
            outFields.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    outFields.push_back(cur);
}

/**
 * Helper function to build index vector [0..N-1] and optionally shuffle it with seed
 */
static std::vector<int> buildIndicesInt(int N, bool shuffle, unsigned seed) {
    std::vector<int> idx(N);
    for (int i = 0; i < N; ++i) idx[i] = i;
    if (shuffle && N > 1) {
        std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
        std::shuffle(idx.begin(), idx.end(), gen);
    }
    return idx;
}

/**
 * Helper function to build mapping filtered_index -> original_index (used when rows were removed)
 */
static std::vector<Eigen::Index> buildFilteredToOriginalMap(const std::vector<size_t>& na_row_indices, Eigen::Index origRows) {
    std::vector<Eigen::Index> filt2orig;
    filt2orig.reserve(static_cast<size_t>(origRows));
    std::size_t remPos = 0;
    for (Eigen::Index orig = 0; orig < origRows; ++orig) {
        if (remPos < na_row_indices.size() && static_cast<Eigen::Index>(na_row_indices[remPos]) == orig) {
            ++remPos; // this original row was removed
        } else {
            filt2orig.push_back(orig);
        }
    }
    return filt2orig;
}

/**
 * Helper function to save matrix to CSV
 */
bool Data::saveMatrixCsv(const std::string &path,
    const Eigen::MatrixXd &M,
    const std::vector<std::string> &colNames,
    bool inverseOutputs) const
    {
    try {
        if (M.size() == 0) {
            std::cerr << "[Data::saveMatrixCsv] Empty matrix, nothing to save: " << path << "\n";
            return false;
        }

        // prepare output matrix: optionally inverse-transform using data's scaler
        Eigen::MatrixXd out = M;
        if (inverseOutputs) {
            try {
                out = this->inverseTransformOutputs(M);
            } catch (const std::exception &ex) {
                std::cerr << "[Data::saveMatrixCsv] inverseTransformOutputs failed: " << ex.what()
                          << " — saving transformed values instead.\n";
                out = M;
            }
        }

        // ensure directory exists
        std::filesystem::path p(path);
        if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());

        std::ofstream ofs(path);
        if (!ofs.is_open()) {
            std::cerr << "[Data::saveMatrixCsv] Cannot open file for write: " << path << "\n";
            return false;
        }
        ofs << std::setprecision(12);

        // optional header
        if (!colNames.empty()) {
            for (size_t c = 0; c < colNames.size(); ++c) {
                ofs << colNames[c];
                if (c + 1 < colNames.size()) ofs << ",";
            }
            ofs << "\n";
        }

        for (Eigen::Index r = 0; r < out.rows(); ++r) {
            for (Eigen::Index c = 0; c < out.cols(); ++c) {
                double v = out(r, c);
                if (std::isfinite(v)) ofs << v;
                else ofs << "NaN";
                if (c + 1 < out.cols()) ofs << ",";
            }
            ofs << "\n";
        }
        ofs.close();
        return true;
    } catch (const std::exception &ex) {
        std::cerr << "[Data::saveMatrixCsv] Exception: " << ex.what() << "\n";
        return false;
    }
}

/**
 * Loads and filters the CSV file and returns number of loaded rows
 */
size_t Data::loadFilteredCSV(const std::string& path,
  const std::unordered_set<std::string>& idFilter,
  const std::vector<std::string>& keepColumns,  // names of numeric columns to extract (e.g. "T1","T2","T3","moisture")
  const std::string& timestampCol,  // name of timestamp column (e.g "hour_start", "date") 
  const std::string& idCol)  // ID of the selected sensor
  
  {
    // Clear or set needed variables
    m_timestamps.clear();  
    m_data.resize(0,0);
    m_colNames = keepColumns;

    // Deal with errors of the CSV file
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open CSV: " + path);
    }

    std::string headerLine;
    if (!std::getline(ifs, headerLine)) {
        throw std::runtime_error("CSV file empty: " + path);
    }

    // Parse header line into fields
    std::vector<std::string> headers;
    splitCSVLine(headerLine, headers);
  
    // Build header to index map (trim headers, reserve space, warn on duplicates)
    std::unordered_map<std::string, size_t> idxMap;
    idxMap.reserve(headers.size());
    for (size_t i = 0; i < headers.size(); ++i) {
        std::string h = trim(headers[i]);  // remove surrounding whitespace
      
        // replace original headers with trimmed form so later code uses same names
        headers[i] = h;

        auto it = idxMap.find(h);
        if (it != idxMap.end()) {
            std::cerr << "[Warning]: Duplicate header '" << h << "'; using last occurrence at column " << i << "\n";
        }
        idxMap[h] = i;
    }

    // Check if required columns exist
    if (idxMap.find(idCol) == idxMap.end()) {
        throw std::runtime_error("ID column not found: " + idCol);
    }
    if (idxMap.find(timestampCol) == idxMap.end()) {
        throw std::runtime_error("Timestamp column not found: " + timestampCol);
    }
    std::vector<size_t> keepIdx;
    keepIdx.reserve(keepColumns.size());
    for (auto &name : keepColumns) {
        auto it = idxMap.find(name);
        if (it == idxMap.end()) {
            throw std::runtime_error("Requested column not found in CSV: " + name);
        }
        keepIdx.push_back(it->second);
    }

    // Prepare a container for rows (we don't know the count up front)
    //std::vector<std::array<double, 1>> dummy;
    std::vector<std::vector<double>> rows;
    rows.reserve(1024);
    std::vector<std::string> times;
    times.reserve(1024);

    std::string line;
    std::vector<std::string> fields;
    size_t lineNo = 1;
    while (std::getline(ifs, line)) {
        ++lineNo;
        if (line.empty()) continue;
        splitCSVLine(line, fields);
        
        // Deal with NAs or missing values
        static bool warned_short_row = false;
        if (fields.size() < headers.size()) {
            if (!warned_short_row) {
                std::cerr << "[Warning]: Some rows have fewer fields than header.\n";
                warned_short_row = true;
            }
            fields.resize(headers.size());
        }

        // Filter rows by ID and skip row if its ID is not in idFilter
        std::string idValue = trim(fields[idxMap.at(idCol)]);
        if (!idFilter.empty() && idFilter.find(idValue) == idFilter.end()) {
            continue;
        }

        // Read and trim timestamp and parse selected numeric columns (empty or bad -> NaN).
        std::string ts = trim(fields[idxMap.at(timestampCol)]);
        std::vector<double> numeric;
        numeric.reserve(keepIdx.size());
        for (size_t j = 0; j < keepIdx.size(); ++j) {
            std::string cell = trim(fields[keepIdx[j]]);
            if (cell.empty()) {
                numeric.push_back(std::numeric_limits<double>::quiet_NaN());
            } else {
                try {
                    double v = std::stod(cell);
                    numeric.push_back(v);
                } catch (...) {
                    // if parse fails, push NaN and continue
                    numeric.push_back(std::numeric_limits<double>::quiet_NaN());
                }
            }
        }

        // Append to storage
        times.push_back(ts);
        rows.push_back(std::move(numeric));
    }

    // Fill Eigen matrix with the data
    const size_t nrows = rows.size();
    const size_t ncols = keepIdx.size();
    m_data.resize(nrows, ncols);
    for (size_t r = 0; r < nrows; ++r) {
        for (size_t c = 0; c < ncols; ++c) {
            m_data(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c)) = rows[r][c];
        }
    }
    m_timestamps = std::move(times);

    // Initialize scaler vectors to match number of columns
    Eigen::Index C = m_data.cols();
    m_scaler.min = Eigen::VectorXd::Zero(C);
    m_scaler.max = Eigen::VectorXd::Zero(C);
    m_scaler.fitted = false;

    return nrows;
}

/**
 * Clean all files in a directory 
 */
void Data::cleanDirectory(const std::string &path) {
    namespace fs = std::filesystem;
    try {
        if (fs::exists(path)) {
            fs::remove_all(path);   // remove directory and everything inside
        }
        fs::create_directories(path); // recreate the empty directory
    } catch (const std::exception &e) {
        std::cerr << "[Data::cleanDirectory] Failed to clean: " << path 
                  << " (" << e.what() << ")\n";
    }
}

/**
 * Clean all files in output directory 
 */
void Data::cleanAllOutputs(const std::string &outDir) {
    namespace fs = std::filesystem;
    try {
        if (!fs::exists(outDir)) {
            fs::create_directories(outDir);
            return;
        }

        // Iterate subdirectories in outDir
        for (const auto &entry : fs::directory_iterator(outDir)) {
            if (fs::is_directory(entry)) {
                fs::remove_all(entry.path());
                fs::create_directories(entry.path()); // recreate empty
            } else {
                fs::remove(entry.path()); // remove stray files in root
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "[Data::cleanAllOutputs] Failed to clean: " << outDir
                  << " (" << e.what() << ")\n";
    }
}

/**
 * Write model settings into a log file
 */
void Data::logRunSettings(std::ostream& os, const RunConfig& cfg, unsigned run_id) const {
    os << "================ Run " << run_id << " Settings ================\n";

    // --- Data ---
    os << "Data file: " << cfg.data_file << "\n";
    os << "ID filter: " << (cfg.id.empty() ? "(none)" : cfg.id) << "\n";
    os << "Columns: ";
    for (const auto& c : cfg.columns) os << c << " ";
    os << "\n";

    // --- Model ---
    os << "Architecture: ";
    for (auto v : cfg.mlp_architecture) os << v << " ";
    os << "\n";
    os << "Trainer: "        << cfg.trainer        << "\n";
    os << "Activation: "     << cfg.activation     << "\n";
    os << "Weight init: "    << cfg.weight_init    << "\n";
    os << "Input numbers: ";
    for (const auto& v : cfg.input_numbers) {
        for (int off : v)
            os << off << ",";
        os << " | ";
    }
    os << "\n";

    // --- Training ---
    os << "Learning rate: "  << cfg.learning_rate  << "\n";
    os << "Max iterations: " << cfg.max_iterations << "\n";
    os << "Max metrics step: " << cfg.max_metrics_step << "\n";
    os << "Max error: "      << cfg.max_error      << "\n";
    os << "Batch size: "     << cfg.batch_size     << "\n";
    os << "Ensemble runs: "  << cfg.ensemble_runs  << "\n";
    os << "Train fraction: " << cfg.train_fraction << "\n";
    os << "Shuffle: "        << (cfg.shuffle ? "true" : "false") << "\n";
    os << "Split shuffle: "  << (cfg.split_shuffle ? "true" : "false") << "\n";
    os << "Seed: "           << cfg.seed           << "\n";

    // --- Transform ---
    os << "Transform: " << cfg.transform << "\n";
    os << "Transform alpha: " << cfg.transform_alpha << "\n";
    os << "Exclude last col from transform: "
       << (cfg.exclude_last_col_from_transform ? "true" : "false") << "\n";
    os << "Remove NA before calib: "
       << (cfg.remove_na_before_calib ? "true" : "false") << "\n";

    // --- Optimization ---
    os << "PSO optimize: " << (cfg.pso_optimize ? "true" : "false") << "\n";

    // --- Paths ---
    os << "Output dir: " << cfg.out_dir << "\n";
    os << "Log dir: " << cfg.log_dir << "\n";
    os << "Real calib data: " << cfg.real_calib << "\n";
    os << "Predict calib data: " << cfg.pred_calib << "\n";
    os << "Real valid data: " << cfg.real_valid << "\n";
    os << "Predict valid data: " << cfg.pred_valid << "\n";
    os << "Weights init: " << cfg.weights_csv_init << "\n";
    os << "Weights final: " << cfg.weights_csv << "\n";
    os << "Metrics calib: " << cfg.metrics_cal << "\n";
    os << "Metrics valid: " << cfg.metrics_val << "\n";

    os << "===========================================================\n";
}

/**
 * Getter for the timestamps
 */
std::vector<std::string> Data::timestamps() const {
  return m_timestamps;
} 

/**
 *  Getter for the data numeric matrix
 */
Eigen::MatrixXd Data::numericData() const {
  return m_data;
} 

/**
 *  Setter for the data numeric matrix
 */
void Data::setNumericData(const Eigen::MatrixXd &newData){
    m_data = newData;
}

/**
 * Getter for the names of numeric columns
 */
std::vector<std::string> Data::numericColNames() const {
  return m_colNames;
} 

/**
 * Print header line, i.e. timestamp + numeric column names
 */
void Data::printHeader(const std::string& timestampColName) const {
    std::cout << "Header: " << timestampColName;
    const auto& cols = numericColNames();

    if (!cols.empty()) std::cout << " | ";

    for (size_t i = 0; i < cols.size(); ++i) {
        std::cout << cols[i];
        if (i + 1 < cols.size()) std::cout << " | ";
    }
    std::cout << "\n";
}

/**
 * Return a copy of the values in a selected column by name
 */
std::vector<double> Data::getColumnValues(const std::string& name) const {
    const auto& cols = numericColNames();
    auto it = std::find(cols.begin(), cols.end(), name);

    if (it == cols.end()) throw std::out_of_range("Column not found: " + name);

    size_t idx = std::distance(cols.begin(), it);
    const auto& mat = numericData();
    std::vector<double> out;
    out.reserve(mat.rows());
    
    for (int r = 0; r < mat.rows(); ++r) out.push_back(mat(r, static_cast<int>(idx)));

    return out;
}

/**
 * Set which transform to apply (applies to all numeric columns)
 */
void Data::setTransform(transform_type t, double alpha, bool excludeLastCol) {
    m_transform = t;
    m_alpha = alpha;
    m_excludeLastCol = excludeLastCol;

    // ensure scaler vectors have correct size (will be filled when applying MINMAX)
    Eigen::Index cols = m_data.cols();
    if (cols <= 0) {
        m_scaler.min.resize(0);
        m_scaler.max.resize(0);
        m_scaler.fitted = false;
    } else {
        m_scaler.min = Eigen::VectorXd::Zero(cols);
        m_scaler.max = Eigen::VectorXd::Zero(cols);
        m_scaler.fitted = false;
    }
}

/**
 * Apply the previously configured transform to m_data
 */
void Data::applyTransform() {
    if (m_transform == transform_type::NONE) return;
    const Eigen::Index R = m_data.rows();
    const Eigen::Index C = m_data.cols();
    if (R == 0 || C == 0) return;

    // helper to decide whether to operate on column c
    auto shouldTransformCol = [&](int c)->bool {
        if (!m_excludeLastCol) return true;
        return c != static_cast<int>(C) - 1;
    };

    switch (m_transform) {
        
        case transform_type::MINMAX:
        {
            // compute min/max and scale each column to [0,1]
            for (Eigen::Index c = 0; c < C; ++c) {
                if (!shouldTransformCol(static_cast<int>(c))) {
                    m_scaler.min(c) = 0.0;
                    m_scaler.max(c) = 1.0;
                    continue;
                }
                double mn = std::numeric_limits<double>::infinity();
                double mx = -std::numeric_limits<double>::infinity();
                std::size_t cnt = 0;
                for (Eigen::Index r = 0; r < R; ++r) {
                    double v = m_data(r, c);
                    if (std::isfinite(v)) { mn = std::min(mn, v); mx = std::max(mx, v); ++cnt; }
                }
                if (cnt == 0) { mn = 0.0; mx = 1.0; }
                m_scaler.min(c) = mn;
                m_scaler.max(c) = mx;
                double span = (mx - mn); if (span == 0.0) span = 1.0;
                for (Eigen::Index r = 0; r < R; ++r) {
                    double &x = m_data(r, c);
                    if (std::isfinite(x)) x = (x - mn) / span;
                }
            }
            m_scaler.fitted = true;
            break;
        }

        case transform_type::NONLINEAR:
        {
            double alpha = m_alpha;
            for (Eigen::Index c = 0; c < C; ++c) {
                if (!shouldTransformCol(static_cast<int>(c))) continue;
                for (Eigen::Index r = 0; r < R; ++r) {
                    double v = m_data(r, c);
                    if (std::isfinite(v)) {
                        double t = 1.0 - std::exp(-alpha * v);
                        m_data(r, c) = std::isfinite(t) ? t : std::numeric_limits<double>::quiet_NaN();
                    }
                }
            }
            break;
        }

        case transform_type::ZSCORE:    // t = (x - mean) / sd
        {
            for (Eigen::Index c = 0; c < C; ++c) {
                if (!shouldTransformCol(static_cast<int>(c))) continue;

                double sum = 0.0;
                double sumsq = 0.0;
                Eigen::Index count = 0;

                for (Eigen::Index r = 0; r < R; ++r) {
                    double v = m_data(r, c);
                    if (std::isfinite(v)) {
                        sum += v;
                        sumsq += v * v;
                        ++count;
                    }
                }

                if (count > 1) {
                    double mean = sum / static_cast<double>(count);
                    double variance = (sumsq / static_cast<double>(count)) - (mean * mean);
                    double stddev = variance > 0.0 ? std::sqrt(variance) : 0.0;

                    for (Eigen::Index r = 0; r < R; ++r) {
                        double v = m_data(r, c);
                        if (std::isfinite(v)) {
                            if (stddev > 0.0) {
                                double t = (v - mean) / stddev;
                                m_data(r, c) = std::isfinite(t) ? t : std::numeric_limits<double>::quiet_NaN();
                            } else {
                                m_data(r, c) = 0.0;
                            }
                        }
                    }
                    // using Scaler struct also for zscore - min = mean, max = sd
                    // might confuse, should change later
                    m_scaler.min(c) = mean;
                    m_scaler.max(c) = stddev;
                    m_scaler.fitted = true;
                }
            }
            break;
        }

        default:
            break;
    }
}

/**
 * Inverse the global transform (to bring predictions back)
 */
void Data::inverseTransform() {
    if (m_transform == transform_type::NONE) return;
    const Eigen::Index R = m_data.rows();
    const Eigen::Index C = m_data.cols();
    if (R == 0 || C == 0) return;

    // small tolerances for numeric clipping
    //const double eps_clip_low = 1e-12;    // allow tiny negative numbers
    //const double eps_clip_high = 1e-12;   // allow tiny >1 numbers to be clamped below 1.0

    auto shouldTransformCol = [&](int c)->bool {
        if (!m_excludeLastCol) return true;
        return c != static_cast<int>(C) - 1;
    };

    switch (m_transform) {
        case transform_type::MINMAX:
        {
            if (!m_scaler.fitted) throw std::runtime_error("Scaler not fitted for inverse");
            for (Eigen::Index c = 0; c < C; ++c) {
                if (!shouldTransformCol(static_cast<int>(c))) continue;
                double mn = m_scaler.min(c);
                double mx = m_scaler.max(c);
                double span = (mx - mn); if (span == 0.0) span = 1.0;
                for (Eigen::Index r = 0; r < R; ++r) {
                    double &x = m_data(r, c);
                    if (std::isfinite(x)) x = x * span + mn;
                }
            }
            break;
        }

        case transform_type::NONLINEAR:
        {
            const double alpha = m_alpha; 
            for (Eigen::Index c = 0; c < C; ++c) {
                if (!shouldTransformCol(static_cast<int>(c))) continue;
                for (Eigen::Index r = 0; r < R; ++r) {
                    double &t = m_data(r, c);
                    if (!std::isfinite(t)) continue;  // preserve NaN/Inf handling
                    // If t >= 1.0 due to rounding/saturation, nudge it below 1.0
                    if (t >= 1.0) t = std::nextafter(1.0, 0.0);
                    // Now safe: compute inverse (allows negative t)
                    double one_minus = 1.0 - t;  // > 0 here
                    if (!(one_minus > 0.0)) throw std::runtime_error("inverseGlobalTransform: 1 - D_trans <= 0");
                    t = -std::log(one_minus) / alpha;
                }
            }
            break;
        }

        case transform_type::ZSCORE:
        {
            if (!m_scaler.fitted) throw std::runtime_error("Scaler not fitted for inverse");
            for (Eigen::Index c = 0; c < C; ++c) {
                if (!shouldTransformCol(static_cast<int>(c))) continue;
                double mean = m_scaler.min(c);
                double stddev = m_scaler.max(c);
                for (Eigen::Index r = 0; r < R; ++r) {
                    double &x = m_data(r, c);
                    if (std::isfinite(x)) x = x * stddev + mean;
                }
            }
            break;
        }

        default:
            break;
    }
}

/**
 * Inverse the global transform for outputs
 */
Eigen::MatrixXd Data::inverseTransformOutputs(const Eigen::MatrixXd& M) const {
    if (M.size() == 0) return M;

    switch (m_transform) {
        case transform_type::NONE:
            return M; // nothing to do

        case transform_type::MINMAX: {
            if (!m_scaler.fitted) {
                throw std::runtime_error("inverseTransformOutputs: MINMAX scaler not fitted");
            }
            // We assume the target column in original m_data is the last column.
            // The MINMAX scaling is per-column; for outputs we apply the inverse min/max for the target col.
            const Eigen::Index targetCol = m_data.cols() - 1;
            if (targetCol < 0) throw std::runtime_error("inverseTransformOutputs: no columns in m_data");
            double mn = m_scaler.min(targetCol);
            double mx = m_scaler.max(targetCol);
            double span = (mx - mn); if (span == 0.0) span = 1.0;

            Eigen::MatrixXd out = M;
            out = out.array() * span + mn;
            return out;
        }

        case transform_type::NONLINEAR: {
            const double alpha = m_alpha;
            Eigen::MatrixXd out = M;
            for (Eigen::Index r = 0; r < out.rows(); ++r) {
                for (Eigen::Index c = 0; c < out.cols(); ++c) {
                    double t = out(r, c);
                    if (!std::isfinite(t)) continue;
                    // clamp / nudge to valid open interval (0,1)
                    if (t >= 1.0) t = std::nextafter(1.0, 0.0);
                    if (t < 0.0) {
                        if (t > -1e-12) t = 0.0; // treat tiny negative as zero
                        else throw std::runtime_error("inverseTransformOutputs: transformed value < 0");
                    }
                    double one_minus = 1.0 - t;
                    if (!(one_minus > 0.0)) throw std::runtime_error("inverseTransformOutputs: 1 - t <= 0");
                    out(r, c) = -std::log(one_minus) / alpha;
                }
            }
            return out;
        }

        case transform_type::ZSCORE: {
            if (!m_scaler.fitted) {
                throw std::runtime_error("inverseTransformOutputs: scaler not fitted");
            }
            // We assume the target column in original m_data is the last column...

            const Eigen::Index targetCol = m_data.cols() - 1;
            if (targetCol < 0) throw std::runtime_error("inverseTransformOutputs: no columns in m_data");
            double mean = m_scaler.min(targetCol);
            double stddev = m_scaler.max(targetCol);

            Eigen::MatrixXd out = M;
            out = out.array() * stddev + mean;
            return out;
        }

        default:
            throw std::runtime_error("inverseTransformOutputs: unknown transform_type");
    }
}

/**
 * Create matrix for backpropagation from data matrix (with NA removal)
 */
void Data::makeCalibMat(std::vector<int> inpNumsOfVars, int outRows) {
    if (outRows <= 0 || std::any_of(inpNumsOfVars.begin(), inpNumsOfVars.end(), [](int x){ return x < 0; }))
        throw std::invalid_argument("inpNumsOfVars values and outRows must be positive");

    const auto maxR = *std::max_element(inpNumsOfVars.begin(), inpNumsOfVars.end());
    if (maxR == 0)
        throw std::runtime_error("At least one value in inpNumsOfVars must be greater than 0");

    const size_t DC = m_data.cols();
    if (DC < 1) throw std::runtime_error("Data has no columns");
    if (inpNumsOfVars.size() != DC) throw std::invalid_argument("inpNumsOfVars size does not match data columns");

    const int CRcand = static_cast<int>(m_data.rows()) - static_cast<int>(maxR) - outRows + 1;
    if (CRcand <= 0) throw std::runtime_error("Not enough rows to build calibration matrices with given inpNumsOfVars/outRows");

    const int CC = static_cast<int>(std::accumulate(inpNumsOfVars.begin(), inpNumsOfVars.end(), 0)) + outRows;

    std::vector<std::vector<double>> rows;
    rows.reserve(static_cast<size_t>(std::max(0, CRcand)));
    m_calib_pattern_filtered_indices.clear();
    m_calib_pattern_orig_indices.clear();

    std::vector<Eigen::Index> filt2orig;
    if (m_has_filtered_rows) {
        filt2orig = buildFilteredToOriginalMap(m_na_row_indices, static_cast<Eigen::Index>(m_data_backup.rows()));
    }

    for (int i = 0; i < CRcand; ++i) {
        bool ok = true;
        std::vector<double> row;
        row.reserve(CC);
        // inputs
        for (size_t j = 0; j < DC; ++j) {
            for (int l = 0; l < inpNumsOfVars[j]; ++l) {
                int rindex = i + static_cast<int>(maxR) - inpNumsOfVars[j] + l;
                double v = m_data(rindex, static_cast<Eigen::Index>(j));
                if (!std::isfinite(v)) { ok = false; break; }
                row.push_back(v);
            }
            if (!ok) break;
        }
        if (!ok) continue;

        // outputs
        for (int j = 0; j < outRows; ++j) {
            int rindex = i + static_cast<int>(maxR) + j;
            double v = m_data(rindex, static_cast<Eigen::Index>(DC - 1));
            if (!std::isfinite(v)) { ok = false; break; }
            row.push_back(v);
        }
        if (!ok) continue;

        rows.push_back(std::move(row));

        int filtered_ref = i + static_cast<int>(maxR);
        m_calib_pattern_filtered_indices.push_back(filtered_ref);
        if (m_has_filtered_rows) {
            if (filtered_ref < 0 || filtered_ref >= static_cast<int>(filt2orig.size()))
                throw std::runtime_error("internal mapping error while building orig indices (makeCalibMat)");
            m_calib_pattern_orig_indices.push_back(static_cast<size_t>(filt2orig[static_cast<size_t>(filtered_ref)]));
        } else {
            m_calib_pattern_orig_indices.push_back(static_cast<size_t>(filtered_ref));
        }
    }

    const int valid = static_cast<int>(rows.size());
    calibMat = Eigen::MatrixXd::Constant(valid, CC, std::numeric_limits<double>::quiet_NaN());
    for (int r = 0; r < valid; ++r) {
        for (int c = 0; c < CC; ++c) calibMat(r, c) = rows[r][c];
    }

    if (valid > 0) {
        splitCalibMat(static_cast<int>(std::accumulate(inpNumsOfVars.begin(), inpNumsOfVars.end(), 0)));
    } else {
        calibInpsMat.resize(0,0);
        calibOutsMat.resize(0,0);
    }
}

/**
 * Create separate calibration inps and outs matrices for backpropagation from data matrix (with NA removal)
 */
void Data::makeCalibMatsSplit(std::vector<int> inpNumsOfVars, int outRows) {
    if (outRows <= 0 || std::any_of(inpNumsOfVars.begin(), inpNumsOfVars.end(), [](int x){ return x < 0; }))
        throw std::invalid_argument("inpNumsOfVars values and outRows must be positive");

    const auto maxR = *std::max_element(inpNumsOfVars.begin(), inpNumsOfVars.end());
    if (maxR == 0)
        throw std::runtime_error("At least one value in inpNumsOfVars must be greater than 0");

    const size_t DC = m_data.cols();
    if (DC < 1) throw std::runtime_error("Data has no columns");
    if (inpNumsOfVars.size() != DC) throw std::invalid_argument("inpNumsOfVars size does not match data columns");

    const int CRcand = static_cast<int>(m_data.rows()) - static_cast<int>(maxR) - outRows + 1;
    if (CRcand <= 0) throw std::runtime_error("Not enough rows to build calibration matrices with given inpNumsOfVars/outRows");

    const int inpC = static_cast<int>(std::accumulate(inpNumsOfVars.begin(), inpNumsOfVars.end(), 0));
    std::vector<std::vector<double>> rowsIn;
    std::vector<std::vector<double>> rowsOut;
    rowsIn.reserve(static_cast<size_t>(std::max(0, CRcand)));
    rowsOut.reserve(static_cast<size_t>(std::max(0, CRcand)));
    m_calib_pattern_filtered_indices.clear();
    m_calib_pattern_orig_indices.clear();

    // Prepare map filtered -> original if original backup exists
    std::vector<Eigen::Index> filt2orig;
    if (m_has_filtered_rows) {
        filt2orig = buildFilteredToOriginalMap(m_na_row_indices, static_cast<Eigen::Index>(m_data_backup.rows()));
    }

    for (int i = 0; i < CRcand; ++i) {
        bool ok = true;
        std::vector<double> inrow;
        inrow.reserve(inpC);
        // collect inputs (for each column j, take last inpNumsOfVars[j] values relative to i+maxR)
        for (size_t j = 0; j < DC; ++j) {
            for (int l = 0; l < inpNumsOfVars[j]; ++l) {
                int rindex = i + static_cast<int>(maxR) - inpNumsOfVars[j] + l;
                double v = m_data(rindex, static_cast<Eigen::Index>(j));
                if (!std::isfinite(v)) { ok = false; break; }
                inrow.push_back(v);
            }
            if (!ok) break;
        }
        if (!ok) continue;

        // collect outputs (target column assumed last column)
        std::vector<double> outrow;
        outrow.reserve(outRows);
        for (int j = 0; j < outRows; ++j) {
            int rindex = i + static_cast<int>(maxR) + j;
            double v = m_data(rindex, static_cast<Eigen::Index>(DC - 1));
            if (!std::isfinite(v)) { ok = false; break; }
            outrow.push_back(v);
        }
        if (!ok) continue;

        // accept the pattern
        rowsIn.push_back(std::move(inrow));
        rowsOut.push_back(std::move(outrow));

        // filtered index that corresponds to the first output (reference) row
        int filtered_ref = i + static_cast<int>(maxR);
        m_calib_pattern_filtered_indices.push_back(filtered_ref);

        // compute original index
        if (m_has_filtered_rows) {
            if (filtered_ref < 0 || filtered_ref >= static_cast<int>(filt2orig.size()))
                throw std::runtime_error("internal mapping error while building orig indices");
            m_calib_pattern_orig_indices.push_back(static_cast<size_t>(filt2orig[static_cast<size_t>(filtered_ref)]));
        } else {
            m_calib_pattern_orig_indices.push_back(static_cast<size_t>(filtered_ref));
        }
    }

    // pack into Eigen matrices
    const int valid = static_cast<int>(rowsIn.size());
    calibInpsMat = Eigen::MatrixXd::Constant(valid, inpC, std::numeric_limits<double>::quiet_NaN());
    calibOutsMat = Eigen::MatrixXd::Constant(valid, outRows, std::numeric_limits<double>::quiet_NaN());

    for (int r = 0; r < valid; ++r) {
        for (int c = 0; c < inpC; ++c) calibInpsMat(r, c) = rowsIn[r][c];
        for (int c = 0; c < outRows; ++c) calibOutsMat(r, c) = rowsOut[r][c];
    }

    // build combined calibMat (left inputs, right outs)
    calibMat = Eigen::MatrixXd::Constant(valid, inpC + outRows, std::numeric_limits<double>::quiet_NaN());
    if (valid > 0) {
        calibMat.leftCols(inpC) = calibInpsMat;
        calibMat.rightCols(outRows) = calibOutsMat;
    }
}

/**
 * Split created calibration matrix into separate inps and outs matrices
 */
void Data::splitCalibMat(int inpLength){
    if (inpLength <= 0 || inpLength >= calibMat.cols())
        throw std::invalid_argument("inpLength must be greater and 0 and less than calibMat columns");

    calibInpsMat = calibMat.leftCols(inpLength);
    calibOutsMat = calibMat.rightCols(calibMat.cols() - inpLength);
}

/**
 * Split calib and valid dataset
 */
std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
Data::splitInputsOutputs(const Eigen::MatrixXd &mat, int inpSize, int outSize) const {
    if (inpSize + outSize != mat.cols()) {
        throw std::runtime_error("[Data::splitInputsOutputs] Column mismatch: "
                                 + std::to_string(inpSize + outSize) +
                                 " expected, got " + std::to_string(mat.cols()));
    }

    Eigen::MatrixXd X = mat.leftCols(inpSize);
    Eigen::MatrixXd Y = mat.rightCols(outSize);
    return {X, Y};
}

/**
 * Split calibration matrix into train/validation and also return indices
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, std::vector<int>, std::vector<int>>
Data::splitCalibMatWithIdx(double trainFraction, bool shuffle, unsigned seed) const {
    if (trainFraction <= 0.0 || trainFraction >= 1.0) {
        throw std::invalid_argument("splitCalibMatWithIdx: trainFraction must be in (0,1)");
    }

    const Eigen::Index R = calibMat.rows();
    const Eigen::Index C = calibMat.cols();
    if (R == 0) {
        return { Eigen::MatrixXd(0, C), Eigen::MatrixXd(0, C), std::vector<int>{}, std::vector<int>{} };
    }

    std::vector<int> idx = buildIndicesInt(static_cast<int>(R), shuffle, seed);

    // compute training count
    int nTrain = static_cast<int>(std::floor(trainFraction * static_cast<double>(R)));
    nTrain = std::max(1, std::min<int>(nTrain, static_cast<int>(R) - 1));

    Eigen::MatrixXd trainMat(nTrain, C);
    Eigen::MatrixXd validMat(R - nTrain, C);
    std::vector<int> trainIdx;
    std::vector<int> validIdx;
    trainIdx.reserve(static_cast<size_t>(nTrain));
    validIdx.reserve(static_cast<size_t>(R - nTrain));

    for (int i = 0; i < nTrain; ++i) {
        trainMat.row(i) = calibMat.row(idx[i]);
        trainIdx.push_back(idx[i]);
    }
    for (int i = nTrain; i < static_cast<int>(R); ++i) {
        validMat.row(i - nTrain) = calibMat.row(idx[i]);
        validIdx.push_back(idx[i]);
    }

    return { trainMat, validMat, trainIdx, validIdx };
}

/**
 * Split raw data rows (m_data) into train/validation and return indices
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, std::vector<int>, std::vector<int>>
Data::splitDataRowsWithIdx(double trainFraction, bool shuffle, unsigned seed) const {
    if (trainFraction <= 0.0 || trainFraction >= 1.0) {
        throw std::invalid_argument("splitDataRowsWithIdx: trainFraction must be in (0,1)");
    }

    const Eigen::Index R = m_data.rows();
    const Eigen::Index C = m_data.cols();
    if (R == 0) {
        return { Eigen::MatrixXd(0, C), Eigen::MatrixXd(0, C), std::vector<int>{}, std::vector<int>{} };
    }

    std::vector<int> idx = buildIndicesInt(static_cast<int>(R), shuffle, seed);

    int nTrain = static_cast<int>(std::floor(trainFraction * static_cast<double>(R)));
    nTrain = std::max(1, std::min<int>(nTrain, static_cast<int>(R) - 1));

    Eigen::MatrixXd trainMat(nTrain, C);
    Eigen::MatrixXd validMat(R - nTrain, C);
    std::vector<int> trainIdx;
    std::vector<int> validIdx;
    trainIdx.reserve(static_cast<size_t>(nTrain));
    validIdx.reserve(static_cast<size_t>(R - nTrain));

    for (int i = 0; i < nTrain; ++i) {
        trainMat.row(i) = m_data.row(idx[i]);
        trainIdx.push_back(idx[i]);
    }
    for (int i = nTrain; i < static_cast<int>(R); ++i) {
        validMat.row(i - nTrain) = m_data.row(idx[i]);
        validIdx.push_back(idx[i]);
    }

    return { trainMat, validMat, trainIdx, validIdx };
}


/**
 * Getter for calibration matrix
 */
Eigen::MatrixXd Data::getCalibMat(){
    return calibMat;
}

/**
 * Setter for calibration matrix
 */
void Data::setCalibMat(const Eigen::MatrixXd &newMat){
    calibMat = newMat;
}

/**
 * Getter for calibration inputs matrix
 */
Eigen::MatrixXd Data::getCalibInpsMat(){
    return calibInpsMat;
}

/**
 * Setter for calibration inputs matrix
 */
void Data::setCalibInpsMat(const Eigen::MatrixXd &newMat){
    calibInpsMat = newMat;
}

/**
 * Getter for calibration outputs matrix
 */
Eigen::MatrixXd Data::getCalibOutsMat(){
    return calibOutsMat;
}

/**
 * Setter for calibration outputs matrix
 */
void Data::setCalibOutsMat(const Eigen::MatrixXd &newMat){
    calibOutsMat = newMat;
}

/**
 * Create random permutation vector for shuffling
 */
std::vector<int> Data::permutationVector(int length){
    if (length <= 0)
        throw std::invalid_argument("length must be greater than 0");

    std::vector<int> permVec(length);
    std::iota(permVec.begin(), permVec.end(), 0);
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::shuffle(permVec.begin(), permVec.end(), gen);
    
    return permVec;
}

/**
 * Shuffle matrix rows
 */
Eigen::MatrixXd Data::shuffleMatrix(const Eigen::MatrixXd &matrix, const std::vector<int>& permVec){
    size_t rows = matrix.rows();
    if (rows != permVec.size())
        throw std::invalid_argument("matrix rows and permVec length dont match");

    Eigen::MatrixXd newmat(rows, matrix.cols());
    for (size_t i = 0; i < rows; ++i) {
        newmat.row(i) = matrix.row(permVec[i]);
    }
    return newmat;
}

/**
 * Unshuffle matrix rows
 */
Eigen::MatrixXd Data::unshuffleMatrix(const Eigen::MatrixXd &matrix, const std::vector<int>& permVec) {
    size_t rows = matrix.rows();
    if (rows != permVec.size())
        throw std::invalid_argument("matrix rows and permVec length dont match");

    Eigen::MatrixXd oldmat(rows, matrix.cols());
    for (size_t i = 0; i < rows; ++i) {
        oldmat.row(permVec[i]) = matrix.row(i);
    }
    return oldmat;
}


/**
 * Find indices of rows that contain any NaN in numeric data
 */
std::vector<size_t> Data::findRowsWithNa() const {
    std::vector<size_t> out;
    if (m_data.size() == 0) return out;
    const Eigen::Index R = m_data.rows();
    const Eigen::Index C = m_data.cols();
    for (Eigen::Index r = 0; r < R; ++r) {
        bool rowHasNa = false;
        for (Eigen::Index c = 0; c < C; ++c) {
            double v = m_data(r, c);
            if (!std::isfinite(v)) { rowHasNa = true; break; }
        }
        if (rowHasNa) out.push_back(static_cast<size_t>(r));
    }
    return out;
}

/**
 * Remove rows that contain any NaN from m_data and m_timestamps, but keep backups and record removed indices so they can be restored later
 */
void Data::removeRowsWithNa() {
    if (m_has_filtered_rows) {
        // already filtered — do nothing
        return;
    }

    // find rows to remove
    auto naIdx = findRowsWithNa();
    if (naIdx.empty()) {
        // no NAs — nothing to do
        m_has_filtered_rows = false;
        m_na_row_indices.clear();
        return;
    }

    // Backup originals
    m_data_backup = m_data;
    m_timestamps_backup = m_timestamps;

    const Eigen::Index R = m_data.rows();
    const Eigen::Index C = m_data.cols();

    // Build a boolean mask of rows to keep
    std::vector<char> keep(R, 1);
    for (size_t i : naIdx) {
        if (i < static_cast<size_t>(R)) keep[static_cast<size_t>(i)] = 0;
    }

    // Count kept rows
    size_t kept_count = 0;
    for (Eigen::Index r = 0; r < R; ++r) if (keep[static_cast<size_t>(r)]) ++kept_count;

    // Create new matrix of kept rows
    Eigen::MatrixXd newmat(static_cast<Eigen::Index>(kept_count), C);
    std::vector<std::string> newtimes;
    newtimes.reserve(kept_count);

    Eigen::Index rr = 0;
    for (Eigen::Index r = 0; r < R; ++r) {
        if (keep[static_cast<size_t>(r)]) {
            newmat.row(rr) = m_data.row(r);
            newtimes.push_back(m_timestamps[static_cast<size_t>(r)]);
            ++rr;
        }
    }

    // Replace m_data and m_timestamps with filtered versions
    m_data = std::move(newmat);
    m_timestamps = std::move(newtimes);

    // Store removed indices (sorted ascending)
    m_na_row_indices = std::move(naIdx);
    std::sort(m_na_row_indices.begin(), m_na_row_indices.end());

    m_has_filtered_rows = true;
}

/**
 * Restore the original (unfiltered) data/timestamps and clear backups
 */
void Data::restoreOriginalData() {
    if (!m_has_filtered_rows) return;
    m_data = std::move(m_data_backup);
    m_timestamps = std::move(m_timestamps_backup);
    m_data_backup.resize(0,0);
    m_timestamps_backup.clear();
    m_na_row_indices.clear();
    m_has_filtered_rows = false;
}

/**
 * Expand predictions on the filtered dataset back to full-length matrix
 */
Eigen::MatrixXd Data::expandPredictionsToFull(const Eigen::MatrixXd& preds) const {
    if (!m_has_filtered_rows) {
        return preds;
    }

    const Eigen::Index origRows = static_cast<Eigen::Index>(m_data_backup.rows());
    const Eigen::Index validRows = static_cast<Eigen::Index>(m_data.rows());
    if (preds.rows() != validRows) {
        throw std::invalid_argument("expandPredictionsToFull (matrix): preds rows != valid rows");
    }
    const Eigen::Index cols = preds.cols();

    Eigen::MatrixXd full(origRows, cols);
    full.setConstant(std::numeric_limits<double>::quiet_NaN());

    std::size_t idxRemPos = 0;
    Eigen::Index pj = 0;
    for (Eigen::Index ri = 0; ri < origRows; ++ri) {
        bool removed = false;
        if (idxRemPos < m_na_row_indices.size() && static_cast<size_t>(ri) == m_na_row_indices[idxRemPos]) {
            removed = true;
            ++idxRemPos;
        }
        if (!removed) {
            full.row(ri) = preds.row(pj++);
        } // else leave NaNs
    }
    return full;
}

/**
 * Expand predictions produced from calibration matrix
 */
Eigen::MatrixXd Data::expandPredictionsFromCalib(const Eigen::MatrixXd& preds, int inpRows) const {
    if (inpRows < 0) throw std::invalid_argument("expandPredictionsFromCalib: inpRows must be >= 0");
    const Eigen::Index CR = preds.rows();
    const Eigen::Index out_horizon = preds.cols();

    // original full row count (before any filtering)
    const Eigen::Index origRows = static_cast<Eigen::Index>(m_data_backup.rows());
    const Eigen::Index validRows = static_cast<Eigen::Index>(m_data.rows()); // filtered rows

    if (CR == 0 || out_horizon == 0) {
        return Eigen::MatrixXd(0,0);
    }

    // Build mapping filtered_index -> original_index
    std::vector<Eigen::Index> filt2orig;
    filt2orig.reserve(static_cast<size_t>(validRows));
    if (m_na_row_indices.empty()) {
        // identity mapping when no rows were removed
        for (Eigen::Index i = 0; i < validRows; ++i) filt2orig.push_back(i);
    } else {
        // iterate original indices and skip removed ones
        std::size_t remPos = 0;
        for (Eigen::Index orig = 0; orig < origRows; ++orig) {
            if (remPos < m_na_row_indices.size() && static_cast<Eigen::Index>(m_na_row_indices[remPos]) == orig) {
                ++remPos; // this original row was removed
            } else {
                filt2orig.push_back(orig);
            }
        }
        if (static_cast<Eigen::Index>(filt2orig.size()) != validRows) {
            throw std::runtime_error("expandPredictionsFromCalib: internal mapping size mismatch");
        }
    }

    // Prepare full matrix with NaN
    Eigen::MatrixXd full(origRows, out_horizon);
    full.setConstant(std::numeric_limits<double>::quiet_NaN());

    // For each calibration pattern i, place preds(i, j) at filtered-row index (i + inpRows + j)
    for (Eigen::Index i = 0; i < CR; ++i) {
        for (Eigen::Index j = 0; j < out_horizon; ++j) {
            Eigen::Index filteredRow = i + inpRows + j;
            if (filteredRow < 0 || filteredRow >= validRows) {
                // If the pattern maps outside valid rows, skip (safe guard)
                continue;
            }
            Eigen::Index origRow = filt2orig[static_cast<size_t>(filteredRow)];
            full(origRow, j) = preds(i, j);
        }
    }

    return full;
}

/**
 * Make cal + val matrices for current fold in k-fold 
 */
std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> 
Data::makeKFoldMats(
    std::vector<int> inpNumsOfVars,
    int outRows,
    int kFolds,
    int foldIdx,
    bool shuffle,
    bool largerPieceCalib,
    unsigned seed)
{
    if (kFolds < 2) throw std::invalid_argument("kFolds must be >= 2");
    if (foldIdx < 0 || foldIdx >= kFolds) throw std::invalid_argument("Invalid fold index");

    const auto maxR = *std::max_element(inpNumsOfVars.begin(), inpNumsOfVars.end());
    if (maxR <= 0) throw std::runtime_error("At least one value in inpNumsOfVars must be > 0");

    const size_t DC = m_data.cols();
    if (DC < 1) throw std::runtime_error("Data has no columns");
    if (inpNumsOfVars.size() != DC) throw std::invalid_argument("inpNumsOfVars size does not match data columns");

    const int CRcand = static_cast<int>(m_data.rows()) - maxR - outRows + 1;
    if (CRcand <= 0) throw std::runtime_error("Not enough rows to build calibration matrices");

    const int inpC = static_cast<int>(std::accumulate(inpNumsOfVars.begin(), inpNumsOfVars.end(), 0));
    std::vector<std::vector<double>> rowsIn;
    std::vector<std::vector<double>> rowsOut;
    rowsIn.reserve(static_cast<size_t>(CRcand));
    rowsOut.reserve(static_cast<size_t>(CRcand));

    for (int i = 0; i < CRcand; ++i) {
        bool ok = true;
        std::vector<double> inrow;
        inrow.reserve(inpC);

        for (size_t j = 0; j < DC; ++j) {
            for (int l = 0; l < inpNumsOfVars[j]; ++l) {
                int rindex = i + maxR - inpNumsOfVars[j] + l;
                double v = m_data(rindex, static_cast<Eigen::Index>(j));
                if (!std::isfinite(v)) { ok = false; break; }
                inrow.push_back(v);
            }
            if (!ok) break;
        }
        if (!ok) continue;

        std::vector<double> outrow;
        outrow.reserve(outRows);
        for (int j = 0; j < outRows; ++j) {
            int rindex = i + maxR + j;
            double v = m_data(rindex, static_cast<Eigen::Index>(DC - 1));
            if (!std::isfinite(v)) { ok = false; break; }
            outrow.push_back(v);
        }
        if (!ok) continue;

        rowsIn.push_back(std::move(inrow));
        rowsOut.push_back(std::move(outrow));
    }

    const int N = static_cast<int>(rowsIn.size());
    if (N == 0) throw std::runtime_error("No valid calibration patterns");

    Eigen::MatrixXd allInps(N, inpC);
    Eigen::MatrixXd allOuts(N, outRows);
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < inpC; ++c) allInps(r, c) = rowsIn[r][c];
        for (int c = 0; c < outRows; ++c) allOuts(r, c) = rowsOut[r][c];
    }

    std::vector<int> allIdx(N);
    std::iota(allIdx.begin(), allIdx.end(), 0);

    int foldSize  = N / kFolds;
    int foldStart = foldIdx * foldSize;
    int foldEnd   = (foldIdx == kFolds - 1) ? N : (foldStart + foldSize);

    std::vector<int> trainIdx;
    std::vector<int> validIdx;

    if(largerPieceCalib){
        for (int i = 0; i < N; ++i) {
            if (i >= foldStart && i < foldEnd) validIdx.push_back(allIdx[i]);
            else trainIdx.push_back(allIdx[i]);
        }
    } else{
        for (int i = 0; i < N; ++i) {
            if (i >= foldStart && i < foldEnd) trainIdx.push_back(allIdx[i]);
            else validIdx.push_back(allIdx[i]);
        }
    }

    if (shuffle) {
        std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
        std::shuffle(trainIdx.begin(), trainIdx.end(), gen);
    }

    Eigen::MatrixXd trainInps(trainIdx.size(), inpC);
    Eigen::MatrixXd trainOuts(trainIdx.size(), outRows);
    Eigen::MatrixXd validInps(validIdx.size(), inpC);
    Eigen::MatrixXd validOuts(validIdx.size(), outRows);

    for (int i = 0; i < (int)trainIdx.size(); ++i) {
        trainInps.row(i) = allInps.row(trainIdx[i]);
        trainOuts.row(i) = allOuts.row(trainIdx[i]);
    }
    for (int i = 0; i < (int)validIdx.size(); ++i) {
        validInps.row(i) = allInps.row(validIdx[i]);
        validOuts.row(i) = allOuts.row(validIdx[i]);
    }

    return { trainInps, trainOuts, validInps, validOuts };
}

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, std::vector<int>, std::vector<int>>
Data::makeMats(
    const std::vector<std::vector<int>>& inpOffsets,
    int outRows,
    double trainFraction,
    bool shuffleCalib,
    unsigned seed
) const
{
    if (outRows <= 0)
        throw std::invalid_argument("outRows must be positive");

    if (trainFraction <= 0.0 || trainFraction >= 1.0)
        throw std::invalid_argument("trainFraction must be in (0,1)");

    const size_t DC = m_data.cols();
    if (DC < 1)
        throw std::runtime_error("Data has no columns");

    if (inpOffsets.size() != DC)
        throw std::invalid_argument("inpOffsets size does not match data columns");

    int inpC = 0;
    for (const auto& v : inpOffsets)
        inpC += static_cast<int>(v.size());

    if (inpC == 0)
        throw std::runtime_error("No input features selected (all inpOffsets empty)");

    int minOff = 0, maxOff = 0;
    for (const auto& v : inpOffsets) {
        for (int o : v) {
            minOff = std::min(minOff, o);
            maxOff = std::max(maxOff, o);
        }
    }

    const int nRows = static_cast<int>(m_data.rows());
    const int lastNeeded = std::max(maxOff, outRows - 1);
    const int CRcand = nRows - lastNeeded + minOff;

    if (CRcand <= 0)
        throw std::runtime_error(
            "Not enough rows to build patterns with given offsets/outRows"
        );

    std::vector<int> globalIdx;
    std::vector<std::vector<double>> rowsIn;
    std::vector<std::vector<double>> rowsOut;

    rowsIn.reserve(static_cast<size_t>(CRcand));
    rowsOut.reserve(static_cast<size_t>(CRcand));

    for (int i = 0; i < CRcand; ++i) {
        bool ok = true;

        // --- vstupy ---
        std::vector<double> inrow;
        inrow.reserve(inpC);

        for (size_t j = 0; j < DC; ++j) {
            for (int off : inpOffsets[j]) {
                int rindex = i - minOff + off;
                double v = m_data(rindex, static_cast<Eigen::Index>(j));
                if (!std::isfinite(v)) {
                    ok = false;
                    break;
                }
                inrow.push_back(v);
            }
            if (!ok) break;
        }
        if (!ok) continue;

        std::vector<double> outrow;
        outrow.reserve(outRows);

        for (int j = 0; j < outRows; ++j) {
            int rindex = i - minOff + j;
            double v = m_data(rindex, static_cast<Eigen::Index>(DC - 1));
            if (!std::isfinite(v)) {
                ok = false;
                break;
            }
            outrow.push_back(v);
        }
        if (!ok) continue;

        rowsIn.push_back(std::move(inrow));
        rowsOut.push_back(std::move(outrow));
        globalIdx.push_back(i);
    }

    const int total = static_cast<int>(rowsIn.size());
    if (total == 0)
        throw std::runtime_error("No valid patterns after filtering NaNs");

    int nTrain = static_cast<int>(std::floor(trainFraction * total));
    nTrain = std::max(1, std::min(nTrain, total - 1));

    std::vector<int> idx(total);
    std::iota(idx.begin(), idx.end(), 0);

    if (shuffleCalib && nTrain > 1) {
        std::mt19937 gen(seed == 0 ? std::random_device{}() : seed);
        std::shuffle(idx.begin(), idx.begin() + nTrain, gen);
    }

    Eigen::MatrixXd trainIn(nTrain, inpC);
    Eigen::MatrixXd trainOut(nTrain, outRows);
    Eigen::MatrixXd validIn(total - nTrain, inpC);
    Eigen::MatrixXd validOut(total - nTrain, outRows);

    for (int i = 0; i < nTrain; ++i) {
        int ri = idx[i];
        for (int c = 0; c < inpC; ++c)
            trainIn(i, c) = rowsIn[ri][c];
        for (int c = 0; c < outRows; ++c)
            trainOut(i, c) = rowsOut[ri][c];
    }

    for (int i = nTrain; i < total; ++i) {
        int ri = idx[i];
        int vi = i - nTrain;
        for (int c = 0; c < inpC; ++c)
            validIn(vi, c) = rowsIn[ri][c];
        for (int c = 0; c < outRows; ++c)
            validOut(vi, c) = rowsOut[ri][c];
    }

    std::vector<int> calIdx(idx.begin(), idx.begin() + nTrain);

    return {trainIn, trainOut, validIn, validOut, globalIdx, calIdx};
}

bool Data::saveVector(const std::vector<int>& v, const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) {
        std::cerr << "[Data::saveVector] Cannot open file: " << path << "\n";
        return false;
    }
    for (size_t i = 0; i < v.size(); ++i) {
        ofs << v[i];
        ofs << "\n";
    }
    ofs.close();
    return true;
}