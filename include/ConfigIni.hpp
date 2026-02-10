
#ifndef CONFIG_HPP
#define CONFIG_HPP

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cctype>
#include <stdexcept>

#include "Data.hpp"   
#include "MLP.hpp"    

/**
 * Helper functions
 */
static inline std::string trimStr(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b-1])) --b;
    return s.substr(a, b-a);
}

// static std::string join(const std::vector<std::string>& v, const std::string& sep = ", ") {
//     std::string out;
//     for (size_t i = 0; i < v.size(); ++i) {
//         out += v[i];
//         if (i + 1 < v.size()) out += sep;
//     }
//     return out;
// }

static inline std::vector<std::string> splitCommaList(const std::string &s) {
    std::vector<std::string> out;
    std::string cur;
    std::istringstream iss(s);
    while (std::getline(iss, cur, ',')) {
        std::string t = trimStr(cur);
        if (!t.empty()) out.push_back(t);
    }
    return out;
}

static inline std::vector<std::string>
splitList(const std::string& s, char delim)
{
    std::vector<std::string> out;
    std::string cur;
    std::istringstream iss(s);
    while (std::getline(iss, cur, delim)) {
        out.push_back(trimStr(cur));
    }
    return out;
}

static inline bool parseBool(const std::string &s) {
    std::string x; x.reserve(s.size());
    for (char c : s) x.push_back(static_cast<char>(std::toupper((unsigned char)c)));
    if (x=="1" || x=="TRUE" || x=="YES" || x=="ON") return true;
    if (x=="0" || x=="FALSE" || x=="NO" || x=="OFF") return false;
    throw std::runtime_error("parseBool: cannot parse boolean from '" + s + "'");
}

/**
 * Run the configuration
 */
struct RunConfig {
    // model / training
    std::string trainer = "online";                 // "online" or "batch"
    std::vector<unsigned> mlp_architecture;         // e.g. [8,6,2]
    std::vector<std::vector<int>> input_numbers;                 // e.g. [-2,-1 0; -1,0,1,2 ; ; -2,-1 ]
    std::string activation = "RELU";
    std::string weight_init = "RANDOM";

    int ensemble_runs = 25;
    int max_iterations = 500;
    int max_metrics_step = 99;
    double max_error = 0.002;
    double learning_rate = 0.001;
    bool shuffle = true;
    unsigned seed = 42;
    int batch_size = 30;
    double train_fraction = 0.8;
    bool split_shuffle = true;

    // transforms
    std::string transform = "NONE";
    double transform_alpha = 0.015;
    bool exclude_last_col_from_transform = false;
    bool remove_na_before_calib = true;

    // data selection
    std::string data_file;
    std::string id = "";
    std::string id_col = "ID";
    std::string timestamp = "date";
    std::vector<std::string> columns;

    // optimization
    bool pso_optimize = true;

    // output paths (parsed from [paths] section)
    std::string out_dir = "";
    std::string log_dir = "";
    std::string calib_mat = "";
    std::string weights_csv_init = "";
    std::string weights_bin_init = "";
    std::string weights_vec_csv_init = "";
    std::string weights_vec_bin_init = "";
    std::string weights_csv = "";
    std::string weights_bin = "";
    std::string weights_vec_csv = "";
    std::string weights_vec_bin = "";
    std::string real_calib = "";
    std::string pred_calib = "";
    std::string real_valid = "";
    std::string pred_valid = "";
    std::string metrics_cal = "";
    std::string metrics_val = "";
    std::string run_info = "";
    std::string errors_csv;
    std::string pattern_indices = "";
};

/**
 * Mapping the helpers
 */
inline activ_func_type strToActivation(const std::string &s) {
    std::string u;
    for (char c : s) u.push_back(static_cast<char>(std::toupper((unsigned char)c)));
    if (u == "RELU") return activ_func_type::RELU;
    if (u == "LEAKYRELU") return activ_func_type::LEAKYRELU;
    if (u == "TANH") return activ_func_type::TANH;
    if (u == "SIGMOID") return activ_func_type::SIGMOID;
    if (u == "LINEAR") return activ_func_type::LINEAR;
    if (u == "IABS") return activ_func_type::IABS;
    if (u == "GAUSSIAN") return activ_func_type::GAUSSIAN;
    if (u == "LOGLOG") return activ_func_type::LOGLOG;
    if (u == "CLOGLOG") return activ_func_type::CLOGLOG;
    if (u == "CLOGLOGM") return activ_func_type::CLOGLOGM;
    if (u == "ROOTSIG") return activ_func_type::ROOTSIG;
    if (u == "LOGSIG") return activ_func_type::LOGSIG;
    if (u == "SECH") return activ_func_type::SECH;
    if (u == "WAVE") return activ_func_type::WAVE;

    throw std::runtime_error("Unknown activation: " + s);
}

inline weight_init_type strToWeightInit(const std::string &s) {
    std::string u;
    for (char c : s) u.push_back(static_cast<char>(std::toupper((unsigned char)c)));
    if (u == "RANDOM") return weight_init_type::RANDOM;
    if (u == "LHS") return weight_init_type::LHS;
    if (u == "LHS2") return weight_init_type::LHS2;
    if (u == "HE") return weight_init_type::HE;

    throw std::runtime_error("Unknown weight_init: " + s);
}

inline transform_type strToTransformType(const std::string &s) {
    std::string u;
    for (char c : s) u.push_back(static_cast<char>(std::toupper((unsigned char)c)));
    if (u == "NONE") return transform_type::NONE;
    if (u == "MINMAX") return transform_type::MINMAX;
    if (u == "NONLINEAR") return transform_type::NONLINEAR;
    if (u == "ZSCORE") return transform_type::ZSCORE;

    throw std::runtime_error("Unknown transform: " + s);
}

enum class TrainerType {
    ONLINE,
    BATCH,
    ONLINE_EPOCH,
    BATCH_EPOCH
};

inline TrainerType strToTrainerType(const std::string& s) {
    if (s == "online")       return TrainerType::ONLINE;
    if (s == "batch")        return TrainerType::BATCH;
    if (s == "online_epoch") return TrainerType::ONLINE_EPOCH;
    if (s == "batch_epoch")  return TrainerType::BATCH_EPOCH;
    throw std::invalid_argument("Unknown trainer type: " + s);
}

/**
 * Parse the ini to map
 */
inline std::unordered_map<std::string, std::string> parseIniToMap(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) throw std::runtime_error("Cannot open INI file: " + path);
    std::unordered_map<std::string,std::string> kv;
    std::string line;
    while (std::getline(ifs, line)) {
        std::string s = trimStr(line);
        if (s.empty()) continue;
        if (s[0] == ';' || s[0] == '#') continue;
        if (s.front() == '[' && s.back() == ']') continue; // skip sections
        // find '='
        auto pos = s.find('=');
        if (pos == std::string::npos) continue;
        std::string key = trimStr(s.substr(0, pos));
        std::string val = trimStr(s.substr(pos+1));
        // remove inline comment after value ( ; or # )
        size_t cpos = val.find_first_of(";#");
        if (cpos != std::string::npos) {
            val = trimStr(val.substr(0, cpos));
        }
        // lowercase key for convenience
        std::string key_lower;
        key_lower.reserve(key.size());
        for (char ch : key) key_lower.push_back(static_cast<char>(std::tolower((unsigned char)ch)));
        kv[key_lower] = val;
    }
    return kv;
}

/**
 * Convert string list to numeric lists
 */
inline std::vector<unsigned> parseUnsignedList(const std::string &s) {
    std::vector<unsigned> out;
    for (auto &tok : splitCommaList(s)) {
        try {
            int tmp = std::stoi(tok);
            if (tmp < 0) throw std::runtime_error("negative integer in unsigned list");
            out.push_back(static_cast<unsigned>(tmp));
        } catch (...) {
            throw std::runtime_error("parseUnsignedList: invalid integer '" + tok + "'");
        }
    }
    return out;
}

inline std::vector<int> parseRangeToken(const std::string& tok)
{
    auto parts = splitList(tok, ':');

    try {
        if (parts.size() == 1) {
            return { std::stoi(parts[0]) };
        }
        else if (parts.size() == 2 || parts.size() == 3) {
            int a = std::stoi(parts[0]);
            int b = std::stoi(parts[1]);
            int step = (parts.size() == 3) ? std::stoi(parts[2]) : 1;
            if (step <= 0)
                throw std::runtime_error("step must be positive");

            std::vector<int> out;
            if (a <= b) {
                for (int i = a; i <= b; i += step) out.push_back(i);
            } else {
                for (int i = a; i >= b; i -= step) out.push_back(i);
            }
            return out;
        }
    } catch (...) {
        throw std::runtime_error("Invalid range token '" + tok + "'");
    }

    throw std::runtime_error("Invalid range token '" + tok + "'");
}

inline std::vector<int> parseOffsetExpr(const std::string& s)
{
    std::vector<int> out;

    for (auto& tok : splitList(s, ',')) {
        if (tok.empty()) continue;
        auto v = parseRangeToken(tok);
        out.insert(out.end(), v.begin(), v.end());
    }

    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());

    return out;
}

inline std::vector<std::vector<int>>
parseOffsetLists(const std::string& s)
{
    std::vector<std::vector<int>> out;

    for (auto& block : splitList(s, '|')) {
        if (block.empty())
            out.emplace_back();
        else
            out.push_back(parseOffsetExpr(block));
    }

    return out;
}

inline std::vector<std::string> parseStringList(const std::string &s) {
    return splitCommaList(s);
}

/**
 * Parse config from ini 
 */
inline RunConfig parseConfigIni(const std::string &path) {
    auto kv = parseIniToMap(path);
    RunConfig cfg;

    auto get = [&](const std::string &k)->std::string {
        std::string kl = k;
        for (char &c : kl) c = static_cast<char>(std::tolower((unsigned char)c));
        auto it = kv.find(kl);
        if (it == kv.end()) return std::string{};
        return it->second;
    };

    // required/important
    std::string sdata = get("data_file");
    if (sdata.empty()) throw std::runtime_error("config: data_file is required");
    cfg.data_file = sdata;

    std::string sid = get("id");
    if (!sid.empty()) cfg.id = trimStr(sid);

    std::string sidcol = get("id_col");
    if (!sidcol.empty()) cfg.id_col = trimStr(sidcol);

    std::string stimestamp = get("timestamp");
    if (!stimestamp.empty()) cfg.timestamp = trimStr(stimestamp);

    std::string scols = get("columns");
    if (!scols.empty()) cfg.columns = parseStringList(scols);

    std::string strainer = get("trainer");
    if (!strainer.empty()) cfg.trainer = trimStr(strainer);

    std::string sarch = get("architecture");
    if (!sarch.empty()) cfg.mlp_architecture = parseUnsignedList(sarch);

    std::string sinpnums = get("input_numbers");
    if (!sinpnums.empty()) cfg.input_numbers = parseOffsetLists(sinpnums);

    std::string sact = get("activation");
    if (!sact.empty()) cfg.activation = trimStr(sact);

    std::string swinit = get("weight_init");
    if (!swinit.empty()) cfg.weight_init = trimStr(swinit);

    std::string sensemblerun = get("ensemble_runs");
    if (!sensemblerun.empty()) cfg.ensemble_runs = std::stoi(sensemblerun);

    std::string smaxit = get("max_iterations");
    if (!smaxit.empty()) cfg.max_iterations = std::stoi(smaxit);

    std::string smaxmetricstep = get("max_metrics_step");
    if (!smaxmetricstep.empty()) cfg.max_metrics_step = std::stoi(smaxmetricstep);

    std::string smaxerr = get("max_error");
    if (!smaxerr.empty()) cfg.max_error = std::stod(smaxerr);

    std::string slr = get("learning_rate");
    if (!slr.empty()) cfg.learning_rate = std::stod(slr);

    std::string sshuffle = get("shuffle");
    if (!sshuffle.empty()) cfg.shuffle = parseBool(sshuffle);

    std::string sseed = get("seed");
    if (!sseed.empty()) cfg.seed = static_cast<unsigned>(std::stoul(sseed));

    std::string sbsize = get("batch_size");
    if (!sbsize.empty()) cfg.batch_size = std::stoi(sbsize);

    std::string strainf = get("train_fraction");
    if (!strainf.empty()) cfg.train_fraction = std::stod(strainf);

    std::string ssplitshuffle = get("split_shuffle");
    if (!ssplitshuffle.empty()) cfg.split_shuffle = parseBool(ssplitshuffle);

    std::string strans = get("transform");
    if (!strans.empty()) cfg.transform = trimStr(strans);

    std::string sa = get("transform_alpha");
    if (!sa.empty()) cfg.transform_alpha = std::stod(sa);

    std::string sexc = get("exclude_last_col_from_transform");
    if (!sexc.empty()) cfg.exclude_last_col_from_transform = parseBool(sexc);

    std::string sremna = get("remove_na_before_calib");
    if (!sremna.empty()) cfg.remove_na_before_calib = parseBool(sremna);

    std::string soptim = get("pso_optimize");
    if (!soptim.empty()) cfg.pso_optimize = parseBool(soptim);
    
    // paths (optional)
    std::string sout = get("out_dir");
    if (!sout.empty()) cfg.out_dir = trimStr(sout);
    std::string slogdir = get("log_dir");
    if (!slogdir.empty()) cfg.log_dir = trimStr(slogdir);
    std::string scalib = get("calib_mat");
    if (!scalib.empty()) cfg.calib_mat = trimStr(scalib);
    std::string sinitweil = get("weights_csv_init");
    if (!sinitweil.empty()) cfg.weights_csv_init = trimStr(sinitweil);
    std::string sinitwellbin = get("weights_bin_init");
    if (!sinitwellbin.empty()) cfg.weights_bin_init = trimStr(sinitwellbin);
    std::string sinitweilvec = get("weights_vec_csv_init");
    if (!sinitweilvec.empty()) cfg.weights_vec_csv_init = trimStr(sinitweilvec);
    std::string sinitwellbinvec = get("weights_vec_bin_init");
    if (!sinitwellbinvec.empty()) cfg.weights_vec_bin_init = trimStr(sinitwellbinvec);
    std::string sweil = get("weights_csv");
    if (!sweil.empty()) cfg.weights_csv = trimStr(sweil);
    std::string swellbin = get("weights_bin");
    if (!swellbin.empty()) cfg.weights_bin = trimStr(swellbin);
    std::string sweilvec = get("weights_vec_csv");
    if (!sweilvec.empty()) cfg.weights_vec_csv = trimStr(sweilvec);
    std::string swellbinvec = get("weights_vec_bin");
    if (!swellbinvec.empty()) cfg.weights_vec_bin = trimStr(swellbinvec);
    std::string srealc = get("real_calib");
    if (!srealc.empty()) cfg.real_calib = trimStr(srealc);
    std::string spredc = get("pred_calib");
    if (!spredc.empty()) cfg.pred_calib = trimStr(spredc);
    std::string srealv = get("real_valid");
    if (!srealv.empty()) cfg.real_valid = trimStr(srealv);
    std::string spredv = get("pred_valid");
    if (!spredv.empty()) cfg.pred_valid = trimStr(spredv);
    std::string smetcal = get("metrics_cal");
    if (!smetcal.empty()) cfg.metrics_cal = trimStr(smetcal);
    std::string smetval = get("metrics_val");
    if (!smetval.empty()) cfg.metrics_val = trimStr(smetval);
    std::string sruninfo = get("run_info");
    if (!sruninfo.empty()) cfg.run_info = trimStr(sruninfo);
    std::string serrors = get("errors_csv");
    if (!serrors.empty()) cfg.errors_csv = trimStr(serrors);
    std::string sindices = get("pattern_indices");
    if (!sindices.empty()) cfg.pattern_indices = trimStr(sindices);
    
    // Basic validation
    if (cfg.mlp_architecture.empty()) throw std::runtime_error("config: architecture is required (e.g. architecture = 8,6,2)");
    if (cfg.input_numbers.empty()) throw std::runtime_error("config: input_numbers is required and length should match number of columns");

    // check that input_numbers length matches columns (if columns provided)
    if (!cfg.columns.empty() && cfg.columns.size() != cfg.input_numbers.size()) {
        throw std::runtime_error("config: columns length and input_numbers length must match");
    }

    return cfg;
}


#endif // CONFIG_HPP
