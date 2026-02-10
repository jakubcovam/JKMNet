#ifndef DATA_HPP
#define DATA_HPP

#pragma once
#include <string>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include <random>
#include <limits> 
#include <ostream> 

#include "eigen-3.4/Eigen/Dense"

enum class transform_type { 
    NONE = 0,
    MINMAX, 
    NONLINEAR,
    ZSCORE
};

struct Scaler {
    Eigen::VectorXd min;  
    Eigen::VectorXd max; 
    bool fitted = false;
};

struct RunConfig;

class Data {
    public:
        Data() = default;  //!< The constructor
        ~Data() = default;  //!< The destructor 
        Data(const Data&) = default;  //!< The copy constructor
        Data& operator=(const Data&) = default;   //!< The assignment operator

        size_t loadFilteredCSV(const std::string& path,
            const std::unordered_set<std::string>& idFilter,
            const std::vector<std::string>& keepColumns,  // names of numeric columns to extract (e.g. "T1","T2","T3","moisture")
            const std::string& timestampCol = "hour_start",  // name of timestamp column (e.g "hour_start", "date")
            const std::string& idCol = "ID");  // ID of the selected sensor
            //!< Returns number of loaded rows
        
        static void cleanDirectory(const std::string &path);  //!< Clean all files in a directory 
        static void cleanAllOutputs(const std::string &outDir);  //!< Clean all files in a otputs directory 

        void logRunSettings(std::ostream& os, const RunConfig& cfg, unsigned run_id) const;  //!< Write model settings into a log file

        bool saveMatrixCsv(const std::string &path,
            const Eigen::MatrixXd &M,
            const std::vector<std::string> &colNames = {},
            bool inverseOutputs = false) const;

        std::vector<std::string> timestamps() const;  //!< Getter for the timestamps
        Eigen::MatrixXd numericData() const;  //!< Getter for the data numeric matrix
        void setNumericData(const Eigen::MatrixXd &newData); //!< Setter for the data numeric matrix
        std::vector<std::string> numericColNames() const;  //!< Getter for the names of numeric columns

        void printHeader(const std::string& timestampColName = "timestamp") const;  //!< Print header line, i.e. timestamp + numeric column names
        std::vector<double> getColumnValues(const std::string& name) const; //!< Return a copy of the values in a selected column by name

        void setTransform(transform_type t, double alpha = 0.015, bool excludeLastCol = false);  //!< Set which transform to apply (applies to all numeric columns)
        void applyTransform();  //!< Apply the previously configured transform to m_data
        void inverseTransform();  //!< Inverse the global transform (to bring predictions back)
        Eigen::MatrixXd inverseTransformOutputs(const Eigen::MatrixXd& M) const;  //!< Inverse the global transform for outputs
        
        std::vector<size_t> calibPatternOriginalIndices() const { return m_calib_pattern_orig_indices; }
        std::vector<int> calibPatternFilteredIndices() const { return m_calib_pattern_filtered_indices; }
        
        void makeCalibMat(std::vector<int> inpNumsOfVars, int outRows); //!< Create calibration matrix (both inps + outs) for backpropagation from data matrix (with NA removal)  
        void makeCalibMatsSplit(std::vector<int> inpNumsOfVars, int outRows); //!< Create separate calibration inps and outs matrices for backpropagation from data matrix (with NA removal)
        void splitCalibMat(int inpLength);  //!< Split created calibration matrix into separate inps and outs matrices 
    
        std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
        splitInputsOutputs(const Eigen::MatrixXd &mat, int inpSize, int outSize) const;  //!< Split calib and valid dataset

        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, std::vector<int>, std::vector<int>>
        splitCalibMatWithIdx(double trainFraction = 0.8, bool shuffle = true, unsigned seed = 0) const;  //!< Split calibration matrix into train/validation and also return indices

        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, std::vector<int>, std::vector<int>>
        splitDataRowsWithIdx(double trainFraction = 0.8, bool shuffle = true, unsigned seed = 0) const;  //!< Split raw data rows (m_data) into train/validation and return indices

        Eigen::MatrixXd getCalibMat();  //!< Getter for calibration matrix
        void setCalibMat(const Eigen::MatrixXd &newMat);  //!< Setter for calibration matrix
        Eigen::MatrixXd getCalibInpsMat();  //!< Getter for calibration inputs matrix
        void setCalibInpsMat(const Eigen::MatrixXd &newMat);  //!< Setter for calibration inputs matrix
        Eigen::MatrixXd getCalibOutsMat();  //!< Getter for calibration outputs matrix
        void setCalibOutsMat(const Eigen::MatrixXd &newMat);  //!< Setter for calibration outputs matrix
        
        std::vector<int> permutationVector(int length);  //!< Create random permutation vector for shuffling
        Eigen::MatrixXd shuffleMatrix(const Eigen::MatrixXd &matrix, const std::vector<int>& permVec); //!< Shuffle matrix rows
        Eigen::MatrixXd unshuffleMatrix(const Eigen::MatrixXd &matrix, const std::vector<int>& permVec); //!< Unshuffle matrix rows

        // Deal with NAs in the dataset
        std::vector<size_t> findRowsWithNa() const;  //!< Find indices of rows that contain any NaN in numeric data
        void removeRowsWithNa();  //!< Remove rows that contain any NaN from m_data and m_timestamps, but keep backups and record removed indices so they can be restored later
        void restoreOriginalData();  //< Restore the original (unfiltered) data/timestamps and clear backups
        Eigen::MatrixXd expandPredictionsToFull(const Eigen::MatrixXd& preds) const;  //!< Expand predictions on the filtered dataset back to full-length matrix
        Eigen::MatrixXd expandPredictionsFromCalib(const Eigen::MatrixXd& preds, int inpRows) const;  //!< Expand predictions produced from calibration matrix
        const std::vector<size_t>& removedRowIndices() const { return m_na_row_indices; }   //!< Get indices of rows removed 
        size_t validRowCount() const { return static_cast<size_t>(m_data.rows()); }  //!< Number of valid rows currently in m_data

        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> //!< Make cal + val matrices for current fold in k-fold 
        makeKFoldMats(
            std::vector<int> inpNumsOfVars,
            int outRows,
            int kFolds,
            int foldIdx,
            bool shuffle,
            bool largerPieceCalib,
            unsigned seed);

        std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, std::vector<int>, std::vector<int>>
        makeMats(const std::vector<std::vector<int>>& inpOffsets,
                          int outRows,
                          double trainFraction,
                          bool shuffleCalib,
                          unsigned seed) const;
        bool saveVector(const std::vector<int>& v, const std::string& path);

    protected:

    private:
        static void splitCSVLine(const std::string& line, std::vector<std::string>& outFields);  //!< Helper function to parse a CSV line into fields (handles quotes)

        std::vector<std::string> m_timestamps;  //!< Time string
        Eigen::MatrixXd m_data;  //!< Matrix with numeric columns (rows x cols) filled with variables
        std::vector<std::string> m_colNames;  //!< Column names for m_data
        Eigen::MatrixXd calibMat; //!< Matrix of inputs and desired outputs for backpropagation
        Eigen::MatrixXd calibInpsMat; //!< Matrix of inputs for backpropagation
        Eigen::MatrixXd calibOutsMat; //!< Matrix of desired outputs for backpropagation

        std::vector<int> m_calib_pattern_filtered_indices;   //!< Indices in filtered m_data for each calib pattern (reference output row)
        std::vector<size_t> m_calib_pattern_orig_indices;    //!< Indices in original unfiltered data (if available)

        // Global transform config
        transform_type m_transform = transform_type::NONE;
        double m_alpha = 0.015;
        bool m_excludeLastCol = false;
        Scaler m_scaler;   //!< Stores per-column min/max after a MINMAX fit

        // Deal with NAs in the dataset
        Eigen::MatrixXd m_data_backup;  //!< Full original data backup (before filtering)
        std::vector<std::string> m_timestamps_backup;  //!< Timestamps backup
        std::vector<size_t> m_na_row_indices;  //!< Indices of removed rows (in original coordinates)
        bool m_has_filtered_rows = false;  //!< True if removeRowsWithNa() was applied

};

#endif // DATA_HPP