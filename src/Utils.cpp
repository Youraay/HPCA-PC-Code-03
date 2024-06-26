// -*- C++ -*-
/*
Created on 10/29/23.
==================================================
Authors: R.Lakos; A.Mithran
Emails: lakos@fias.uni-frankfurt.de; mithran@fias.uni-frankfurt.de
==================================================
*/

#include "Utils.h"
#include <algorithm>
#include <chrono>
#include <immintrin.h>
#include <random>
#include <stdexcept>
#include <vector>

namespace Utils {
void MatVecMul(const std::vector<std::vector<float>> &matrix,
               const std::vector<float> &vector, std::vector<float> &result) {
  // checks if all conditions are meet for vactor matrix multiplication
  if (matrix.size() > 0 || matrix[0].size() == vector.size() ||
      result.size() == matrix.size()) {

    for (long i = 0; i < matrix.size();
         i++) { // Itterate thougt every Row of the Matrix
      for (long j = 0; j < matrix[i].size();
           j++) { // Itterate thougt every Column of the Matrix
        result[i] += vector[j] * matrix[i][j];
      }
    }
  } else {
    std::cerr << "The vector matrix multiplication is not possible";
  }
}

void MatVecMulSimd(const std::vector<std::vector<float>> &matrix,
                   const std::vector<float> &vector,
                   std::vector<float> &result) {

  // Calculate the number of rows and columns in the matrix
  size_t rows = matrix.size();
  size_t cols = matrix[0].size();

  // Initialize the result vector with zeros
  for (size_t i = 0; i < result.size(); i++) {
    result[i] = 0.0;

    // Use SIMD instructions to accumulate results in chunks of 8 floats
    size_t j = 0;
    __m256 sum = _mm256_setzero_ps();
    for (; j +7 < cols; j += 8) {
      __m256 a = _mm256_loadu_ps(&vector[j]);
      __m256 b = _mm256_loadu_ps(&matrix[i][j]);
      __m256 c = _mm256_mul_ps(a, b);
      sum = _mm256_add_ps(c, sum);
    }
    //Store the accumulated sum in the result vector
    float temp[8];
    _mm256_storeu_ps(temp, sum);

    // Accumulate the remaining elements
    for (int k = 0; k < 8; k++) {
      result[i] += temp[k];
    }

    // Accumulate the remaining elements
    for (; j < cols; j++) {
      result[i] += vector[j] * matrix[i][j];
    }
  }
}

void MatTransposeVecMul(const std::vector<std::vector<float>> &matrix,
                        const std::vector<float> &vector,
                        std::vector<float> &result) {
  if (matrix.size() > 0 && vector.size() > 0) {
    for (size_t i = 0; i < matrix[0].size(); ++i) {
      for (size_t j = 0; j < matrix.size(); ++j) {
        result[i] += vector[j] * matrix[j][i];
      }
    }
  } else {
    throw std::runtime_error(
        "Matrix or Vector is empty in MatTransposeVecMul()."); // or just return
                                                               // matrix
  }
}

void Transpose(const std::vector<std::vector<float>> &matrix,
               std::vector<std::vector<float>> &result) {
  // Get the dimensions of the original matrix
  size_t rows = matrix.size();
  size_t cols = matrix[0].size();

  // Resize the result matrix to match the transposed dimensions
  result.resize(cols);
  for (size_t i = 0; i < cols; ++i) {
    result[i].resize(rows);
  }

  if (matrix.size() > 0) {
    for (size_t i = 0; i < rows; ++i) {
      for (size_t j = 0; j < matrix[i].size(); ++j) {
        result[j][i] = matrix[i][j];
      }
    }
  } else {
    throw std::runtime_error(
        "Matrix is empty in Transpose()."); // or just return matrix
  }
}

void VecAdd(std::vector<float> &vectorA, std::vector<float> &vectorB,
            std::vector<float> &result) {
  if (vectorA.size() == vectorB.size()) {
    for (int i = 1; i < vectorA.size(); i++)
      result[i] = vectorA[i] + vectorB[i];
  } else {
    std::cerr << "the vector vector addition is not possible due to diffrent "
                 "vector sizes";
  }
}

void VecSub(std::vector<float> &vectorA, std::vector<float> &vectorB,
            std::vector<float> &result) {
  size_t size = vectorA.size();

  for (size_t idx = 0; idx < size; idx++) {
    result[idx] = vectorA[idx] - vectorB[idx];
  }
}

void VecSca(std::vector<float> &vector, float scalar,
            std::vector<float> &result) {
  size_t size = vector.size();

  for (size_t idx = 0; idx < size; idx++) {
    result[idx] = vector[idx] * scalar;
  }
}

void AffineTransform(const std::vector<std::vector<float>> &matrixA,
                     std::vector<float> &vectorX, std::vector<float> &vectorB,
                     std::vector<float> &result) {
  MatVecMulSimd(matrixA, vectorX, result);
  // MatVecMul(matrixA, vectorX, result);
  VecAdd(result, vectorB, result);
}

void OuterProduct(const std::vector<float> &a, const std::vector<float> &b,
                  std::vector<std::vector<float>> result) {
  for (size_t i = 0; i < a.size(); ++i) {
    for (size_t j = 0; j < b.size(); ++j) {
      result[i][j] = a[i] * b[j];
    }
  }
}

void OuterProductAdd(const std::vector<float> &a, const std::vector<float> &b,
                     std::vector<std::vector<float>> &result) {
  for (size_t i = 0; i < a.size(); ++i) {
    for (size_t j = 0; j < b.size(); ++j) {
      result[i][j] += a[i] * b[j];
    }
  }
}

void HadamardProduct(const std::vector<float> &vectorA,
                     const std::vector<float> &vectorB,
                     std::vector<float> &result) {
  if (vectorA.size() == vectorB.size()) {
    for (size_t i; i < vectorA.size();
         ++i) { // vectorB has the same size per definition.
      result[i] = vectorA[i] * vectorB[i];
    }
  } else {
    throw std::runtime_error(
        "Both vectors need to be of equal size in HadamardProduct.");
  }
}

void FillRandomly(std::vector<float> &vector, float lowerBound,
                  float upperBound) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(lowerBound, upperBound);
  for (float &element : vector) {
    element = dis(gen);
  }
}

void FillRandomly(std::vector<std::vector<float>> &matrix, float lowerBound,
                  float upperBound) {
  for (std::vector<float> &row : matrix) {
    FillRandomly(row, lowerBound, upperBound);
  }
}

void FillRandomlyPyTorch(std::vector<float> &vector, size_t nInputFeatures) {
  float k = sqrtf(1.f / float(nInputFeatures));
  std::random_device rand_dev;
  std::mt19937 generator(rand_dev());
  std::uniform_real_distribution<float> dist(-k, k);

  for (float &element : vector) {
    element = dist(generator);
  }
}

void FillRandomlyPyTorch(std::vector<std::vector<float>> &matrix,
                         size_t nInputFeatures) {
  for (std::vector<float> &row : matrix) {
    FillRandomlyPyTorch(row, nInputFeatures);
  }
}

void Shuffle(std::vector<std::vector<float>> &inputFeatures,
             std::vector<size_t> &labels) {
  // Check if the sizes of inputFeatures and labels are the same
  if (inputFeatures.size() != labels.size()) {
    throw std::invalid_argument(
        "The sizes of inputFeatures and labels must be the same.");
  }

  // Create a random device, which generates a true random number.
  std::random_device rd;

  // Seed the engine using rd, which will generate a sequence of pseudo-random
  // numbers.
  std::default_random_engine engine(rd());

  // Create a vector of indices
  std::vector<size_t> indices(inputFeatures.size());

  // Fill indices vector with consecutive numbers, starting from 0.
  std::iota(indices.begin(), indices.end(), 0);

  // Shuffle the indices
  std::shuffle(indices.begin(), indices.end(), engine);

  // Create temporary vectors to hold the shuffled features and labels
  std::vector<std::vector<float>> shuffledFeatures(inputFeatures.size());
  std::vector<size_t> shuffledLabels(labels.size());

  // Fill the temporary vectors with the shuffled data
  for (size_t i = 0; i < indices.size(); ++i) {
    shuffledFeatures[i] = inputFeatures[indices[i]];
    shuffledLabels[i] = labels[indices[i]];
  }
  // Swap the shuffled data with the original data
  inputFeatures.swap(shuffledFeatures);
  labels.swap(shuffledLabels);
}

void Zeros(std::vector<float> &vector) {
  std::fill(vector.begin(), vector.end(), 0.f);
}

void Zeros(std::vector<std::vector<float>> &matrix) {
  for (std::vector<float> &row : matrix) {
    Zeros(row);
  }
}

void Print(std::vector<std::vector<float>> &matrix) {
  size_t rows = matrix.size();
  size_t cols = matrix[0].size();

  std::cout << "Matrix: " << rows << " x " << cols << std::endl;

  std::cout << "{ ";
  for (size_t row = 0; row < rows; row++) {
    std::cout << "[" << row << "]\t" << "{ ";

    for (size_t col = 0; col < cols; col++) {
      std::cout << matrix[row][col];

      if (col != cols - 1)
        std::cout << ", ";
    }

    std::cout << " }" << std::endl;
  }

  std::cout << " }" << std::endl;
}

void Print(std::vector<float> &vector) {
  size_t cols = vector.size();

  std::cout << "Vector: " << cols << " x 1" << std::endl;

  std::cout << "{ ";
  for (size_t col = 0; col < cols; col++) {
    std::cout << std::fixed << std::setprecision(4) << vector[col];

    if (col != cols - 1)
      std::cout << ", ";
  }

  std::cout << " }" << std::endl;
}

void CompareRuntimes() {
  // Initialize your matrix, vector, and result here
  const int rows = 100; // Set the desired number of rows
  const int cols = 50;  // Set the desired number of columns
  std::vector<std::vector<float>> matrix;
  matrix.resize(rows);
  for (int i = 0; i < rows; ++i) {
    matrix[i].resize(cols);
    for (int j = 0; j < cols; ++j) {
      matrix[i][j] = static_cast<float>(rand()) / RAND_MAX;
    }
  }

  std::vector<std::vector<float>> transponedMatrix;

  std::vector<float> vector(rows, 2.0);
  std::vector<float> result1(rows, .0);
  std::vector<float> result2(rows, .0);

  // Start the clock before MatVecMul and Transpose
  auto start1 = std::chrono::high_resolution_clock::now();
  Transpose(matrix, transponedMatrix);
  MatVecMul(transponedMatrix, vector, result1);
  auto end1 = std::chrono::high_resolution_clock::now();

  // Start the clock before MatTransposeVecMul
  auto start2 = std::chrono::high_resolution_clock::now();
  MatTransposeVecMul(matrix, vector, result2);
  auto end2 = std::chrono::high_resolution_clock::now();

  // Calculate and print the time differences
  auto duration1 =
      std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1)
          .count();
  auto duration2 =
      std::chrono::duration_cast<std::chrono::microseconds>(end2 - start2)
          .count();

  std::cout << "Time taken by MatVecMul and Transpose: " << duration1
            << " microseconds" << std::endl;
  std::cout << "Time taken by MatTransposeVecMul: " << duration2
            << " microseconds" << std::endl;
  // Print(result1);
  // Print(result2);
  if (result1 == result2)
    std::cout << "Results are equal!\n";
}

} // namespace Utils
