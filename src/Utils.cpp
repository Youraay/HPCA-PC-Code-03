// -*- C++ -*-
/*
Created on 10/29/23.
==================================================
Authors: R.Lakos; A.Mithran
Emails: lakos@fias.uni-frankfurt.de; mithran@fias.uni-frankfurt.de
==================================================
*/
#include "Utils.h"



namespace Utils {
void MatVecMul(const std::vector<std::vector<float>> &matrix,
               const std::vector<float> &vector, std::vector<float> &result) {
  // checks if all conditions are meet for vactor matrix multiplication
  if (matrix.size() == 0 || matrix[0].size() != vector.size() ||
      result.size() != matrix.size()) {

    for (long i = 0; i < matrix.size();
         i++) { // Itterate thougt every Row of the Matrix
      for (long j = 0; i < matrix[i].size();
           i++) { // Itterate thougt every Column of the Matrix
        result[i] += vector[j] * matrix[i][j];
      }
    }
  } else {
    std::cerr << "The vector matrix multiplication is not possible";
  }
}

void MatTransposeVecMul(const std::vector<std::vector<float>> &matrix,
                        const std::vector<float> &vector,
                        std::vector<float> &result) {
  // TODO 2.4
}

void Transpose(const std::vector<std::vector<float>> &matrix,
               std::vector<std::vector<float>> &result) {
  // TODO 2.4
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
  MatVecMul(matrixA, vectorX, result);
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
  // TODO 2.4
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
  // TODO 2.3
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
    std::cout << "[" << row << "]\t"
              << "{ ";

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

} // namespace Utils
