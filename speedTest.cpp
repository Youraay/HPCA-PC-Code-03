// -*- C++ -*-
/*
Created on 10/29/23.
==================================================
Authors: R.Lakos; A.Mithran
Emails: lakos@fias.uni-frankfurt.de; mithran@fias.uni-frankfurt.de
==================================================
*/

#include <iostream>

#include "Utils.h"

int main(int argc, char *argv[]) {
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

  std::vector<float> vector1(rows, 2.0);
  std::vector<float> vector2(rows, 4.0);
  std::vector<float> result1(rows, .0);
  std::vector<float> result2(rows, .0);

  // Start the clock before MatVecMul and Transpose
  auto start1 = std::chrono::high_resolution_clock::now();
  Utils::AffineTransform(matrix, vector1, vector2, result1);
  auto end1 = std::chrono::high_resolution_clock::now();
  auto duration1 =
      std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1)
          .count();

  std::cout << "AffineTransform takes " << duration1 << " microseconds"
            << std::endl;
}
