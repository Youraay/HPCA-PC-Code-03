// -*- C++ -*-
/*
Created on 10/29/23.
==================================================
Authors: R.Lakos; A.Mithran
Emails: lakos@fias.uni-frankfurt.de; mithran@fias.uni-frankfurt.de
==================================================
*/

#include <iostream>
#include <cmath>

#include "MLPHandler.h"


int main(int argc, char* argv[])
{
  Utils::CompareRuntimes();
  std::string filePath;

  if (argc > 1) {
    filePath = argv[1];
    std::cout << "File path provided: " << filePath << std::endl;

  } else {
    std::cout << "Usage: " << argv[0] << " <FILEPATH>" << std::endl;
    std::cout << "Example: " << argv[0] << " \"/home/username/Downloads\"" << std::endl;
    std::cout << "[Use the directory as path, were training- and test-file are located]" << std::endl;
    exit(1);
  }

  // topology: given as size of each layer (for MNIST, first layer size has to be 784, last layer size has to be 10)
  // activation: given as string, possible values: "None", "TanH", "LeakyReLU", "Softmax"
  std::vector<size_t> topology = {784, 800, 10};
  std::vector<std::string> activations = {"None", "LeakyReLU", "Softmax"};

  MLPHandler mlp(topology,             // topology
                 activations,         // activation order
                 60000,    // nTrainingSamples
                 1000,     // nTestingSamples
                 10,              // nEpochs
                 10);           // batchSize

  mlp.ReadMNISTFiles(filePath);
  mlp.StartTraining();

//   std::vector<size_t> topology2 = {784, 10};
//   std::vector<std::string> activations2 = {"None", "Softmax"};
//   MLPHandler mlp2(topology2, activations2, 60000, 1000, 10, 10);
//   mlp2.ReadMNISTFiles(filePath);
//   mlp2.StartTraining();

  /* std::vector<size_t> topology3 = {784, 800, 10};
  std::vector<std::string> activations3 = {"None", "TanH", "Softmax"}; */
  
  /* MLPHandler mlp3(topology3, activations3, 60000, 1000, 10, 10);

  mlp3.ReadMNISTFiles(filePath);
  mlp3.StartTraining(); */

  return 0;
}
