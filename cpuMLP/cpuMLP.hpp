#include <string>
#include <vector>
#include <Eigen/Dense>
#include <cstdlib>
#include "iostream"
#include <fstream>
#include <cmath>
#include "Types.h"

template<int StateDim, int ActionDim>
class MLP_3 {

 public:
  typedef Eigen::Matrix<float, ActionDim, 1> Action;
  typedef Eigen::Matrix<float, StateDim, 1> State;

  MLP_3() {

    layersizes.push_back(StateDim);
    layersizes.reserve(layersizes.size() + hiddensizes.size());
    layersizes.insert(layersizes.end(), hiddensizes.begin(), hiddensizes.end());
    layersizes.push_back(ActionDim);
    ///[input hidden output]

    params.resize(2 * (layersizes.size() - 1));
    Ws.resize(layersizes.size() - 1);
    bs.resize(layersizes.size() - 1);
    lo.resize(layersizes.size());
    Stdev.resize(ActionDim);

    for (int i = 0; i < (int)(layersizes.size()); i++){
        lo[i].resize(layersizes[i], 1);
	lo[i].setZero();
    }

    for (int i = 0; i < (int)(params.size()); i++) {
//      int paramSize = 0;

      if (i % 2 == 0) ///W resize
      {
        Ws[i / 2].resize(layersizes[i / 2 + 1], layersizes[i / 2]);
        params[i].resize(layersizes[i / 2] * layersizes[i / 2 + 1]);
        Ws[i / 2].setZero();
        params[i].setZero();
      }
      if (i % 2 == 1) ///b resize
      {
        bs[(i - 1) / 2].resize(layersizes[(i + 1) / 2]);
        bs[(i - 1) / 2].setZero();
        params[i].resize(layersizes[(i + 1) / 2]);
        params[i].setZero();
      }
    }

  }

  void updateParamFromTxt(std::string fileName) {
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

    std::ifstream indata;
    indata.open(fileName);
    std::string line;
    getline(indata, line);
    std::stringstream lineStream(line);
    std::string cell;

    int totalN = 0;
    ///assign parameters
    for (int i = 0; i < (int)(params.size()); i++) {

      int paramSize = 0;
      while (std::getline(lineStream, cell, ',')) { ///Read param
        params[i](paramSize++) = std::stof(cell);
        if (paramSize == params[i].size()){
	       	break;
        }
      }
      totalN += paramSize;
      if (i % 2 == 0) ///W copy
        memcpy(Ws[i / 2].data(), params[i].data(), sizeof(float) * Ws[i / 2].size());
      if (i % 2 == 1) ///b copy
        memcpy(bs[(i - 1) / 2].data(), params[i].data(), sizeof(float) * bs[(i - 1) / 2].size());
    }

  }

  inline Action forward(State &state) {

    lo[0] = state;
    for (int cnt = 0; cnt < (int)(Ws.size()) - 1; cnt++) {
      lo[cnt + 1] = Ws[cnt] * lo[cnt] + bs[cnt];
      lo[cnt + 1] = lo[cnt + 1].cwiseMax(1e-2*lo[cnt + 1]);
    }

    lo[lo.size() - 1] = Ws[Ws.size() - 1] * lo[lo.size() - 2] + bs[bs.size() - 1]; /// output layer
    if ((lo.back().array() != lo.back().array()).all())
       std::cout<<"state 2 nan"<<std::endl;
    return lo.back();
  }

 private:
  std::vector<Eigen::Matrix<float, -1, 1>> params;
  std::vector<Eigen::Matrix<float, -1, -1>> Ws;
  std::vector<Eigen::Matrix<float, -1, 1>> bs;
  std::vector<Eigen::Matrix<float, -1, 1>> lo;

  Action Stdev;

  std::vector<int> layersizes, hiddensizes = {256, 128, 32};
  bool isTanh = false;
};

/*
 *
 * MLP2 for state estimation
 */

template<int StateDim, int ActionDim>
class MLP_2 {

 public:
  typedef Eigen::Matrix<float, ActionDim, 1> Action;
  typedef Eigen::Matrix<float, StateDim, 1> State;

  MLP_2() {

    layersizes.push_back(StateDim);
    layersizes.reserve(layersizes.size() + hiddensizes.size());
    layersizes.insert(layersizes.end(), hiddensizes.begin(), hiddensizes.end());
    layersizes.push_back(ActionDim);
    ///[input hidden output]

    params.resize(2 * (layersizes.size() - 1));
    Ws.resize(layersizes.size() - 1);
    bs.resize(layersizes.size() - 1);
    lo.resize(layersizes.size());
    Stdev.resize(ActionDim);

    for (int i = 0; i < (int)(layersizes.size()); i++){
        lo[i].resize(layersizes[i], 1);
	lo[i].setZero();
    }

    for (int i = 0; i < (int)(params.size()); i++) {
//      int paramSize = 0;

      if (i % 2 == 0) ///W resize
      {
        Ws[i / 2].resize(layersizes[i / 2 + 1], layersizes[i / 2]);
        params[i].resize(layersizes[i / 2] * layersizes[i / 2 + 1]);
        Ws[i / 2].setZero();
        params[i].setZero();
      }
      if (i % 2 == 1) ///b resize
      {
        bs[(i - 1) / 2].resize(layersizes[(i + 1) / 2]);
        bs[(i - 1) / 2].setZero();
        params[i].resize(layersizes[(i + 1) / 2]);
        params[i].setZero();
      }
    }

  }

  void updateParamFromTxt(std::string fileName) {
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

    std::ifstream indata;
    indata.open(fileName);
    std::string line;
    getline(indata, line);
    std::stringstream lineStream(line);
    std::string cell;

    int totalN = 0;
    ///assign parameters
    for (int i = 0; i < (int)(params.size()); i++) {

      int paramSize = 0;
      while (std::getline(lineStream, cell, ',')) { ///Read param
        params[i](paramSize++) = std::stof(cell);
        if (paramSize == params[i].size()){
	       	break;
        }
      }
      totalN += paramSize;
      if (i % 2 == 0) ///W copy
        memcpy(Ws[i / 2].data(), params[i].data(), sizeof(float) * Ws[i / 2].size());
      if (i % 2 == 1) ///b copy
        memcpy(bs[(i - 1) / 2].data(), params[i].data(), sizeof(float) * bs[(i - 1) / 2].size());
    }

  }

  inline Action forward(State &state) {
    lo[0] = state;
    for (int cnt = 0; cnt < (int)(Ws.size()) - 1; cnt++) {
      lo[cnt + 1] = Ws[cnt] * lo[cnt] + bs[cnt];
      lo[cnt + 1] = lo[cnt + 1].cwiseMax(1e-2*lo[cnt + 1]);
    }

    lo[lo.size() - 1] = Ws[Ws.size() - 1] * lo[lo.size() - 2] + bs[bs.size() - 1]; /// output layer
    if ((lo.back().array() != lo.back().array()).all())
       std::cout<<"state 2 nan"<<std::endl;
    return lo.back();
  }

 private:
  std::vector<Eigen::Matrix<float, -1, 1>> params;
  std::vector<Eigen::Matrix<float, -1, -1>> Ws;
  std::vector<Eigen::Matrix<float, -1, 1>> bs;
  std::vector<Eigen::Matrix<float, -1, 1>> lo;

  Action Stdev;

  std::vector<int> layersizes, hiddensizes = {256, 128};
  bool isTanh = false;
};
