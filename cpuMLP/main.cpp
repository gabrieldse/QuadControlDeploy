#include <Eigen/Core>
#include <iostream>
#include "Interface.hpp"

int main() {
  Eigen::VectorXf input;
  input.setRandom(5);

  MLP_3<132, 12> net = MLP_3<132, 12>();
  // net.load_checkpoint("/home/maractin/Workspace/quadruped-replay/checkpoints/vel_3d/flat/p2/jit_20000.pt");
  Eigen::Matrix<float, 132, 1> input1 = Eigen::Matrix<float, 132, 1>::Zero();
  std::cout << net.forward(input1) << std::endl;

  MLP_2<123, 11> net2 = MLP_2<123, 11>();

  std::cout << "----" << std::endl;
  Interface test = Interface();
  std::string polDirName =
      "../../tmp_checkpoints/sym_pose/energy/6cm/policy-08-03-01-20-47/";
  std::string estDirName =
      "../../tmp_checkpoints/state_estimation/symmetric_state_estimator.txt";
  test.initialize(polDirName, estDirName, Vector12::Ones());
}
