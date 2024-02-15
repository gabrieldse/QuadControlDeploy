#ifndef TYPES_H_INCLUDED
#define TYPES_H_INCLUDED

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

using Vector1 = Eigen::Matrix<float, 1, 1>;
using Vector2 = Eigen::Matrix<float, 2, 1>;
using Vector3 = Eigen::Matrix<float, 3, 1>;
using Vector4 = Eigen::Matrix<float, 4, 1>;
using Vector5 = Eigen::Matrix<float, 5, 1>;
using Vector6 = Eigen::Matrix<float, 6, 1>;
using Vector7 = Eigen::Matrix<float, 7, 1>;
using Vector8 = Eigen::Matrix<float, 8, 1>;
using Vector11 = Eigen::Matrix<float, 11, 1>;
using Vector12 = Eigen::Matrix<float, 12, 1>;
using Vector18 = Eigen::Matrix<float, 18, 1>;
using Vector19 = Eigen::Matrix<float, 19, 1>;
using Vector24 = Eigen::Matrix<float, 24, 1>;
using Vector123 = Eigen::Matrix<float, 123, 1>;
using Vector132 = Eigen::Matrix<float, 132, 1>;
using VectorN = Eigen::Matrix<float, Eigen::Dynamic, 1>;

using Matrix2 = Eigen::Matrix<float, 2, 2>;
using Matrix3 = Eigen::Matrix<float, 3, 3>;
using Matrix4 = Eigen::Matrix<float, 4, 4>;
using Matrix12 = Eigen::Matrix<float, 12, 12>;

using Matrix13 = Eigen::Matrix<float, 1, 3>;
using Matrix14 = Eigen::Matrix<float, 1, 4>;
using Matrix112 = Eigen::Matrix<float, 1, 12>;
using Matrix118 = Eigen::Matrix<float, 1, 18>;
using Matrix34 = Eigen::Matrix<float, 3, 4>;
using Matrix43 = Eigen::Matrix<float, 4, 3>;
using Matrix64 = Eigen::Matrix<float, 6, 4>;
using Matrix74 = Eigen::Matrix<float, 7, 4>;
using MatrixN4 = Eigen::Matrix<float, Eigen::Dynamic, 4>;
using Matrix3N = Eigen::Matrix<float, 3, Eigen::Dynamic>;
using Matrix6N = Eigen::Matrix<float, 6, Eigen::Dynamic>;
using MatrixN = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

#endif  // TYPES_H_INCLUDED
