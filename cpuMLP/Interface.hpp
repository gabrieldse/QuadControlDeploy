///////////////////////////////////////////////////////////////////////////////////////////////////
///
/// \brief This is the header for Interface class
///
/// \details C++ interface between the control loop and the low-level neural
/// network code
///
//////////////////////////////////////////////////////////////////////////////////////////////////

#include <chrono>
#include "Types.h"
#include "cpuMLP.hpp"

class Interface {
 public:
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Empty constructor
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Interface();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Destructor.
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  ~Interface() {}

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Initializer
  ///
  /// \param[in] polDirName Name of directory that contains policy parameters
  /// \param[in] estFileName Name of file that contains estimation parameters
  /// \param[in] q_init Initial joint configuration of the robot
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void initialize(std::string polDirName, std::string estFileName,
                  Vector12 q_init);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief  Forward pass
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Vector12 forward();

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Run observation network and refresh observation vector
  ///
  /// \param[in] pos Joint positions
  /// \param[in] vel Joint velocities
  /// \param[in] ori Base orientation (Euler angles)
  /// \param[in] gyro Base angular velocities
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void update_observation(Vector12 pos, Vector12 vel, Vector3 ori,
                          Vector3 gyro);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Refresh history vector
  ///
  /// \param[in] pos Joint positions
  /// \param[in] vel Joint velocities
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  void update_history(Vector12 pos, Vector12 vel);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Convert roll, pitch and yaw angles into the corresponding rotation
  /// matrix
  ///
  /// \param[in] r Roll angle
  /// \param[in] p Pitch angle
  /// \param[in] y Yaw angle
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Matrix3 rpyToMatrix(float r, float p, float y);

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Return the observation vector
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  Vector132 get_observation() { return obs_; }

  ////////////////////////////////////////////////////////////////////////////////////////////////
  ///
  /// \brief Return the computation time to run the observation and control
  /// networks
  ///
  ////////////////////////////////////////////////////////////////////////////////////////////////
  float get_computation_time() {
    return static_cast<float>(
        std::chrono::duration_cast<std::chrono::microseconds>(t_end_ - t_start_)
            .count());
  }

  // Control policy
  MLP_3<132, 12> policy_;

  // Estimation policy
  MLP_2<123, 3> state_estimator_;

  // Normalization of the input of the control policy
  Vector132 obs_mean_;
  Vector132 obs_var_;

  // History buffers
  const int num_history_stack_ = 6;
  MatrixN q_pos_error_hist_;
  MatrixN qd_hist_;
  Vector12 previous_action_;
  Vector12 preprevious_action_;

  // Misc
  Vector12 P_;
  Vector12 D_;
  Vector12 pTarget12_;
  Vector12 q_init_;
  Vector3 vel_command_;
  Vector123 state_est_obs_;
  Vector132 obs_;
  Vector132 obs_normalized_;
  Vector132 bound_;
  Vector12 bound_pi_;
  std::chrono::time_point<std::chrono::steady_clock> t_start_;
  std::chrono::time_point<std::chrono::steady_clock> t_end_;
};

Interface::Interface()
    : obs_mean_(Vector132::Zero()),
      obs_var_(Vector132::Zero()),
      q_pos_error_hist_(MatrixN::Zero(num_history_stack_, 12)),
      qd_hist_(MatrixN::Zero(num_history_stack_, 12)),
      previous_action_(Vector12::Zero()),
      preprevious_action_(Vector12::Zero()),
      P_(Vector12::Zero()),
      D_(Vector12::Zero()),
      pTarget12_(Vector12::Zero()),
      q_init_(Vector12::Zero()),
      vel_command_(Vector3::Zero()),
      state_est_obs_(Eigen::Matrix<float, 123, 1>::Zero()),
      obs_(Vector132::Zero()),
      obs_normalized_(Vector132::Zero()),
      bound_(Vector132::Ones() * 10.0f),
      bound_pi_(Vector12::Ones() * 3.1415f) {
  // Empty
}

void Interface::initialize(std::string polDirName, std::string estFileName,
                           Vector12 q_init) {
  // Control policy
  policy_.updateParamFromTxt(polDirName + "full_2000.txt");

  // Estimation policy
  state_estimator_.updateParamFromTxt(estFileName);

  // Normalization
  std::string in_line;
  std::ifstream obsMean_file, obsVariance_file;
  obsMean_file.open(polDirName + "mean2000.csv");
  obsVariance_file.open(polDirName + "var2000.csv");
  if (obsMean_file.is_open()) {
    for (int i = 0; i < obs_mean_.size(); i++) {
      std::getline(obsMean_file, in_line);
      obs_mean_(i) = std::stof(in_line);
    }
    obsMean_file.close();
  } else {
    throw std::runtime_error("Failed to open obsMean file.");
  }

  if (obsVariance_file.is_open()) {
    for (int i = 0; i < obs_var_.size(); i++) {
      std::getline(obsVariance_file, in_line);
      obs_var_(i) = std::stof(in_line);
    }
    obsVariance_file.close();
  } else {
    throw std::runtime_error("Failed to open obsVariance file.");
  }

  // Control gains
  P_ = (Vector3(3.0f, 3.0f, 3.0f)).replicate<4, 1>();
  D_ = (Vector3(0.2f, 0.2f, 0.2f)).replicate<4, 1>();

  // Velocity command
  vel_command_ = Vector3(0.5f, 0.0f, 0.0f);

  // Initial position
  q_init_ = q_init;
  previous_action_ = q_init;
  preprevious_action_ = q_init;

  // Initial times
  t_start_ = std::chrono::steady_clock::now();
  t_end_ = std::chrono::steady_clock::now();
}

Vector12 Interface::forward() {
  obs_normalized_ =
      ((obs_ - obs_mean_).array() /
       (obs_var_ + .1E-8f * Vector132::Ones()).cwiseSqrt().array())
          .matrix();
  obs_normalized_ = obs_normalized_.cwiseMax(-bound_).cwiseMin(bound_);

  pTarget12_ = q_init_ + 0.3f * policy_.forward(obs_normalized_)
                                    .cwiseMax(-bound_pi_)
                                    .cwiseMin(bound_pi_);

  // Log time
  t_end_ = std::chrono::steady_clock::now();

  return pTarget12_;
}

void Interface::update_observation(Vector12 pos, Vector12 vel, Vector3 ori,
                                   Vector3 gyro) {
  // Log time
  t_start_ = std::chrono::steady_clock::now();

  // Update the history
  update_history(pos, vel);

  // Update observation vectors
  state_est_obs_.head(3) = rpyToMatrix(ori(0, 0), ori(1, 0), ori(2, 0)).row(2);
  state_est_obs_.segment<12>(3 + 12 * 0) = pos;
  state_est_obs_.segment<12>(3 + 12 * 1) = vel;
  state_est_obs_.segment<12>(3 + 12 * 2) = previous_action_;
  state_est_obs_.segment<12>(3 + 12 * 3) = preprevious_action_;
  state_est_obs_.segment<12>(3 + 12 * 4) = q_pos_error_hist_.row(0);
  state_est_obs_.segment<12>(3 + 12 * 5) = qd_hist_.row(0);
  state_est_obs_.segment<12>(3 + 12 * 6) = q_pos_error_hist_.row(2);
  state_est_obs_.segment<12>(3 + 12 * 7) = qd_hist_.row(2);
  state_est_obs_.segment<12>(3 + 12 * 8) = q_pos_error_hist_.row(4);
  state_est_obs_.segment<12>(3 + 12 * 9) = qd_hist_.row(4);

  obs_.head(3) = rpyToMatrix(ori(0, 0), ori(1, 0), ori(2, 0)).row(2);
  obs_.segment<3>(3) = state_estimator_.forward(state_est_obs_).head(3);
  obs_.segment<3>(6) = gyro;
  obs_.segment<3>(9) = vel_command_;
  obs_.segment<12>(12) = pos;
  obs_.segment<12>(12 + 12 * 1) = vel;
  obs_.segment<12>(12 + 12 * 2) = previous_action_;
  obs_.segment<12>(12 + 12 * 3) = preprevious_action_;
  obs_.segment<12>(12 + 12 * 4) = q_pos_error_hist_.row(0);
  obs_.segment<12>(12 + 12 * 5) = qd_hist_.row(0);
  obs_.segment<12>(12 + 12 * 6) = q_pos_error_hist_.row(2);
  obs_.segment<12>(12 + 12 * 7) = qd_hist_.row(2);
  obs_.segment<12>(12 + 12 * 8) = q_pos_error_hist_.row(4);
  obs_.segment<12>(12 + 12 * 9) = qd_hist_.row(4);
}

void Interface::update_history(Vector12 pos, Vector12 vel) {
  // Age pos error history
  for (int index = 1; index < q_pos_error_hist_.rows(); index++) {
    q_pos_error_hist_.row(index - 1).swap(q_pos_error_hist_.row(index));
  }

  // Insert a new line at the end of the pos error history
  q_pos_error_hist_.row(q_pos_error_hist_.rows() - 1) = pTarget12_ - pos;

  // Age vel history
  for (int index = 1; index < qd_hist_.rows(); index++) {
    qd_hist_.row(index - 1).swap(qd_hist_.row(index));
  }

  // Insert a new line at the end of the vel history
  qd_hist_.row(qd_hist_.rows() - 1) = vel;

  // Remember previous actions
  preprevious_action_ = previous_action_;
  previous_action_ = pTarget12_;
}

Matrix3 Interface::rpyToMatrix(float r, float p, float y) {
  typedef Eigen::AngleAxis<float> AngleAxis;
  return (AngleAxis(y, Vector3::UnitZ()) * AngleAxis(p, Vector3::UnitY()) *
          AngleAxis(r, Vector3::UnitX()))
      .toRotationMatrix();
}