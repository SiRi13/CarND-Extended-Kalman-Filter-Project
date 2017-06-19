#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  * Calculate the RMSE here.
  */
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
    cout << "Either Sizes don't match or size == 0" << endl;
    return rmse;
  }

  // accumulate squared residuals
  for (unsigned i = 0; i < estimations.size(); ++i) {
    VectorXd tmp = estimations[i] - ground_truth[i];
    tmp = tmp.array() * tmp.array();
    rmse += tmp;
  }

  // calculate the mean
  rmse = rmse / estimations.size();

  // calculate the squared root
  rmse = rmse.array().sqrt();

  // return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state) {
  /**
  * Calculate a Jacobian here.
  */
  MatrixXd Hj(3, 4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float px_2 = px * px;
  float py_2 = py * py;
  float pxy_2 = px_2 + py_2;
  float g = sqrt(pxy_2);
  float px_ = pow(pxy_2, 1.5);
  float vxpyvypx = (vx * py) - (vy * px);
  float vypxvxpy = (vy * px) - (vx * py);

  // check division by zero
  if (pxy_2 == 0 || g == 0 || px_ == 0) {
    cout << "Division by Zero!" << endl;
  } else {
    // compute the Jacobian matrix
    Hj << (px / g), (py / g), 0, 0, -(py / pxy_2), (px / pxy_2), 0, 0,
        ((py * vxpyvypx) / px_), ((px * vypxvxpy) / px_), (px / g), (py / g);
  }

  return Hj;
}
