#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
	is_initialized_ = false;

	previous_timestamp_ = 0;

	// initializing matrices
	R_laser_ = MatrixXd(2, 2);
	R_radar_ = MatrixXd(3, 3);
	H_laser_ = MatrixXd(2, 4);
	Hj_ = MatrixXd(3, 4);

	//measurement covariance matrix - laser
	R_laser_ << 0.0225, 0,
	      0, 0.0225;

	//measurement covariance matrix - radar
	R_radar_ << 0.09, 0, 0,
	      0, 0.0009, 0,
	      0, 0, 0.09;

	H_laser_ << 1, 0, 0, 0,
	      0, 1, 0, 0;

	// 4x4 matrix
	ekf_.F_ = MatrixXd(4, 4);
	ekf_.F_ << 1, 0, 1, 0,
	     0, 1, 0, 1,
	     0, 0, 1, 0,
	     0, 0, 0, 1;

	// 4x4 matrix
	ekf_.P_ = MatrixXd(4, 4);
	ekf_.P_ << 1, 0, 0, 0,
	     0, 1, 0, 0,
	     0, 0, 1000, 0,
	     0, 0, 0, 1000;

	noise_ax = 9;
	noise_ay = 9;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
	/*****************************************************************************
	*  Initialization
	****************************************************************************/
	if (!is_initialized_) {
		/**
		* Initialize the state ekf_.x_ with the first measurement.
		* Create the covariance matrix.
		* Remember: you'll need to convert radar from polar to cartesian coordinates.
		*/
		
		// first measurement
		cout << "EKF: " << endl;
		ekf_.x_ = VectorXd(4);
		// important for RMSE
		ekf_.x_ << 1, 1, 1, 1;

		if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
			/**
			Convert radar from polar to cartesian coordinates and initialize state.
			*/
			ekf_.x_(0) = measurement_pack.raw_measurements_(0) * cos(measurement_pack.raw_measurements_(1));
			ekf_.x_(1) = measurement_pack.raw_measurements_(0) * sin(measurement_pack.raw_measurements_(1));
		}
		else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
			/**
			Initialize state.
			*/
			ekf_.x_(0) = measurement_pack.raw_measurements_(0);
			ekf_.x_(1) = measurement_pack.raw_measurements_(1);
		}

		ekf_.F_ << 1, 0, 1, 0,
		       0, 1, 0, 1,
		       0, 0, 1, 0,
		       0, 0, 0, 1;
		previous_timestamp_ = measurement_pack.timestamp_;

		// done initializing, no need to predict or update
		is_initialized_ = true;
		return;
	}

	/*****************************************************************************
	*  Prediction
	****************************************************************************/

	/**
	* Update the state transition matrix F according to the new elapsed time.
	- Time is measured in seconds.
	* Update the process noise covariance matrix.
	* Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
	*/
	float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
	previous_timestamp_ = measurement_pack.timestamp_;

	//1. Modify the F matrix so that the time is integrated
	ekf_.F_(0, 2) = dt;
	ekf_.F_(1, 3) = dt;

	//2. Set the process covariance matrix Q
	float dt_2 = dt * dt;
	float dt_3 = dt_2 * dt;
	float dt_4 = dt_3 * dt;
	ekf_.Q_ = MatrixXd(4, 4);
	ekf_.Q_ << (dt_4/4)*noise_ax, 0, (dt_3/2)*noise_ax, 0,
			0, (dt_4/4)*noise_ay, 0, (dt_3/2)*noise_ay,
			(dt_3/2)*noise_ax, 0, dt_2*noise_ax, 0,
			0, (dt_3/2)*noise_ay, 0, dt_2*noise_ay;

	//3. Call the Kalman Filter predict() function
	ekf_.Predict();

	/*****************************************************************************
	*  Update
	****************************************************************************/

	/**
	* Use the sensor type to perform the update step.
	* Update the state and covariance matrices.
	*/
	//4. Call the Kalman Filter update() function
	// with the most recent raw measurements_
	if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
		// Radar updates
		ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
		ekf_.R_ = R_radar_;

		ekf_.UpdateEKF(measurement_pack.raw_measurements_);
	}
	else {
		// Laser updates
		ekf_.H_ = H_laser_;
		ekf_.R_ = R_laser_;
		ekf_.Update(measurement_pack.raw_measurements_);
	}

	// print the output
	cout << "x_ = " << ekf_.x_ << endl;
	cout << "P_ = " << ekf_.P_ << endl;
}
