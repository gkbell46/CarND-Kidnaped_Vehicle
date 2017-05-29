/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#define NUM_OF_PARTICLES 200
using namespace std;
// declare a random engine to be used across multiple and various method calls
static default_random_engine gen;



void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	normal_distribution<double> x_noise(0,std[0]);
	normal_distribution<double> y_noise(0,std[1]);
	normal_distribution<double> theta_noise(0,std[2]);
	
	num_particles = NUM_OF_PARTICLES;
	weights.resize(num_particles,1.0);
	for (int i=0; i< num_particles; i++)
	{
		Particle pf_;
		pf_.x = x;
    pf_.y = y;
    pf_.theta = theta; 
		pf_.id = i;
		pf_.weight = 1.0;
		

		// Initialize and add noise 
		pf_.x += x_noise(gen);
		pf_.y += y_noise(gen);
		pf_.theta += theta_noise(gen);

		particles.push_back(pf_);

	}
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/pa

	normal_distribution<double> x_noise(0, std_pos[0]);
	normal_distribution<double> y_noise(0, std_pos[1]);
	normal_distribution<double> theta_noise(0, std_pos[2]);

	double ctrv_mul = velocity/yaw_rate;

	for (int i = 0; i < num_particles; ++i)
	{
		/* code */
		double delta_theta = particles[i].theta + yaw_rate * delta_t;
		double theta = particles[i].theta;
		if (fabs(yaw_rate) > 0.00001)
		{
			particles[i].x += ctrv_mul * (sin(delta_theta)- sin(theta));
			particles[i].y += ctrv_mul * (cos(theta) - cos(delta_theta));
			particles[i].theta += yaw_rate * delta_t;
		}
		else
		{
			particles[i].x += velocity * delta_t * cos(theta);
			particles[i].y += velocity * delta_t * sin(theta);
 		}
 		
 		//Add noise
 		particles[i].x += x_noise(gen);
 		particles[i].y += y_noise(gen);
 		particles[i].theta += theta_noise(gen);
	}

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (unsigned i =0 ; i< observations.size(); i++)
	{
		//Current Observation
		LandmarkObs c_obs = observations[i];

		double min_dist = INFINITY;

		int id = -1;


		for (unsigned j=0; j < predicted.size(); j++)
		{
			//Current Prediction
			LandmarkObs c_pred = predicted[j];

			//dist_x = predicted[j].x - observations[i].x;
			//dist_y = predicted[j].y - observations[j].y;
			//distance = dist_x * dist_x + dist_y * dist_y;
			double cur_dist = dist(c_obs.x,c_obs.y,c_pred.x,c_pred.y);
			if (cur_dist < min_dist)
			{
				min_dist = cur_dist;
				id = c_pred.id;
			}

		}
		//lm_observations.push_back(predicted[id]);
		observations[i].id = id; 

	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	
	double sigma_lm[2];		// landmark measurement uncertinity [x[m], y[m]]
	sigma_lm[0] = std_landmark[0];
	sigma_lm[1] = std_landmark[1];
	for (unsigned i = 0; i < particles.size(); ++i)
	{
		
		/* code */
		// current particle information
		double p_x = particles[i].x;
		double p_y = particles[i].y;
		double p_theta = particles[i].theta;
		
		//Create storage for predicted landmarks
		std::vector<LandmarkObs> predicted_landmarks;
		for (auto map_lm : map_landmarks.landmark_list)
		{
			LandmarkObs pred_lm;
			pred_lm.x = map_lm.x_f;
			pred_lm.y = map_lm.y_f;
			pred_lm.id = map_lm.id_i;
		
			//Add only if in the range
			if (fabs(pred_lm.x - p_x) <= sensor_range && fabs(pred_lm.y - p_y) <= sensor_range)
				predicted_landmarks.push_back(pred_lm);

		}

		//Create storage for trasnformed Observations
		vector<LandmarkObs> transformed_obs;
				
		for (auto obs_lm : observations)
		{
			/* code */
			LandmarkObs obs_global;
			obs_global.x = obs_lm.x * cos(p_theta) - obs_lm.y * sin(p_theta) + p_x;
			obs_global.y = obs_lm.x * sin(p_theta) + obs_lm.y * cos(p_theta) + p_y;
			obs_global.id = obs_lm.id;
			transformed_obs.push_back(obs_global); 
		}

        
		//Find out associate landmarks
		dataAssociation(predicted_landmarks,transformed_obs);
		
		//Init Weight
		particles[i].weight = 1.0;
		

		//Calculate weight
		for (unsigned j =0; j< transformed_obs.size(); j++)
		{
			double pred_x, pred_y, obs_x, obs_y;
			auto transformed_ob = transformed_obs[j];
			for (unsigned k =0; k< predicted_landmarks.size(); k++)
			{
				if (predicted_landmarks[k].id == transformed_ob.id)
				{
					pred_x = predicted_landmarks[k].x;
					pred_y = predicted_landmarks[k].y;

				}

			}

			double cov_x = sigma_lm[0] * sigma_lm[0];
			double cov_y = sigma_lm[1] * sigma_lm[1];
			double normalizer =  2.0 * M_PI * sigma_lm[0] * sigma_lm[1];
			double dist_x = (pred_x - transformed_obs[j].x );
			double dist_y = (pred_y - transformed_obs[j].y);
			double pdf = exp (-(dist_x * dist_x/(2*cov_x) + dist_y* dist_y/(2*cov_y)))/normalizer;
   			
   			particles[i].weight *= pdf;
		}

	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	//discrete_distribution<int> d(weights.begin(),weights.end());

	vector<Particle> new_particles;

  // get all of the current weights
  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);
  }

  // generate random starting index for resampling wheel
  uniform_int_distribution<int> uniintdist(0, num_particles-1);
  auto index = uniintdist(gen);

  // get max weight
  double max_weight = *max_element(weights.begin(), weights.end());

  // uniform random distribution [0.0, max_weight)
  uniform_real_distribution<double> unirealdist(0.0, max_weight);

  double beta = 0.0;

  // spin the resample wheel!
  for (int i = 0; i < num_particles; i++) {
    beta += unirealdist(gen) * 2.0;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }

particles = new_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
