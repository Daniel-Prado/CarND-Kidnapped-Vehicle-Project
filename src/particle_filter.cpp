/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */
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

using namespace std;


void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 50;

	default_random_engine gen;

	normal_distribution<double> Ndist_x_init(x, std[0]);
	normal_distribution<double> Ndist_y_init(y, std[1]);
	normal_distribution<double> Ndist_theta_init(theta, std[2]);

	for (int i = 0; i < num_particles; i++) {
		Particle particle;
		particle.id = i;
		particle.x = Ndist_x_init(gen);
		particle.y = Ndist_y_init(gen);
		particle.theta = Ndist_theta_init(gen);
		particle.weight = 1;

		particles.push_back(particle);
		weights.push_back(1.0);
	}

	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	default_random_engine gen;

	// define normal distributions for sensor noise
	normal_distribution<double> N_x(0, std_pos[0]);
	normal_distribution<double> N_y(0, std_pos[1]);
	normal_distribution<double> N_theta(0, std_pos[2]);

	for (auto& p : particles) {

				double yaw = p.theta;
				double yaw_p = yaw + (yaw_rate * delta_t);

		if(fabs(yaw_rate < 0.0001)){

			p.x += velocity * delta_t * cos( yaw );
			p.y += velocity * delta_t * sin( yaw );
		
		} else {

			p.x += (velocity / yaw_rate) * ( sin( yaw_p ) - sin( yaw ));
			p.y += (velocity / yaw_rate) * ( cos( yaw ) - cos( yaw_p ));
		}
		p.theta += yaw_rate * delta_t;

		p.x += N_x(gen);
		p.y += N_y(gen);
		p.theta += N_theta(gen);
	}


}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	double min_dist;

	for (int i = 0; i < observations.size(); i++) {

			// init minimum distance to the maximum possible
			min_dist = numeric_limits<double>::max();
			
			for (unsigned int j = 0; j < predicted.size(); j++) {
				// grab current prediction
				LandmarkObs p = predicted[j];
				
				// get distance between current/predicted landmarks
				double cur_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);

				// find the predicted landmark nearest the current observed landmark
				if (cur_dist < min_dist) {
					min_dist = cur_dist;
					observations[i].id = predicted[j].id;
				}
			}
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

	weights.clear();
	for(auto& part : particles)
	{
		vector<LandmarkObs> trans_observations;

		// Transform coordinates of the car measurements from the local car coordinate system to the
		// map's coordinate system, assuming that the measurement would be taken from the current particle.
		for (auto& obs: observations) {
			LandmarkObs trans_obs;

			// space transformation from vehicle to map coordinates
			//trans_obs.id = obs.id;
			trans_obs.x = part.x + obs.x * cos(part.theta) - obs.y * sin(part.theta);
			trans_obs.y = part.y + obs.x * sin(part.theta) + obs.y * cos(part.theta);

			trans_observations.push_back(trans_obs);
		}

		// Get the list of landmarks that are within sensor range.
		vector<LandmarkObs> nearby_landmarks;
		for ( auto& lm : map_landmarks.landmark_list ) {
						double distance = dist(part.x, part.y, lm.x_f, lm.y_f);
						if (distance < sensor_range) {
								nearby_landmarks.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
						}
		}

		//for (int i = 0 ; i < nearby_landmarks.size() ; i++)
		//{
		//	cout << "nearby_landmarks[" << i << "].ID: " << nearby_landmarks[i].id << endl;
		//}

		// If there are no nearby landmarks, we assign a weight zero to make the particle die.
		if (nearby_landmarks.size() == 0) {
						part.weight = 0.0;
						// we continue the loop for the next particle
						continue;
		}

		// Associate each measurement with a landmark identifier,
				// we only need to compare with the landmarks within range
		dataAssociation(nearby_landmarks, trans_observations);
		//for (int i = 0 ; i < trans_observations.size() ; i++)
		//{
		//	cout << "AFTER  ASS.,trans_observations[" << i << "].ID: " << trans_observations[i].id << endl;
		//}

		// Calculate the particle final weight by the product of
		// each measurement's Multivariate-Gaussian probability.

				vector<int> associations;
				vector<double> sense_x;
		vector<double> sense_y;
		double weight = 1.0;


		for(auto& o : trans_observations) {
			//we find the landmark association with the same ID that the transformed observation:
			auto pred = find_if(begin(nearby_landmarks), end(nearby_landmarks), [o](LandmarkObs const& l) {
								return (l.id == o.id);
						});

			double mu_x = pred->x; //map_landmarks.landmark_list[association].x_f;
			double mu_y = pred->y; // map_landmarks.landmark_list[association].y_f;

			double sig_x = std_landmark[0];
			double sig_y = std_landmark[1];

			long double multiplier = exp(-(pow(o.x - mu_x, 2) / (2 * pow(sig_x, 2))
				+ pow(o.y - mu_y, 2) / (2 * pow(sig_y, 2))) ) / (2 * M_PI * sig_x * sig_y);

			weight *= multiplier;

			associations.push_back(o.id);
			sense_x.push_back(o.x);
			sense_y.push_back(o.y);
		}

			if (associations.size() == 0) {
				// if no landmarks selected, we  let the particle die with a weight zero.
				part.weight = 0.0;
			} else {

					part = SetAssociations(part, associations, sense_x, sense_y);
			}
		part.weight = weight;
		weights.push_back(part.weight);
	}
}


void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	default_random_engine gen;
	discrete_distribution<int> distribution(weights.begin(), weights.end());

	vector<Particle> resample_particles;
	for (int i = 0; i < num_particles; i++)
	{
		resample_particles.push_back(particles[distribution(gen)]);
	}
	particles = resample_particles;
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
	
	//cout <<"particle ID("<<best.id<<") num.associations: " << best.associations.size();
	//for (int i = 0 ; i < best.associations.size(); i++)
	//{
	//	cout << ", assoc " << best.associations[i] << ",  x:" << best.sense_x[i] << ",  y:" << best.sense_y[i] << endl;
	//
	//}
	//cout << endl;
	
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

