#pragma once

#include "mymath.cuh"

struct hit_record
{
	double t;
	Vec3 p;
	Vec3 normal;
};

class Ray {
public:
	__device__ Ray() {};
	__device__ Ray(const Vec3& ori, const Vec3& dir) :origin(ori), direction(dir) { direction.normalize(); }

	__device__ Vec3 point_at_parameter(double t) const { return origin + direction*t; }

	Vec3 origin;
	Vec3 direction;
};