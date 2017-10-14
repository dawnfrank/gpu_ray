#pragma once

#include "mymath.cuh"
#include "ray.cuh"

class Sphere{
public:
	__device__ Sphere() {}
	__device__ Sphere(Vec3 cen, double r) : center(cen), radius(r) {};

	__device__ bool hit(const Ray& r, double tmin, double tmax, hit_record& rec) const;

	Vec3 center;
	double radius;
};

__device__ bool Sphere::hit(const Ray& r, double t_min, double t_max, hit_record& rec) const {
	double t = 0.5*(r.direction.y + 1.0);
	return Vec3(1.0, 1.0, 1.0)*(1.0 - t) + Vec3(0.5, 0.7, 1.0)*t;

	Vec3 oc = r.origin - center;
	double a = r.direction*r.direction;
	double b = oc*r.direction;
	double c = oc*oc - radius*radius;
	double discriminant = b*b - a*c;
	if (discriminant > 0) {
		double temp = (-b - sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			return true;
		}
		temp = (-b + sqrt(discriminant)) / a;
		if (temp < t_max && temp > t_min) {
			rec.t = temp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			return true;
		}
	}
	return false;
}