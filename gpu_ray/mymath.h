#pragma once

#include <math.h>

#define EPSILON 1e-6f
#define PI 3.14159265358979323846f
#define DEG2RAD (PI / 180.0f)
#define RAD2DEG (180.0f / PI)

struct Vec2;
struct Vec3;
struct Vec4;
struct Mat3;
struct Mat4;
struct quat;

/*****************************************************************************/
/*                                                                           */
/* Vec2                                                                      */
/*                                                                           */
/*****************************************************************************/

struct Vec2 {

	inline Vec2() : x(0), y(0) { }
	inline Vec2(double x, double y) : x(x), y(y) { }
	inline Vec2(const double *v) : x(v[0]), y(v[1]) { }
	inline Vec2(const Vec2 &v) : x(v.x), y(v.y) { }

	inline int operator==(const Vec2 &v) { return (fabs(x - v.x) < EPSILON && fabs(y - v.y) < EPSILON); }
	inline int operator!=(const Vec2 &v) { return !(*this == v); }

	inline const Vec2 operator*(double f) const { return Vec2(x * f, y * f); }
	inline const Vec2 operator/(double f) const { return Vec2(x / f, y / f); }
	inline const Vec2 operator+(const Vec2 &v) const { return Vec2(x + v.x, y + v.y); }
	inline const Vec2 operator-() const { return Vec2(-x, -y); }
	inline const Vec2 operator-(const Vec2 &v) const { return Vec2(x - v.x, y - v.y); }

	inline Vec2 &operator*=(double f) { return *this = *this * f; }
	inline Vec2 &operator/=(double f) { return *this = *this / f; }
	inline Vec2 &operator+=(const Vec2 &v) { return *this = *this + v; }
	inline Vec2 &operator-=(const Vec2 &v) { return *this = *this - v; }

	inline double operator*(const Vec2 &v) const { return x * v.x + y * v.y; }

	inline operator double*() { return (double*)&x; }
	inline operator const double*() const { return (double*)&x; }

	inline double &operator[](int i) { return ((double*)&x)[i]; }
	inline const double operator[](int i) const { return ((double*)&x)[i]; }

	inline double magnitude() const { return sqrt(x * x + y * y); }
	inline double normalize() {
		double inv, length = magnitude();
		if (length < EPSILON) return 0.0;
		inv = 1.0 / length;
		x *= inv;
		y *= inv;
		return length;
	}

	union {
		struct {
			double x, y;
		};
		double v[2];
	};
};

inline Vec2 operator*(double t, const Vec2 &v) { return v * t; }
inline Vec2 operator/(double t, const Vec2 &v) { return v / t; }


/*****************************************************************************/
/*                                                                           */
/* Vec3                                                                      */
/*                                                                           */
/*****************************************************************************/

struct Vec3 {

	inline Vec3() : x(0), y(0), z(0) { }
	inline Vec3(double x, double y, double z) : x(x), y(y), z(z) { }
	inline Vec3(const double *v) : x(v[0]), y(v[1]), z(v[2]) { }
	inline Vec3(const Vec3 &v) : x(v.x), y(v.y), z(v.z) { }
	inline Vec3(const Vec4 &v);

	inline int operator==(const Vec3 &v) { return (fabs(x - v.x) < EPSILON && fabs(y - v.y) < EPSILON && fabs(z - v.z) < EPSILON); }
	inline int operator!=(const Vec3 &v) { return !(*this == v); }

	inline const Vec3 operator*(double f) const { return Vec3(x * f, y * f, z * f); }
	inline const Vec3 operator/(double f) const { return Vec3(x / f, y / f, z / f); }
	inline const Vec3 operator+(const Vec3 &v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
	inline const Vec3 operator-() const { return Vec3(-x, -y, -z); }
	inline const Vec3 operator-(const Vec3 &v) const { return Vec3(x - v.x, y - v.y, z - v.z); }

	inline Vec3 &operator*=(double f) { return *this = *this * f; }
	inline Vec3 &operator/=(double f) { return *this = *this / f; }
	inline Vec3 &operator+=(const Vec3 &v) { return *this = *this + v; }
	inline Vec3 &operator-=(const Vec3 &v) { return *this = *this - v; }

	inline double operator*(const Vec3 &v) const { return x * v.x + y * v.y + z * v.z; }
	inline double operator*(const Vec4 &v) const;

	inline operator double*() { return (double*)&x; }
	inline operator const double*() const { return (double*)&x; }

	inline double &operator[](int i) { return ((double*)&x)[i]; }
	inline const double operator[](int i) const { return ((double*)&x)[i]; }

	inline double magnitude() const { return sqrt(x * x + y * y + z * z); }
	inline double normalize() {
		double inv, length = magnitude();
		if (length < EPSILON) return 0.0;
		inv = 1.0 / length;
		x *= inv;
		y *= inv;
		z *= inv;
		return length;
	}
	inline void cross(const Vec3 &v1, const Vec3 &v2) {
		x = v1.y * v2.z - v1.z * v2.y;
		y = v1.z * v2.x - v1.x * v2.z;
		z = v1.x * v2.y - v1.y * v2.x;
	}

	union {
		struct {
			double x, y, z;
		};
		struct {
			double r, g, b;
		};
		double v[3];
	};
};

inline Vec3 cross(const Vec3 &v1, const Vec3 &v2) {
	Vec3 ret;
	ret.x = v1.y * v2.z - v1.z * v2.y;
	ret.y = v1.z * v2.x - v1.x * v2.z;
	ret.z = v1.x * v2.y - v1.y * v2.x;
	return ret;
}

inline Vec3 operator*(double t, const Vec3 &v) { return v * t; }
inline Vec3 operator/(double t, const Vec3 &v) { return v / t; }

/*****************************************************************************/
/*                                                                           */
/* Vec4                                                                      */
/*                                                                           */
/*****************************************************************************/

struct Vec4 {

	inline Vec4() : x(0), y(0), z(0), w(1) { }
	inline Vec4(double x, double y, double z, double w) : x(x), y(y), z(z), w(w) { }
	inline Vec4(const double *v) : x(v[0]), y(v[1]), z(v[2]), w(v[3]) { }
	inline Vec4(const Vec3 &v) : x(v.x), y(v.y), z(v.z), w(1) { }
	inline Vec4(const Vec3 &v, double w) : x(v.x), y(v.y), z(v.z), w(w) { }
	inline Vec4(const Vec4 &v) : x(v.x), y(v.y), z(v.z), w(v.w) { }

	inline int operator==(const Vec4 &v) { return (fabs(x - v.x) < EPSILON && fabs(y - v.y) < EPSILON && fabs(z - v.z) < EPSILON && fabs(w - v.w) < EPSILON); }
	inline int operator!=(const Vec4 &v) { return !(*this == v); }

	inline const Vec4 operator*(double f) const { return Vec4(x * f, y * f, z * f, w * f); }
	inline const Vec4 operator/(double f) const { return Vec4(x / f, y / f, z / f, w / f); }
	inline const Vec4 operator+(const Vec4 &v) const { return Vec4(x + v.x, y + v.y, z + v.z, w + v.w); }
	inline const Vec4 operator-() const { return Vec4(-x, -y, -z, -w); }
	inline const Vec4 operator-(const Vec4 &v) const { return Vec4(x - v.x, y - v.y, z - v.z, z - v.w); }

	inline Vec4 &operator*=(double f) { return *this = *this * f; }
	inline Vec4 &operator/=(double f) { return *this = *this / f; }
	inline Vec4 &operator+=(const Vec4 &v) { return *this = *this + v; }
	inline Vec4 &operator-=(const Vec4 &v) { return *this = *this - v; }

	inline double operator*(const Vec3 &v) const { return x * v.x + y * v.y + z * v.z + w; }
	inline double operator*(const Vec4 &v) const { return x * v.x + y * v.y + z * v.z + w * v.w; }

	inline operator double*() { return (double*)&x; }
	inline operator const double*() const { return (double*)&x; }

	inline double &operator[](int i) { return ((double*)&x)[i]; }
	inline const double operator[](int i) const { return ((double*)&x)[i]; }

	union {
		struct {
			double x, y, z, w;
		};
		double v[4];
	};
};

inline Vec3::Vec3(const Vec4 &v) {
	x = v.x;
	y = v.y;
	z = v.z;
}

inline double Vec3::operator*(const Vec4 &v) const {
	return x * v.x + y * v.y + z * v.z + v.w;
}

inline Vec4 operator*(double t, const Vec4 &v) { return v * t; }
inline Vec4 operator/(double t, const Vec4 &v) { return v / t; }


/*****************************************************************************/
/*                                                                           */
/* Mat3                                                                      */
/*                                                                           */
/*****************************************************************************/

struct Mat3 {

	Mat3() {
		mat[0] = 1.0; mat[3] = 0.0; mat[6] = 0.0;
		mat[1] = 0.0; mat[4] = 1.0; mat[7] = 0.0;
		mat[2] = 0.0; mat[5] = 0.0; mat[8] = 1.0;
	}
	Mat3(const double *m) {
		mat[0] = m[0]; mat[3] = m[3]; mat[6] = m[6];
		mat[1] = m[1]; mat[4] = m[4]; mat[7] = m[7];
		mat[2] = m[2]; mat[5] = m[5]; mat[8] = m[8];
	}
	Mat3(const Mat3 &m) {
		mat[0] = m[0]; mat[3] = m[3]; mat[6] = m[6];
		mat[1] = m[1]; mat[4] = m[4]; mat[7] = m[7];
		mat[2] = m[2]; mat[5] = m[5]; mat[8] = m[8];
	}
	Mat3(const Vec3 &v1, const Vec3 &v2, const Vec3 &v3) {
		mat[0] = v1[0]; mat[3] = v2[0]; mat[6] = v3[0];
		mat[1] = v1[1]; mat[4] = v2[1]; mat[7] = v3[1];
		mat[2] = v1[2]; mat[5] = v2[2]; mat[8] = v3[2];
	}
	Mat3(const Mat4 &m);

	Vec3 operator*(const Vec3 &v) const {
		Vec3 ret;
		ret[0] = mat[0] * v[0] + mat[3] * v[1] + mat[6] * v[2];
		ret[1] = mat[1] * v[0] + mat[4] * v[1] + mat[7] * v[2];
		ret[2] = mat[2] * v[0] + mat[5] * v[1] + mat[8] * v[2];
		return ret;
	}
	Vec4 operator*(const Vec4 &v) const {
		Vec4 ret;
		ret[0] = mat[0] * v[0] + mat[3] * v[1] + mat[6] * v[2];
		ret[1] = mat[1] * v[0] + mat[4] * v[1] + mat[7] * v[2];
		ret[2] = mat[2] * v[0] + mat[5] * v[1] + mat[8] * v[2];
		ret[3] = v[3];
		return ret;
	}
	Mat3 operator*(double f) const {
		Mat3 ret;
		ret[0] = mat[0] * f; ret[3] = mat[3] * f; ret[6] = mat[6] * f;
		ret[1] = mat[1] * f; ret[4] = mat[4] * f; ret[7] = mat[7] * f;
		ret[2] = mat[2] * f; ret[5] = mat[5] * f; ret[8] = mat[8] * f;
		return ret;
	}
	Mat3 operator*(const Mat3 &m) const {
		Mat3 ret;
		ret[0] = mat[0] * m[0] + mat[3] * m[1] + mat[6] * m[2];
		ret[1] = mat[1] * m[0] + mat[4] * m[1] + mat[7] * m[2];
		ret[2] = mat[2] * m[0] + mat[5] * m[1] + mat[8] * m[2];
		ret[3] = mat[0] * m[3] + mat[3] * m[4] + mat[6] * m[5];
		ret[4] = mat[1] * m[3] + mat[4] * m[4] + mat[7] * m[5];
		ret[5] = mat[2] * m[3] + mat[5] * m[4] + mat[8] * m[5];
		ret[6] = mat[0] * m[6] + mat[3] * m[7] + mat[6] * m[8];
		ret[7] = mat[1] * m[6] + mat[4] * m[7] + mat[7] * m[8];
		ret[8] = mat[2] * m[6] + mat[5] * m[7] + mat[8] * m[8];
		return ret;
	}
	Mat3 operator+(const Mat3 &m) const {
		Mat3 ret;
		ret[0] = mat[0] + m[0]; ret[3] = mat[3] + m[3]; ret[6] = mat[6] + m[6];
		ret[1] = mat[1] + m[1]; ret[4] = mat[4] + m[4]; ret[7] = mat[7] + m[7];
		ret[2] = mat[2] + m[2]; ret[5] = mat[5] + m[5]; ret[8] = mat[8] + m[8];
		return ret;
	}
	Mat3 operator-(const Mat3 &m) const {
		Mat3 ret;
		ret[0] = mat[0] - m[0]; ret[3] = mat[3] - m[3]; ret[6] = mat[6] - m[6];
		ret[1] = mat[1] - m[1]; ret[4] = mat[4] - m[4]; ret[7] = mat[7] - m[7];
		ret[2] = mat[2] - m[2]; ret[5] = mat[5] - m[5]; ret[8] = mat[8] - m[8];
		return ret;
	}

	Mat3 &operator*=(double f) { return *this = *this * f; }
	Mat3 &operator*=(const Mat3 &m) { return *this = *this * m; }
	Mat3 &operator+=(const Mat3 &m) { return *this = *this + m; }
	Mat3 &operator-=(const Mat3 &m) { return *this = *this - m; }

	operator double*() { return mat; }
	operator const double*() const { return mat; }

	double &operator[](int i) { return mat[i]; }
	const double operator[](int i) const { return mat[i]; }

	Mat3 transpose() const {
		Mat3 ret;
		ret[0] = mat[0]; ret[3] = mat[1]; ret[6] = mat[2];
		ret[1] = mat[3]; ret[4] = mat[4]; ret[7] = mat[5];
		ret[2] = mat[6]; ret[5] = mat[7]; ret[8] = mat[8];
		return ret;
	}
	double det() const {
		double det;
		det = mat[0] * mat[4] * mat[8];
		det += mat[3] * mat[7] * mat[2];
		det += mat[6] * mat[1] * mat[5];
		det -= mat[6] * mat[4] * mat[2];
		det -= mat[3] * mat[1] * mat[8];
		det -= mat[0] * mat[7] * mat[5];
		return det;
	}
	Mat3 inverse() const {
		Mat3 ret;
		double idet = 1.0 / det();
		ret[0] = (mat[4] * mat[8] - mat[7] * mat[5]) * idet;
		ret[1] = -(mat[1] * mat[8] - mat[7] * mat[2]) * idet;
		ret[2] = (mat[1] * mat[5] - mat[4] * mat[2]) * idet;
		ret[3] = -(mat[3] * mat[8] - mat[6] * mat[5]) * idet;
		ret[4] = (mat[0] * mat[8] - mat[6] * mat[2]) * idet;
		ret[5] = -(mat[0] * mat[5] - mat[3] * mat[2]) * idet;
		ret[6] = (mat[3] * mat[7] - mat[6] * mat[4]) * idet;
		ret[7] = -(mat[0] * mat[7] - mat[6] * mat[1]) * idet;
		ret[8] = (mat[0] * mat[4] - mat[3] * mat[1]) * idet;
		return ret;
	}

	void zero() {
		mat[0] = 0.0; mat[3] = 0.0; mat[6] = 0.0;
		mat[1] = 0.0; mat[4] = 0.0; mat[7] = 0.0;
		mat[2] = 0.0; mat[5] = 0.0; mat[8] = 0.0;
	}
	void identity() {
		mat[0] = 1.0; mat[3] = 0.0; mat[6] = 0.0;
		mat[1] = 0.0; mat[4] = 1.0; mat[7] = 0.0;
		mat[2] = 0.0; mat[5] = 0.0; mat[8] = 1.0;
	}
	void rotate(const Vec3 &axis, double angle) {
		double rad = angle * DEG2RAD;
		double c = cos(rad);
		double s = sin(rad);
		Vec3 v = axis;
		v.normalize();
		double xx = v.x * v.x;
		double yy = v.y * v.y;
		double zz = v.z * v.z;
		double xy = v.x * v.y;
		double yz = v.y * v.z;
		double zx = v.z * v.x;
		double xs = v.x * s;
		double ys = v.y * s;
		double zs = v.z * s;
		mat[0] = (1.0 - c) * xx + c; mat[3] = (1.0 - c) * xy - zs; mat[6] = (1.0 - c) * zx + ys;
		mat[1] = (1.0 - c) * xy + zs; mat[4] = (1.0 - c) * yy + c; mat[7] = (1.0 - c) * yz - xs;
		mat[2] = (1.0 - c) * zx - ys; mat[5] = (1.0 - c) * yz + xs; mat[8] = (1.0 - c) * zz + c;
	}
	void rotate(double x, double y, double z, double angle) {
		rotate(Vec3(x, y, z), angle);
	}
	void rotate_x(double angle) {
		double rad = angle * DEG2RAD;
		double c = cos(rad);
		double s = sin(rad);
		mat[0] = 1.0; mat[3] = 0.0; mat[6] = 0.0;
		mat[1] = 0.0; mat[4] = c; mat[7] = -s;
		mat[2] = 0.0; mat[5] = s; mat[8] = c;
	}
	void rotate_y(double angle) {
		double rad = angle * DEG2RAD;
		double c = cos(rad);
		double s = sin(rad);
		mat[0] = c; mat[3] = 0.0; mat[6] = s;
		mat[1] = 0.0; mat[4] = 1.0; mat[7] = 0.0;
		mat[2] = -s; mat[5] = 0.0; mat[8] = c;
	}
	void rotate_z(double angle) {
		double rad = angle * DEG2RAD;
		double c = cos(rad);
		double s = sin(rad);
		mat[0] = c; mat[3] = -s; mat[6] = 0.0;
		mat[1] = s; mat[4] = c; mat[7] = 0.0;
		mat[2] = 0.0; mat[5] = 0.0; mat[8] = 1.0;
	}
	void scale(const Vec3 &v) {
		mat[0] = v.x; mat[3] = 0.0; mat[6] = 0.0;
		mat[1] = 0.0; mat[4] = v.y; mat[7] = 0.0;
		mat[2] = 0.0; mat[5] = 0.0; mat[8] = v.z;
	}
	void scale(double x, double y, double z) {
		scale(Vec3(x, y, z));
	}
	void orthonormalize() {
		Vec3 x(mat[0], mat[1], mat[2]);
		Vec3 y(mat[3], mat[4], mat[5]);
		Vec3 z;
		x.normalize();
		z.cross(x, y);
		z.normalize();
		y.cross(z, x);
		y.normalize();
		mat[0] = x.x; mat[3] = y.x; mat[6] = z.x;
		mat[1] = x.y; mat[4] = y.y; mat[7] = z.y;
		mat[2] = x.z; mat[5] = y.z; mat[8] = z.z;
	}

	double mat[9];
};

/*****************************************************************************/
/*                                                                           */
/* Mat4                                                                      */
/*                                                                           */
/*****************************************************************************/

struct Mat4 {

	Mat4() {
		mat[0] = 1.0; mat[4] = 0.0; mat[8] = 0.0; mat[12] = 0.0;
		mat[1] = 0.0; mat[5] = 1.0; mat[9] = 0.0; mat[13] = 0.0;
		mat[2] = 0.0; mat[6] = 0.0; mat[10] = 1.0; mat[14] = 0.0;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 1.0;
	}
	Mat4(const Vec3 &v) {
		translate(v);
	}
	Mat4(double x, double y, double z) {
		translate(x, y, z);
	}
	Mat4(const Vec3 &axis, double angle) {
		rotate(axis, angle);
	}
	Mat4(double x, double y, double z, double angle) {
		rotate(x, y, z, angle);
	}
	Mat4(const Mat3 &m) {
		mat[0] = m[0]; mat[4] = m[3]; mat[8] = m[6]; mat[12] = 0.0;
		mat[1] = m[1]; mat[5] = m[4]; mat[9] = m[7]; mat[13] = 0.0;
		mat[2] = m[2]; mat[6] = m[5]; mat[10] = m[8]; mat[14] = 0.0;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 1.0;
	}
	Mat4(const double *m) {
		mat[0] = m[0]; mat[4] = m[4]; mat[8] = m[8]; mat[12] = m[12];
		mat[1] = m[1]; mat[5] = m[5]; mat[9] = m[9]; mat[13] = m[13];
		mat[2] = m[2]; mat[6] = m[6]; mat[10] = m[10]; mat[14] = m[14];
		mat[3] = m[3]; mat[7] = m[7]; mat[11] = m[11]; mat[15] = m[15];
	}
	Mat4(const Mat4 &m) {
		mat[0] = m[0]; mat[4] = m[4]; mat[8] = m[8]; mat[12] = m[12];
		mat[1] = m[1]; mat[5] = m[5]; mat[9] = m[9]; mat[13] = m[13];
		mat[2] = m[2]; mat[6] = m[6]; mat[10] = m[10]; mat[14] = m[14];
		mat[3] = m[3]; mat[7] = m[7]; mat[11] = m[11]; mat[15] = m[15];
	}

	Vec3 operator*(const Vec3 &v) const {
		Vec3 ret;
		ret[0] = mat[0] * v[0] + mat[4] * v[1] + mat[8] * v[2] + mat[12];
		ret[1] = mat[1] * v[0] + mat[5] * v[1] + mat[9] * v[2] + mat[13];
		ret[2] = mat[2] * v[0] + mat[6] * v[1] + mat[10] * v[2] + mat[14];
		return ret;
	}
	Vec4 operator*(const Vec4 &v) const {
		Vec4 ret;
		ret[0] = mat[0] * v[0] + mat[4] * v[1] + mat[8] * v[2] + mat[12] * v[3];
		ret[1] = mat[1] * v[0] + mat[5] * v[1] + mat[9] * v[2] + mat[13] * v[3];
		ret[2] = mat[2] * v[0] + mat[6] * v[1] + mat[10] * v[2] + mat[14] * v[3];
		ret[3] = mat[3] * v[0] + mat[7] * v[1] + mat[11] * v[2] + mat[15] * v[3];
		return ret;
	}
	Mat4 operator*(double f) const {
		Mat4 ret;
		ret[0] = mat[0] * f; ret[4] = mat[4] * f; ret[8] = mat[8] * f; ret[12] = mat[12] * f;
		ret[1] = mat[1] * f; ret[5] = mat[5] * f; ret[9] = mat[9] * f; ret[13] = mat[13] * f;
		ret[2] = mat[2] * f; ret[6] = mat[6] * f; ret[10] = mat[10] * f; ret[14] = mat[14] * f;
		ret[3] = mat[3] * f; ret[7] = mat[7] * f; ret[11] = mat[11] * f; ret[15] = mat[15] * f;
		return ret;
	}
	Mat4 operator*(const Mat4 &m) const {
		Mat4 ret;
		ret[0] = mat[0] * m[0] + mat[4] * m[1] + mat[8] * m[2] + mat[12] * m[3];
		ret[1] = mat[1] * m[0] + mat[5] * m[1] + mat[9] * m[2] + mat[13] * m[3];
		ret[2] = mat[2] * m[0] + mat[6] * m[1] + mat[10] * m[2] + mat[14] * m[3];
		ret[3] = mat[3] * m[0] + mat[7] * m[1] + mat[11] * m[2] + mat[15] * m[3];
		ret[4] = mat[0] * m[4] + mat[4] * m[5] + mat[8] * m[6] + mat[12] * m[7];
		ret[5] = mat[1] * m[4] + mat[5] * m[5] + mat[9] * m[6] + mat[13] * m[7];
		ret[6] = mat[2] * m[4] + mat[6] * m[5] + mat[10] * m[6] + mat[14] * m[7];
		ret[7] = mat[3] * m[4] + mat[7] * m[5] + mat[11] * m[6] + mat[15] * m[7];
		ret[8] = mat[0] * m[8] + mat[4] * m[9] + mat[8] * m[10] + mat[12] * m[11];
		ret[9] = mat[1] * m[8] + mat[5] * m[9] + mat[9] * m[10] + mat[13] * m[11];
		ret[10] = mat[2] * m[8] + mat[6] * m[9] + mat[10] * m[10] + mat[14] * m[11];
		ret[11] = mat[3] * m[8] + mat[7] * m[9] + mat[11] * m[10] + mat[15] * m[11];
		ret[12] = mat[0] * m[12] + mat[4] * m[13] + mat[8] * m[14] + mat[12] * m[15];
		ret[13] = mat[1] * m[12] + mat[5] * m[13] + mat[9] * m[14] + mat[13] * m[15];
		ret[14] = mat[2] * m[12] + mat[6] * m[13] + mat[10] * m[14] + mat[14] * m[15];
		ret[15] = mat[3] * m[12] + mat[7] * m[13] + mat[11] * m[14] + mat[15] * m[15];
		return ret;
	}
	Mat4 operator+(const Mat4 &m) const {
		Mat4 ret;
		ret[0] = mat[0] + m[0]; ret[4] = mat[4] + m[4]; ret[8] = mat[8] + m[8]; ret[12] = mat[12] + m[12];
		ret[1] = mat[1] + m[1]; ret[5] = mat[5] + m[5]; ret[9] = mat[9] + m[9]; ret[13] = mat[13] + m[13];
		ret[2] = mat[2] + m[2]; ret[6] = mat[6] + m[6]; ret[10] = mat[10] + m[10]; ret[14] = mat[14] + m[14];
		ret[3] = mat[3] + m[3]; ret[7] = mat[7] + m[7]; ret[11] = mat[11] + m[11]; ret[15] = mat[15] + m[15];
		return ret;
	}
	Mat4 operator-(const Mat4 &m) const {
		Mat4 ret;
		ret[0] = mat[0] - m[0]; ret[4] = mat[4] - m[4]; ret[8] = mat[8] - m[8]; ret[12] = mat[12] - m[12];
		ret[1] = mat[1] - m[1]; ret[5] = mat[5] - m[5]; ret[9] = mat[9] - m[9]; ret[13] = mat[13] - m[13];
		ret[2] = mat[2] - m[2]; ret[6] = mat[6] - m[6]; ret[10] = mat[10] - m[10]; ret[14] = mat[14] - m[14];
		ret[3] = mat[3] - m[3]; ret[7] = mat[7] - m[7]; ret[11] = mat[11] - m[11]; ret[15] = mat[15] - m[15];
		return ret;
	}

	Mat4 &operator*=(double f) { return *this = *this * f; }
	Mat4 &operator*=(const Mat4 &m) { return *this = *this * m; }
	Mat4 &operator+=(const Mat4 &m) { return *this = *this + m; }
	Mat4 &operator-=(const Mat4 &m) { return *this = *this - m; }

	operator double*() { return mat; }
	operator const double*() const { return mat; }

	double &operator[](int i) { return mat[i]; }
	const double operator[](int i) const { return mat[i]; }

	Mat4 rotation() const {
		Mat4 ret;
		ret[0] = mat[0]; ret[4] = mat[4]; ret[8] = mat[8]; ret[12] = 0;
		ret[1] = mat[1]; ret[5] = mat[5]; ret[9] = mat[9]; ret[13] = 0;
		ret[2] = mat[2]; ret[6] = mat[6]; ret[10] = mat[10]; ret[14] = 0;
		ret[3] = 0; ret[7] = 0; ret[11] = 0; ret[15] = 1;
		return ret;
	}
	Mat4 transpose() const {
		Mat4 ret;
		ret[0] = mat[0]; ret[4] = mat[1]; ret[8] = mat[2]; ret[12] = mat[3];
		ret[1] = mat[4]; ret[5] = mat[5]; ret[9] = mat[6]; ret[13] = mat[7];
		ret[2] = mat[8]; ret[6] = mat[9]; ret[10] = mat[10]; ret[14] = mat[11];
		ret[3] = mat[12]; ret[7] = mat[13]; ret[11] = mat[14]; ret[15] = mat[15];
		return ret;
	}
	Mat4 transpose_rotation() const {
		Mat4 ret;
		ret[0] = mat[0]; ret[4] = mat[1]; ret[8] = mat[2]; ret[12] = mat[12];
		ret[1] = mat[4]; ret[5] = mat[5]; ret[9] = mat[6]; ret[13] = mat[13];
		ret[2] = mat[8]; ret[6] = mat[9]; ret[10] = mat[10]; ret[14] = mat[14];
		ret[3] = mat[3]; ret[7] = mat[7]; ret[14] = mat[14]; ret[15] = mat[15];
		return ret;
	}

	double det() const {
		double det;
		det = mat[0] * mat[5] * mat[10];
		det += mat[4] * mat[9] * mat[2];
		det += mat[8] * mat[1] * mat[6];
		det -= mat[8] * mat[5] * mat[2];
		det -= mat[4] * mat[1] * mat[10];
		det -= mat[0] * mat[9] * mat[6];
		return det;
	}

	Mat4 inverse() const {
		Mat4 ret;
		double idet = 1.0 / det();
		ret[0] = (mat[5] * mat[10] - mat[9] * mat[6]) * idet;
		ret[1] = -(mat[1] * mat[10] - mat[9] * mat[2]) * idet;
		ret[2] = (mat[1] * mat[6] - mat[5] * mat[2]) * idet;
		ret[3] = 0.0;
		ret[4] = -(mat[4] * mat[10] - mat[8] * mat[6]) * idet;
		ret[5] = (mat[0] * mat[10] - mat[8] * mat[2]) * idet;
		ret[6] = -(mat[0] * mat[6] - mat[4] * mat[2]) * idet;
		ret[7] = 0.0;
		ret[8] = (mat[4] * mat[9] - mat[8] * mat[5]) * idet;
		ret[9] = -(mat[0] * mat[9] - mat[8] * mat[1]) * idet;
		ret[10] = (mat[0] * mat[5] - mat[4] * mat[1]) * idet;
		ret[11] = 0.0;
		ret[12] = -(mat[12] * ret[0] + mat[13] * ret[4] + mat[14] * ret[8]);
		ret[13] = -(mat[12] * ret[1] + mat[13] * ret[5] + mat[14] * ret[9]);
		ret[14] = -(mat[12] * ret[2] + mat[13] * ret[6] + mat[14] * ret[10]);
		ret[15] = 1.0;
		return ret;
	}

	void zero() {
		mat[0] = 0.0; mat[4] = 0.0; mat[8] = 0.0; mat[12] = 0.0;
		mat[1] = 0.0; mat[5] = 0.0; mat[9] = 0.0; mat[13] = 0.0;
		mat[2] = 0.0; mat[6] = 0.0; mat[10] = 0.0; mat[14] = 0.0;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 0.0;
	}
	void identity() {
		mat[0] = 1.0; mat[4] = 0.0; mat[8] = 0.0; mat[12] = 0.0;
		mat[1] = 0.0; mat[5] = 1.0; mat[9] = 0.0; mat[13] = 0.0;
		mat[2] = 0.0; mat[6] = 0.0; mat[10] = 1.0; mat[14] = 0.0;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 1.0;
	}
	void rotate(const Vec3 &axis, double angle) {
		double rad = angle * DEG2RAD;
		double c = cos(rad);
		double s = sin(rad);
		Vec3 v = axis;
		v.normalize();
		double xx = v.x * v.x;
		double yy = v.y * v.y;
		double zz = v.z * v.z;
		double xy = v.x * v.y;
		double yz = v.y * v.z;
		double zx = v.z * v.x;
		double xs = v.x * s;
		double ys = v.y * s;
		double zs = v.z * s;
		mat[0] = (1.0 - c) * xx + c; mat[4] = (1.0 - c) * xy - zs; mat[8] = (1.0 - c) * zx + ys; mat[12] = 0.0;
		mat[1] = (1.0 - c) * xy + zs; mat[5] = (1.0 - c) * yy + c; mat[9] = (1.0 - c) * yz - xs; mat[13] = 0.0;
		mat[2] = (1.0 - c) * zx - ys; mat[6] = (1.0 - c) * yz + xs; mat[10] = (1.0 - c) * zz + c; mat[14] = 0.0;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 1.0;
	}
	void rotate(double x, double y, double z, double angle) {
		rotate(Vec3(x, y, z), angle);
	}
	void rotate_x(double angle) {
		double rad = angle * DEG2RAD;
		double c = cos(rad);
		double s = sin(rad);
		mat[0] = 1.0; mat[4] = 0.0; mat[8] = 0.0; mat[12] = 0.0;
		mat[1] = 0.0; mat[5] = c; mat[9] = -s; mat[13] = 0.0;
		mat[2] = 0.0; mat[6] = s; mat[10] = c; mat[14] = 0.0;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 1.0;
	}
	void rotate_y(double angle) {
		double rad = angle * DEG2RAD;
		double c = cos(rad);
		double s = sin(rad);
		mat[0] = c; mat[4] = 0.0; mat[8] = s; mat[12] = 0.0;
		mat[1] = 0.0; mat[5] = 1.0; mat[9] = 0.0; mat[13] = 0.0;
		mat[2] = -s; mat[6] = 0.0; mat[10] = c; mat[14] = 0.0;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 1.0;
	}
	void rotate_z(double angle) {
		double rad = angle * DEG2RAD;
		double c = cos(rad);
		double s = sin(rad);
		mat[0] = c; mat[4] = -s; mat[8] = 0.0; mat[12] = 0.0;
		mat[1] = s; mat[5] = c; mat[9] = 0.0; mat[13] = 0.0;
		mat[2] = 0.0; mat[6] = 0.0; mat[10] = 1.0; mat[14] = 0.0;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 1.0;
	}
	void scale(const Vec3 &v) {
		mat[0] = v.x; mat[4] = 0.0; mat[8] = 0.0; mat[12] = 0.0;
		mat[1] = 0.0; mat[5] = v.y; mat[9] = 0.0; mat[13] = 0.0;
		mat[2] = 0.0; mat[6] = 0.0; mat[10] = v.z; mat[14] = 0.0;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 1.0;
	}
	void scale(double x, double y, double z) {
		scale(Vec3(x, y, z));
	}
	void translate(const Vec3 &v) {
		mat[0] = 1.0; mat[4] = 0.0; mat[8] = 0.0; mat[12] = v.x;
		mat[1] = 0.0; mat[5] = 1.0; mat[9] = 0.0; mat[13] = v.y;
		mat[2] = 0.0; mat[6] = 0.0; mat[10] = 1.0; mat[14] = v.z;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 1.0;
	}
	void translate(double x, double y, double z) {
		translate(Vec3(x, y, z));
	}
	void reflect(const Vec4 &plane) {
		double x = plane.x;
		double y = plane.y;
		double z = plane.z;
		double x2 = x * 2.0f;
		double y2 = y * 2.0f;
		double z2 = z * 2.0f;
		mat[0] = 1.0 - x * x2; mat[4] = -y * x2; mat[8] = -z * x2; mat[12] = -plane.w * x2;
		mat[1] = -x * y2; mat[5] = 1.0 - y * y2; mat[9] = -z * y2; mat[13] = -plane.w * y2;
		mat[2] = -x * z2; mat[6] = -y * z2; mat[10] = 1.0 - z * z2; mat[14] = -plane.w * z2;
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = 0.0; mat[15] = 1.0;
	}
	void reflect(double x, double y, double z, double w) {
		reflect(Vec4(x, y, z, w));
	}

	void perspective(double fov, double aspect, double znear, double zfar) {
		double y = tan(fov * PI / 360.0f);
		double x = y * aspect;
		mat[0] = 1.0 / x; mat[4] = 0.0; mat[8] = 0.0; mat[12] = 0.0;
		mat[1] = 0.0; mat[5] = 1.0 / y; mat[9] = 0.0; mat[13] = 0.0;
		mat[2] = 0.0; mat[6] = 0.0; mat[10] = -(zfar + znear) / (zfar - znear); mat[14] = -(2.0f * zfar * znear) / (zfar - znear);
		mat[3] = 0.0; mat[7] = 0.0; mat[11] = -1.0; mat[15] = 0.0;
	}
	void look_at(const Vec3 &eye, const Vec3 &dir, const Vec3 &up) {
		Vec3 x, y, z;
		Mat4 m0, m1;
		z = eye - dir;
		z.normalize();
		x.cross(up, z);
		x.normalize();
		y.cross(z, x);
		y.normalize();
		m0[0] = x.x; m0[4] = x.y; m0[8] = x.z; m0[12] = 0.0;
		m0[1] = y.x; m0[5] = y.y; m0[9] = y.z; m0[13] = 0.0;
		m0[2] = z.x; m0[6] = z.y; m0[10] = z.z; m0[14] = 0.0;
		m0[3] = 0.0; m0[7] = 0.0; m0[11] = 0.0; m0[15] = 1.0;
		m1.translate(-eye);
		*this = m0 * m1;
	}
	void look_at(const double *eye, const double *dir, const double *up) {
		look_at(Vec3(eye), Vec3(dir), Vec3(up));
	}

	double mat[16];
};

inline Mat3::Mat3(const Mat4 &m) {
	mat[0] = m[0]; mat[3] = m[4]; mat[6] = m[8];
	mat[1] = m[1]; mat[4] = m[5]; mat[7] = m[9];
	mat[2] = m[2]; mat[5] = m[6]; mat[8] = m[10];
}

/*****************************************************************************/
/*                                                                           */
/* quat                                                                      */
/*                                                                           */
/*****************************************************************************/

struct quat {

	quat() : x(0), y(0), z(0), w(1) { }
	quat(const Vec3 &dir, double angle) {
		set(dir, angle);
	}
	quat(double x, double y, double z, double angle) {
		set(x, y, z, angle);
	}
	quat(const Mat3 &m) {
		double trace = m[0] + m[4] + m[8];
		if (trace > 0.0) {
			double s = sqrt(trace + 1.0);
			q[3] = 0.5f * s;
			s = 0.5f / s;
			q[0] = (m[5] - m[7]) * s;
			q[1] = (m[6] - m[2]) * s;
			q[2] = (m[1] - m[3]) * s;
		}
		else {
			static int next[3] = { 1, 2, 0 };
			int i = 0;
			if (m[4] > m[0]) i = 1;
			if (m[8] > m[3 * i + i]) i = 2;
			int j = next[i];
			int k = next[j];
			double s = sqrt(m[3 * i + i] - m[3 * j + j] - m[3 * k + k] + 1.0);
			q[i] = 0.5f * s;
			if (s != 0) s = 0.5f / s;
			q[3] = (m[3 * j + k] - m[3 * k + j]) * s;
			q[j] = (m[3 * i + j] + m[3 * j + i]) * s;
			q[k] = (m[3 * i + k] + m[3 * k + i]) * s;
		}
	}

	operator double*() { return (double*)&x; }
	operator const double*() const { return (double*)&x; }

	double &operator[](int i) { return ((double*)&x)[i]; }
	const double operator[](int i) const { return ((double*)&x)[i]; }

	quat operator*(const quat &q) const {
		quat ret;
		ret.x = w * q.x + x * q.x + y * q.z - z * q.y;
		ret.y = w * q.y + y * q.w + z * q.x - x * q.z;
		ret.z = w * q.z + z * q.w + x * q.y - y * q.x;
		ret.w = w * q.w - x * q.x - y * q.y - z * q.z;
		return ret;
	}

	void set(const Vec3 &dir, double angle) {
		double length = dir.magnitude();
		if (length != 0.0) {
			length = 1.0 / length;
			double sinangle = sin(angle * DEG2RAD / 2.0f);
			x = dir[0] * length * sinangle;
			y = dir[1] * length * sinangle;
			z = dir[2] * length * sinangle;
			w = cos(angle * DEG2RAD / 2.0f);
		}
		else {
			x = y = z = 0.0;
			w = 1.0;
		}
	}
	void set(double x, double y, double z, double angle) {
		set(Vec3(x, y, z), angle);
	}

	void slerp(const quat &q0, const quat &q1, double t) {
		double k0, k1, cosomega = q0.x * q1.x + q0.y * q1.y + q0.z * q1.z + q0.w * q1.w;
		quat q;
		if (cosomega < 0.0) {
			cosomega = -cosomega;
			q.x = -q1.x;
			q.y = -q1.y;
			q.z = -q1.z;
			q.w = -q1.w;
		}
		else {
			q.x = q1.x;
			q.y = q1.y;
			q.z = q1.z;
			q.w = q1.w;
		}
		if (1.0 - cosomega > 1e-6) {
			double omega = acos(cosomega);
			double sinomega = sin(omega);
			k0 = sin((1.0 - t) * omega) / sinomega;
			k1 = sin(t * omega) / sinomega;
		}
		else {
			k0 = 1.0 - t;
			k1 = t;
		}
		x = q0.x * k0 + q.x * k1;
		y = q0.y * k0 + q.y * k1;
		z = q0.z * k0 + q.z * k1;
		w = q0.w * k0 + q.w * k1;
	}

	Mat3 to_matrix() const {
		Mat3 ret;
		double x2 = x + x;
		double y2 = y + y;
		double z2 = z + z;
		double xx = x * x2;
		double yy = y * y2;
		double zz = z * z2;
		double xy = x * y2;
		double yz = y * z2;
		double xz = z * x2;
		double wx = w * x2;
		double wy = w * y2;
		double wz = w * z2;
		ret[0] = 1.0 - (yy + zz); ret[3] = xy - wz; ret[6] = xz + wy;
		ret[1] = xy + wz; ret[4] = 1.0 - (xx + zz); ret[7] = yz - wx;
		ret[2] = xz - wy; ret[5] = yz + wx; ret[8] = 1.0 - (xx + yy);
		return ret;
	}

	union {
		struct {
			double x, y, z, w;
		};
		double q[4];
	};
};