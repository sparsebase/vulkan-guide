#include <glm/matrix.hpp>

class Frustum
{
public:
	Frustum() {}

	// m = ProjectionMatrix * ViewMatrix 
	Frustum(glm::mat4 m);

	// http://iquilezles.org/www/articles/frustumcorrect/frustumcorrect.htm
	bool IsBoxVisible(const glm::vec3& minp, const glm::vec3& maxp) const;

private:
	enum Planes
	{
		Left = 0,
		Right,
		Bottom,
		Top,
		Near,
		Far,
		Count,
		Combinations = Count * (Count - 1) / 2
	};

	template<Planes i, Planes j>
	struct ij2k
	{
		enum { k = i * (9 - i) / 2 + j - 1 };
	};

	template<Planes a, Planes b, Planes c>
	glm::vec3 intersection(const glm::vec3* crosses) const;

	glm::vec4   planes_[Count];
	glm::vec3   points_[8];
};

inline Frustum::Frustum(glm::mat4 m)
{
	m = glm::transpose(m);
	planes_[Left] = m[3] + m[0];
	planes_[Right] = m[3] - m[0];
	planes_[Bottom] = m[3] + m[1];
	planes_[Top] = m[3] - m[1];
	planes_[Near] = m[3] + m[2];
	planes_[Far] = m[3] - m[2];

	glm::vec3 crosses[Combinations] = {
		glm::cross(glm::vec3(planes_[Left]),   glm::vec3(planes_[Right])),
		glm::cross(glm::vec3(planes_[Left]),   glm::vec3(planes_[Bottom])),
		glm::cross(glm::vec3(planes_[Left]),   glm::vec3(planes_[Top])),
		glm::cross(glm::vec3(planes_[Left]),   glm::vec3(planes_[Near])),
		glm::cross(glm::vec3(planes_[Left]),   glm::vec3(planes_[Far])),
		glm::cross(glm::vec3(planes_[Right]),  glm::vec3(planes_[Bottom])),
		glm::cross(glm::vec3(planes_[Right]),  glm::vec3(planes_[Top])),
		glm::cross(glm::vec3(planes_[Right]),  glm::vec3(planes_[Near])),
		glm::cross(glm::vec3(planes_[Right]),  glm::vec3(planes_[Far])),
		glm::cross(glm::vec3(planes_[Bottom]), glm::vec3(planes_[Top])),
		glm::cross(glm::vec3(planes_[Bottom]), glm::vec3(planes_[Near])),
		glm::cross(glm::vec3(planes_[Bottom]), glm::vec3(planes_[Far])),
		glm::cross(glm::vec3(planes_[Top]),    glm::vec3(planes_[Near])),
		glm::cross(glm::vec3(planes_[Top]),    glm::vec3(planes_[Far])),
		glm::cross(glm::vec3(planes_[Near]),   glm::vec3(planes_[Far]))
	};

	points_[0] = intersection<Left, Bottom, Near>(crosses);
	points_[1] = intersection<Left, Top, Near>(crosses);
	points_[2] = intersection<Right, Bottom, Near>(crosses);
	points_[3] = intersection<Right, Top, Near>(crosses);
	points_[4] = intersection<Left, Bottom, Far>(crosses);
	points_[5] = intersection<Left, Top, Far>(crosses);
	points_[6] = intersection<Right, Bottom, Far>(crosses);
	points_[7] = intersection<Right, Top, Far>(crosses);

}

// http://iquilezles.org/www/articles/frustumcorrect/frustumcorrect.htm
inline bool Frustum::IsBoxVisible(const glm::vec3& minp, const glm::vec3& maxp) const
{
	// check box outside/inside of frustum
	for (int i = 0; i < Count; i++)
	{
		if ((glm::dot(planes_[i], glm::vec4(minp.x, minp.y, minp.z, 1.0f)) < 0.0) &&
			(glm::dot(planes_[i], glm::vec4(maxp.x, minp.y, minp.z, 1.0f)) < 0.0) &&
			(glm::dot(planes_[i], glm::vec4(minp.x, maxp.y, minp.z, 1.0f)) < 0.0) &&
			(glm::dot(planes_[i], glm::vec4(maxp.x, maxp.y, minp.z, 1.0f)) < 0.0) &&
			(glm::dot(planes_[i], glm::vec4(minp.x, minp.y, maxp.z, 1.0f)) < 0.0) &&
			(glm::dot(planes_[i], glm::vec4(maxp.x, minp.y, maxp.z, 1.0f)) < 0.0) &&
			(glm::dot(planes_[i], glm::vec4(minp.x, maxp.y, maxp.z, 1.0f)) < 0.0) &&
			(glm::dot(planes_[i], glm::vec4(maxp.x, maxp.y, maxp.z, 1.0f)) < 0.0))
		{
			return false;
		}
	}

	// check frustum outside/inside box
	int out;
	out = 0; for (int i = 0; i < 8; i++) out += ((points_[i].x > maxp.x) ? 1 : 0); if (out == 8) return false;
	out = 0; for (int i = 0; i < 8; i++) out += ((points_[i].x < minp.x) ? 1 : 0); if (out == 8) return false;
	out = 0; for (int i = 0; i < 8; i++) out += ((points_[i].y > maxp.y) ? 1 : 0); if (out == 8) return false;
	out = 0; for (int i = 0; i < 8; i++) out += ((points_[i].y < minp.y) ? 1 : 0); if (out == 8) return false;
	out = 0; for (int i = 0; i < 8; i++) out += ((points_[i].z > maxp.z) ? 1 : 0); if (out == 8) return false;
	out = 0; for (int i = 0; i < 8; i++) out += ((points_[i].z < minp.z) ? 1 : 0); if (out == 8) return false;

	return true;
}

template<Frustum::Planes a, Frustum::Planes b, Frustum::Planes c>
inline glm::vec3 Frustum::intersection(const glm::vec3* crosses) const
{
	float D = glm::dot(glm::vec3(planes_[a]), crosses[ij2k<b, c>::k]);
	glm::vec3 res = glm::mat3(crosses[ij2k<b, c>::k], -crosses[ij2k<a, c>::k], crosses[ij2k<a, b>::k]) *
		glm::vec3(planes_[a].w, planes_[b].w, planes_[c].w);
	return res * (-1.0f / D);
}