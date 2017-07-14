#include "FixValues.h"

vect rotateToFix(vect p)
{
    vector<Eigen::Vector3d> ps;
    for (size_t i = 0; i < (size_t) p.rows(); i += 3) {
        ps.push_back(p.block(i, 0, 3, 1));
    }

    auto delta = ps[0];
    for (auto& p : ps)
        p -= delta;
    for (size_t i = 0; i < ps.size(); i++) {
        p.block(i * 3, 0, 3, 1) = ps[i];
    }

    LOG_INFO("p after first  transformation = {}", p.transpose());

    if (ps[1].block(1, 0, 2, 1).norm() > 1e-7) {
        Eigen::Vector3d ox = {1, 0, 0};
        Eigen::Vector3d axis = ps[1].cross(ox);
        axis /= axis.norm();
        double angle = atan2(ps[1].cross(ox).norm(), ps[1].dot(ox));
        Eigen::Matrix3d m = Eigen::AngleAxisd(angle, axis).matrix();

        for (auto& p : ps)
            p = m * p;
        for (size_t i = 0; i < ps.size(); i++) {
            p.block(i * 3, 0, 3, 1) = ps[i];
        }
    }

    LOG_INFO("p after second transformation = {}", p.transpose());

    if (abs(ps[2](2)) > 1e-7) {
        Eigen::Vector3d axis = {1, 0, 0};
        double angle = atan2(ps[2](2), ps[2](1));

        LOG_ERROR("{}", angle);

        Eigen::Matrix3d m = Eigen::AngleAxisd(-angle, axis).matrix();

        for (auto& p : ps)
            p = m * p;
        for (size_t i = 0; i < ps.size(); i++) {
            p.block(i * 3, 0, 3, 1) = ps[i];
        }
    }
    LOG_INFO("p after third  transformation = {}", p.transpose());

    for (size_t i = 0; i < 9; i++)
        if (i != 3 && i != 6 && i != 7)
            assert(abs(p(0)) < 1e-7);

    return p;
}

vect moveToFix(vect p)
{
    vector<Eigen::Vector3d> ps;
    for (size_t i = 0; i < (size_t) p.rows(); i += 3) {
        ps.push_back(p.block(i, 0, 3, 1));
    }

    auto delta = ps[0];
    for (auto& p : ps)
        p -= delta;
    for (size_t i = 0; i < ps.size(); i++) {
        p.block(i * 3, 0, 3, 1) = ps[i];
    }

    if (ps[1].block(1, 0, 2, 1).norm() > 1e-7) {
        Eigen::Vector3d ox = {1, 0, 0};
        Eigen::Vector3d axis = ps[1].cross(ox);
        axis /= axis.norm();
        double angle = atan2(ps[1].cross(ox).norm(), ps[1].dot(ox));
        Eigen::Matrix3d m = Eigen::AngleAxisd(angle, axis).matrix();

        for (auto& p : ps)
            p = m * p;
        for (size_t i = 0; i < ps.size(); i++) {
            p.block(i * 3, 0, 3, 1) = ps[i];
        }
    }

    for (size_t i = 0; i < 9; i++)
        assert(i == 3 || abs(p(0)) < 1e-7);

    return p;

//    assert(p.size() % 3 == 0);
//
//    for (size_t i = p.rows(); i; i -= 3)
//        p.block(i - 3, 0, 3, 1) -= p.block(0, 0, 3, 1);
//
//
//    return p;
}

