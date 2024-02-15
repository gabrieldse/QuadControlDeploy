#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "Interface.hpp"

namespace py = pybind11;

PYBIND11_MODULE(policy,  m) {
  py::class_<MLP_3<132, 12>>(m, "PolicyMLP3")
    .def(py::init<>())
    .def("load", &MLP_3<132, 12>::updateParamFromTxt)
    .def("forward", &MLP_3<132, 12>::forward)
    ;
}

PYBIND11_MODULE(stateEst, m) {
  py::class_<MLP_2<123, 11>>(m, "StateEstMLP2")
    .def(py::init<>())
    .def("load", &MLP_2<123, 11>::updateParamFromTxt)
    .def("forward", &MLP_2<123, 11>::forward)
    ;
}

PYBIND11_MODULE(stateEst3, m) {
  py::class_<MLP_2<123, 3>>(m, "StateEstMLP3")
    .def(py::init<>())
    .def("load", &MLP_2<123, 3>::updateParamFromTxt)
    .def("forward", &MLP_2<123, 3>::forward)
    ;
}

PYBIND11_MODULE(interface, m) {
  py::class_<Interface>(m, "Interface")
    .def(py::init<>())
    .def("initialize", &Interface::initialize)
    .def("forward", &Interface::forward)
    .def("update_observation", &Interface::update_observation)
    .def("get_observation", &Interface::get_observation)
    .def("get_computation_time", &Interface::get_computation_time)
    .def_readwrite("obs_mean_", &Interface::obs_mean_)
    .def_readwrite("obs_var_", &Interface::obs_var_)
    .def_readwrite("P", &Interface::P_)
    .def_readwrite("D", &Interface::D_)
    .def_readwrite("pTarget12", &Interface::pTarget12_)
    .def_readwrite("q_init_", &Interface::q_init_)
    .def_readwrite("vel_command", &Interface::vel_command_)
    .def_readwrite("state_est_obs_", &Interface::state_est_obs_)
    .def_readwrite("obs_", &Interface::obs_)
    .def_readwrite("bound_", &Interface::bound_)
    .def_readwrite("bound_pi_", &Interface::bound_pi_)
    ;
}
