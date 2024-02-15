"""This class will log 1d array in Nd matrix from device and qualisys object"""
import numpy as np
from datetime import datetime as datetime
from time import time
import pinocchio as pin


class Logger:
    def __init__(self, device=None, qualisys=None, logSize=60e3):
        logSize = np.int(logSize)
        self.logSize = logSize
        nb_motors = 12
        self.k = 0

        # Allocate the data:
        # IMU and actuators:
        self.q_mes = np.zeros([logSize, nb_motors])
        self.q_des = np.zeros([logSize, nb_motors])
        self.P = np.zeros([logSize, nb_motors])
        self.D = np.zeros([logSize, nb_motors])

        self.v_mes = np.zeros([logSize, nb_motors])
        self.torquesFromCurrentMeasurment = np.zeros([logSize, nb_motors])
        self.baseOrientation = np.zeros([logSize, 3])
        self.baseOrientationQuat = np.zeros([logSize, 4])
        self.baseAngularVelocity = np.zeros([logSize, 3])
        self.baseLinearAcceleration = np.zeros([logSize, 3])
        self.baseAccelerometer = np.zeros([logSize, 3])
        self.current = np.zeros(logSize)
        self.voltage = np.zeros(logSize)
        self.energy = np.zeros(logSize)
        self.q_des = np.zeros([logSize, nb_motors])
        self.v_des = np.zeros([logSize, nb_motors])

        # Observation by neural network
        self.observation = np.zeros([logSize, 132])
        self.computation_time = np.zeros(logSize)

        # Motion capture:
        self.mocapPosition = np.zeros([logSize, 3])
        self.mocapVelocity = np.zeros([logSize, 3])
        self.mocapAngularVelocity = np.zeros([logSize, 3])
        self.mocapOrientationMat9 = np.zeros([logSize, 3, 3])
        self.mocapOrientationQuat = np.zeros([logSize, 4])

        # Timestamps
        self.tstamps = np.zeros(logSize)

    def sample(self, device, policy, qdes, obs, ctime, qualisys=None):
        # Logging from the device (data coming from the robot)
        self.q_mes[self.k] = device.joints.positions
        self.q_des[self.k] = qdes
        self.P[self.k] = policy.P
        self.D[self.k] = policy.D
        self.v_mes[self.k] = device.joints.velocities
        self.baseOrientation[self.k] = device.imu.attitude_euler
        self.baseOrientationQuat[self.k] = device.imu.attitude_quaternion
        self.baseAngularVelocity[self.k] = device.imu.gyroscope
        self.baseLinearAcceleration[self.k] = device.imu.linear_acceleration
        self.baseAccelerometer[self.k] = device.imu.accelerometer
        self.torquesFromCurrentMeasurment[self.k] = device.joints.measured_torques
        self.current[self.k] = device.powerboard.current
        self.voltage[self.k] = device.powerboard.voltage
        self.energy[self.k] = device.powerboard.energy

        # Logging observation of neural network
        self.observation[self.k] = obs
        self.computation_time[self.k] = ctime

        # Logging from qualisys (motion -- End of script --capture)
        if qualisys is not None:
            self.mocapPosition[self.k] = qualisys.getPosition()
            self.mocapVelocity[self.k] = qualisys.getVelocity()
            self.mocapAngularVelocity[self.k] = qualisys.getAngularVelocity()
            self.mocapOrientationMat9[self.k] = qualisys.getOrientationMat9()
            self.mocapOrientationQuat[self.k] = qualisys.getOrientationQuat()
        else:  # Logging from PyBullet simulator through fake device
            self.mocapPosition[self.k] = device.baseState[0]
            self.mocapVelocity[self.k] = device.baseVel[0]
            self.mocapAngularVelocity[self.k] = device.baseVel[1]
            self.mocapOrientationMat9[self.k] = device.rot_oMb
            self.mocapOrientationQuat[self.k] = device.baseState[1]

        # Logging timestamp
        self.tstamps[self.k] = time()

        self.k += 1

    def saveAll(self, fileName="dataSensors", suffix=""):
        date_str = datetime.now().strftime("_%Y_%m_%d_%H_%M")

        np.savez_compressed(
            fileName + date_str + suffix + ".npz",
            q_mes=self.q_mes,
            q_des=self.q_des,
            P=self.P,
            D=self.D,
            v_mes=self.v_mes,
            baseOrientation=self.baseOrientation,
            baseOrientationQuat=self.baseOrientationQuat,
            baseAngularVelocity=self.baseAngularVelocity,
            baseLinearAcceleration=self.baseLinearAcceleration,
            baseAccelerometer=self.baseAccelerometer,
            torquesFromCurrentMeasurment=self.torquesFromCurrentMeasurment,
            current=self.current,
            voltage=self.voltage,
            energy=self.energy,
            observation=self.observation,
            computation_time=self.computation_time,
            mocapPosition=self.mocapPosition,
            mocapVelocity=self.mocapVelocity,
            mocapAngularVelocity=self.mocapAngularVelocity,
            mocapOrientationMat9=self.mocapOrientationMat9,
            mocapOrientationQuat=self.mocapOrientationQuat,
            tstamps=self.tstamps,
        )

    def loadAll(self, fileName):
        data = np.load(fileName)

        self.q_mes = data["q_mes"]
        self.q_des = data["q_des"]
        self.P = data["P"]
        self.D = data["D"]
        self.v_mes = data["v_mes"]
        self.baseOrientation = data["baseOrientation"]
        self.baseOrientationQuat = data["baseOrientationQuat"]
        self.baseAngularVelocity = data["baseAngularVelocity"]
        self.baseLinearAcceleration = data["baseLinearAcceleration"]
        self.baseAccelerometer = data["baseAccelerometer"]
        self.torquesFromCurrentMeasurment = data["torquesFromCurrentMeasurment"]
        self.current = data["current"]
        self.voltage = data["voltage"]
        self.energy = data["energy"]
        self.observation = data["observation"]
        self.computation_time = data["computation_time"]
        self.mocapPosition = data["mocapPosition"]
        self.mocapVelocity = data["mocapVelocity"]
        self.mocapAngularVelocity = data["mocapAngularVelocity"]
        self.mocapOrientationMat9 = data["mocapOrientationMat9"]
        self.mocapOrientationQuat = data["mocapOrientationQuat"]
        self.tstamps = data["tstamps"]

    def processMocap(self):
        self.mocap_pos = np.zeros([self.logSize, 3])
        self.mocap_h_v = np.zeros([self.logSize, 3])
        self.mocap_b_w = np.zeros([self.logSize, 3])
        self.mocap_RPY = np.zeros([self.logSize, 3])

        for i in range(self.logSize):
            self.mocap_RPY[i] = pin.rpy.matrixToRpy(
                pin.Quaternion(self.mocapOrientationQuat[i]).toRotationMatrix()
            )

        # Robot world to Mocap initial translation and rotation
        mTo = np.array([self.mocapPosition[0, 0], self.mocapPosition[0, 1], 0.0])
        mRo = pin.rpy.rpyToMatrix(0.0, 0.0, self.mocap_RPY[0, 2])

        for i in range(self.logSize):
            oRb = self.mocapOrientationMat9[i]

            oRh = pin.rpy.rpyToMatrix(
                0.0, 0.0, self.mocap_RPY[i, 2] - self.mocap_RPY[0, 2]
            )

            self.mocap_h_v[i] = (
                oRh.transpose()
                @ mRo.transpose()
                @ self.mocapVelocity[i].reshape((3, 1))
            ).ravel()
            self.mocap_b_w[i] = (
                oRb.transpose() @ self.mocapAngularVelocity[i].reshape((3, 1))
            ).ravel()
            self.mocap_pos[i] = (
                mRo.transpose() @ (self.mocapPosition[i, :] - mTo).reshape((3, 1))
            ).ravel()

    def custom_suptitle(self, name):
        from matplotlib import pyplot as plt

        fig = plt.gcf()
        fig.suptitle(name)
        fig.canvas.manager.set_window_title(name)

    def plotAll(self, dt, replay):
        from matplotlib import pyplot as plt

        print(self.logSize)
        t_range = np.array([k * dt for k in range(self.logSize)])

        self.processMocap()

        index6 = [1, 3, 5, 2, 4, 6]
        index12 = [1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12]

        ####
        # Desired & Measured actuator positions
        ####
        lgd1 = ["HAA", "HFE", "Knee"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            (h1,) = plt.plot(t_range, self.q_des[:, i], color="r", linewidth=3)
            (h2,) = plt.plot(t_range, self.q_mes[:, i], color="b", linewidth=3)
            plt.xlabel("Time [s]")
            plt.ylabel(lgd1[i % 3] + " " + lgd2[int(i / 3)] + " [rad]")
            plt.legend(
                [h1, h2],
                [
                    "Ref " + lgd1[i % 3] + " " + lgd2[int(i / 3)],
                    lgd1[i % 3] + " " + lgd2[int(i / 3)],
                ],
                prop={"size": 8},
            )
        self.custom_suptitle("Actuator positions")

        ####
        # Desired & Measured actuator velocity
        ####
        lgd1 = ["HAA", "HFE", "Knee"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            # h1, = plt.plot(t_range, replay.v[:, i], color='r', linewidth=3)
            (h2,) = plt.plot(t_range, self.v_mes[:, i], color="b", linewidth=3)
            plt.xlabel("Time [s]")
            plt.ylabel(lgd1[i % 3] + " " + lgd2[int(i / 3)] + " [rad]")
            # plt.legend([h1, h2], ["Ref "+lgd1[i % 3]+" "+lgd2[int(i/3)],
            #                      lgd1[i % 3]+" "+lgd2[int(i/3)]], prop={'size': 8})
        self.custom_suptitle("Actuator velocities")

        ####
        # FF torques & FB torques & Sent torques & Meas torques
        ####
        lgd1 = ["HAA", "HFE", "Knee"]
        lgd2 = ["FL", "FR", "HL", "HR"]
        plt.figure()
        for i in range(12):
            if i == 0:
                ax0 = plt.subplot(3, 4, index12[i])
            else:
                plt.subplot(3, 4, index12[i], sharex=ax0)
            """tau_fb = replay.P[:, i] * (replay.q[:, i] - self.q_mes[:, 6]) + \
                replay.D[:, i] * (replay.v[:, i] - self.v_mes[:, i])
            h1, = plt.plot(t_range, replay.FF[:, i] * replay.tau[:, i], "r", linewidth=3)
            h2, = plt.plot(t_range, tau_fb, "b", linewidth=3)
            h3, = plt.plot(t_range, replay.FF[:, i] * replay.tau[:, i] + tau_fb, "g", linewidth=3)"""
            (h4,) = plt.plot(
                t_range[:-1],
                self.torquesFromCurrentMeasurment[1:, i],
                "violet",
                linewidth=3,
                linestyle="--",
            )
            plt.xlabel("Time [s]")
            plt.ylabel(lgd1[i % 3] + " " + lgd2[int(i / 3)] + " [Nm]")
            tmp = lgd1[i % 3] + " " + lgd2[int(i / 3)]
            # plt.legend([h1, h2, h3, h4], ["FF "+tmp, "FB "+tmp, "PD+ "+tmp, "Meas "+tmp], prop={'size': 8})
            plt.ylim([-8.0, 8.0])
        self.custom_suptitle("Torques")

        ####
        # Power supply profile
        ####

        plt.figure()
        for i in range(4):
            if i == 0:
                ax0 = plt.subplot(4, 1, i + 1)
            else:
                plt.subplot(4, 1, i + 1, sharex=ax0)

            if i == 0:
                plt.plot(t_range, self.current[:], linewidth=2)
                plt.ylabel("Bus current [A]")
            elif i == 1:
                plt.plot(t_range, self.voltage[:], linewidth=2)
                plt.ylabel("Bus voltage [V]")
            elif i == 2:
                plt.plot(t_range, self.energy[:], linewidth=2)
                plt.ylabel("Bus energy [J]")
            else:
                power = self.current[:] * self.voltage[:]
                plt.plot(t_range, power, linewidth=2)
                N = 10
                plt.plot(
                    t_range[(N - 1) :],
                    np.convolve(power, np.ones(N) / N, mode="valid"),
                    linewidth=2,
                )
                plt.legend(["Raw", "Averaged 10 ms"])
                plt.ylabel("Bus power [W]")
                plt.xlabel("Time [s]")
        self.custom_suptitle("Energy profiles")

        ####
        # Measured & Reference position and orientation (ideal world frame)
        ####
        lgd = ["Pos X", "Pos Y", "Pos Z", "Roll", "Pitch", "Yaw"]
        plt.figure()
        for i in range(6):
            if i == 0:
                ax0 = plt.subplot(3, 2, index6[i])
            else:
                plt.subplot(3, 2, index6[i], sharex=ax0)

            if i < 3:
                plt.plot(t_range, self.mocap_pos[:, i], "k", linewidth=3)
            else:
                plt.plot(t_range, self.mocap_RPY[:, i - 3], "k", linewidth=3)
            plt.legend(["Motion capture"], prop={"size": 8})
            plt.ylabel(lgd[i])
        self.custom_suptitle("Position and orientation")

        ####
        # Measured & Reference linear and angular velocities (horizontal frame)
        ####
        lgd = [
            "Linear vel X",
            "Linear vel Y",
            "Linear vel Z",
            "Angular vel Roll",
            "Angular vel Pitch",
            "Angular vel Yaw",
        ]
        plt.figure()
        for i in range(6):
            if i == 0:
                ax0 = plt.subplot(3, 2, index6[i])
            else:
                plt.subplot(3, 2, index6[i], sharex=ax0)

            if i < 3:
                plt.plot(t_range, self.observation[:, 3 + i], "r", linewidth=3)
                plt.plot(t_range, self.mocap_h_v[:, i], "k", linewidth=3)
                if i < 2:
                    plt.plot(t_range, self.observation[:, 9 + i], "b", linewidth=3)
                plt.legend(
                    ["Observation", "Motion capture", "Reference"], prop={"size": 8}
                )
            else:
                plt.plot(t_range, self.mocap_b_w[:, i - 3], "k", linewidth=3)
                if i == 5:
                    plt.plot(t_range, self.observation[:, 11], "b", linewidth=3)
                plt.legend(["Motion capture", "Reference"], prop={"size": 8})
            plt.ylabel(lgd[i])
        self.custom_suptitle("Linear and angular velocities")

        ####
        # Duration of each loop
        ####

        plt.figure()
        plt.plot(t_range[:-1], np.diff(self.tstamps))
        self.custom_suptitle("Duration of each loop [s]")

        ####
        # Duration of each loop
        ####

        plt.figure()
        plt.plot(t_range, self.computation_time)
        self.custom_suptitle("Computation time of the neural network [us]")

        ###############################
        # Display all graphs and wait #
        ###############################
        plt.show(block=True)


def analyse_consumption():
    import matplotlib.pyplot as plt
    import numpy as np
    import glob

    files = np.sort(glob.glob("test_policies_2022_08_05/*.npz"))[:-2]

    for file in files:
        data = np.load(file)
        current = data["current"]
        voltage = data["voltage"]
        power = data["current"] * data["voltage"]
        mocap_v = data["mocapVelocity"]
        mocap_oRh = data["mocapOrientationMat9"]
        mocap_h_v = np.zeros(mocap_v.shape)
        for k in range(mocap_v.shape[0]):
            mocap_h_v[k, :] = (
                mocap_oRh[k].transpose() @ mocap_v[k].reshape((-1, 1))
            ).ravel()
        joystick = data["observation"][:, 9]
        mask = joystick > 0.99
        # plt.plot(power)
        # plt.plot(joystick)
        # plt.plot(mask)
        # plt.figure()
        # plt.plot(power[mask])
        # plt.show()
        print(
            "{} \t Average power going {:.3f} m/s : {:.3f} W ".format(
                (file.split("_")[-1]).split(sep=".")[0],
                mocap_h_v[mask, 0].mean(),
                power[mask].mean(),
            )
        )


if __name__ == "__main__":
    import sys
    import os
    from sys import argv

    sys.path.insert(0, os.getcwd())  # adds current directory to python path

    # Data file name to load
    file_name = "dataSensors_2022_08_04_18_07_w3.npz"

    # Retrieve length of log file
    N = np.load(file_name)["tstamps"].shape[0]

    # Create loggers
    logger = Logger(logSize=N)

    # Load data from .npz file
    logger.loadAll(file_name)

    # Call all ploting functions
    logger.plotAll(0.001, None)
